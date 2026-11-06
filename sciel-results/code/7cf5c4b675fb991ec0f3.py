"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-start deterministic beam search with incremental energy tracking,
followed by a fast local search (kink moves) with strict time budgeting.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

_TOTAL_DEADLINE = 320.0  # seconds wall-clock (10s safety margin from 330s limit)
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0  # 5s extra safety margin


def _neighbours(point):
    """Yield 6 neighbours on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


class _BeamState:
    """A single beam entry with incremental energy tracking."""
    __slots__ = ['score', 'chain', 'occupied', 'h_positions']

    def __init__(self, score, chain, occupied, h_positions):
        self.score = score  # -contacts (higher is better, i.e., more negative energy)
        self.chain = chain
        self.occupied = occupied  # dict: point -> index
        self.h_positions = h_positions  # set of indices where H residues are


def _beam_growth(sequence, dimension, beam_width, time_limit):
    """Beam search growth with incremental energy. Returns best chain or None."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]

    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_h = {0} if sequence[0] == 'H' else set()

    beam = [_BeamState(0, initial_chain, initial_occupied, initial_h)]

    for i in range(1, n):
        new_beam = []
        residue = sequence[i]

        for state in beam:
            last = state.chain[-1]
            for site in _neighbours(last):
                if site in state.occupied:
                    continue

                # Calculate incremental gain
                gain = 0
                if residue == 'H':
                    for q in _neighbours(site):
                        j = state.occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # Small bonus for free space to encourage exploration
                free = 0
                for q in _neighbours(site):
                    if q not in state.occupied:
                        free += 1

                new_score = state.score + gain + 0.02 * free
                new_chain = state.chain + [site]
                new_occupied = dict(state.occupied)
                new_occupied[site] = i
                new_h = state.h_positions | ({i} if residue == 'H' else set())

                new_beam.append(_BeamState(new_score, new_chain, new_occupied, new_h))

        if not new_beam:
            break

        # Sort by score descending and keep top beam_width
        new_beam.sort(key=lambda x: x.score, reverse=True)
        beam = new_beam[:beam_width]

        # Early exit if time is running low
        if _time_remaining() < 3.0:
            break

    if not beam:
        return None

    best_entry = max(beam, key=lambda x: x.score)
    return best_entry.chain


def _kink_move(chain, rng):
    """Try a kink move: pick two non-adjacent positions and reverse the subchain."""
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(0, n - 4)
    j = rng.randint(i + 2, n - 1)
    segment = chain[i + 1:j + 1]
    new_segment = segment[::-1]

    # Check self-avoidance for the new positions
    occupied = set(chain)
    for p in segment:
        occupied.discard(p)

    valid = True
    for p in new_segment:
        if p in occupied:
            valid = False
            break
    if not valid:
        return None

    # Check connectivity at boundaries
    # chain[i] connects to new_segment[0]
    if abs(new_segment[0][0] - chain[i][0]) + abs(new_segment[0][1] - chain[i][1]) + abs(new_segment[0][2] - chain[i][2]) != 1:
        return None
    # new_segment[-1] connects to chain[j+1]
    if j + 1 < n:
        if abs(new_segment[-1][0] - chain[j + 1][0]) + abs(new_segment[-1][1] - chain[j + 1][1]) + abs(new_segment[-1][2] - chain[j + 1][2]) != 1:
            return None

    new_chain = chain[:i + 1] + new_segment + chain[j + 1:]
    return new_chain


def _local_search(sequence, chain, rng, time_budget):
    """Fast local search with kink moves and greedy acceptance."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    start_t = time.time()
    no_improve_count = 0
    max_no_improve = 500  # stop if no improvement for this many moves

    while time.time() - start_t < time_budget and no_improve_count < max_no_improve:
        new_chain = _kink_move(chain, rng)

        if new_chain is None:
            no_improve_count += 1
            continue

        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy

        if delta_e < 0:
            chain = new_chain
            current_energy = new_energy
            no_improve_count = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            no_improve_count += 1

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence using multi-start beam search + local search."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(sequence)
    rng = random.Random(12345 + n)

    # Allocate time based on sequence length
    if n <= 48:
        num_restarts = 8
        beam_width = 25
        ls_budget = 2.0
    elif n <= 64:
        num_restarts = 6
        beam_width = 20
        ls_budget = 3.0
    elif n <= 96:
        num_restarts = 4
        beam_width = 15
        ls_budget = 4.0
    else:
        num_restarts = 3
        beam_width = 12
        ls_budget = 5.0

    best_chain = None
    best_energy = 0

    for restart in range(num_restarts):
        if _time_remaining() < 10.0:
            break

        # Beam search growth with slightly varied parameters for diversity
        bw = beam_width + (restart % 3) * 2  # vary beam width slightly
        chain = _beam_growth(sequence, dimension, bw, _time_remaining() / 2.0)
        if chain is None:
            continue

        energy = _energy(sequence, chain)

        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)

        # Local search to improve
        ls_time = min(ls_budget, _time_remaining() / 3.0)
        if ls_time > 0.3:
            improved_chain, improved_energy = _local_search(sequence, chain, rng, ls_time)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
