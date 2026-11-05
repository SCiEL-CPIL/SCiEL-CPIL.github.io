"""HP lattice protein folding: beam search + local refinement."""
import time
import random
import heapq
from collections import defaultdict

_DEADLINE = 310.0  # safety margin before 330s kill
_start_time = None


def _time_left():
    if _start_time is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _start_time)


def _neighbours3(p):
    x, y, z = p
    return ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1))


def _energy(sequence, chain):
    """Compute energy: -1 per non-bonded H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours3(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_fold(sequence, dimension, beam_width=50, max_time=60.0):
    """Beam search: grow chain keeping top-K partial conformations by score."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)] if n == 1 else []

    # Each beam entry: (score, chain, occupied_set)
    # Score = negative contacts so far + small bonus for free space (encourages spread)
    start = (0, 0, 0)
    beams = [(-0.0, [start], {start})]

    for i in range(1, n):
        if _time_left() < 1.0:
            break
        new_beams = []
        residue = sequence[i]
        for score, chain, occupied in beams:
            last = chain[-1]
            candidates = []
            for site in _neighbours3(last):
                if site in occupied:
                    continue
                # Compute incremental gain
                gain = 0.0
                if residue == 'H':
                    for nb in _neighbours3(site):
                        if nb in occupied:
                            idx = chain.index(nb) if nb in chain else -1
                            # Check if occupied position is H and not adjacent in chain
                            pos_in_chain = -1
                            for k, p in enumerate(chain):
                                if p == nb:
                                    pos_in_chain = k
                                    break
                            if pos_in_chain >= 0 and pos_in_chain < i - 1 and sequence[pos_in_chain] == 'H':
                                gain += 1.0
                # Small bonus for sites with more free neighbours (avoid dead ends)
                free = 0
                for nb in _neighbours3(site):
                    if nb not in occupied:
                        free += 1
                total_score = score - gain + 0.05 * free
                candidates.append((total_score, site))
            if not candidates:
                continue
            # Keep best candidate from this beam
            candidates.sort(key=lambda x: x[0])
            best_score, best_site = candidates[0]
            new_chain = chain + [best_site]
            new_occ = occupied | {best_site}
            new_beams.append((best_score, new_chain, new_occ))

        if not new_beams:
            break

        # Prune to beam_width
        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_width]

    if not beams:
        return None

    # Return best full chain
    beams.sort(key=lambda x: x[0])
    best_chain = beams[0][1]
    return best_chain


def _local_refine(sequence, chain, max_steps=2000, seed=42):
    """Monte Carlo local refinement: try single-residue moves and 2-residue moves."""
    n = len(chain)
    if n <= 2:
        return chain
    rng = random.Random(seed)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}

    step = 0
    while step < max_steps and _time_left() > 0.5:
        step += 1
        # Pick a random interior residue to move
        i = rng.randint(1, n - 2)
        old_pos = best_chain[i]
        # Try moving to a random neighbour
        for _ in range(6):
            candidates = list(_neighbours3(old_pos))
            rng.shuffle(candidates)
            new_pos = None
            for cand in candidates:
                if cand in occupied and occupied[cand] != i:
                    continue
                # Check it doesn't create adjacency conflict with neighbors i-1, i+1
                prev_pos = best_chain[i-1]
                next_pos = best_chain[i+1]
                if cand == prev_pos or cand == next_pos:
                    continue
                # Check new position's neighbours don't conflict (for self-avoidance of the moved residue)
                # Actually we just need cand not occupied by another residue
                new_pos = cand
                break
            if new_pos is None:
                continue

            # Tentatively move
            occupied.pop(old_pos)
            occupied[new_pos] = i
            best_chain[i] = new_pos

            new_energy = _energy(sequence, best_chain)
            if new_energy <= best_energy:
                best_energy = new_energy
            else:
                # Revert
                best_chain[i] = old_pos
                occupied.pop(new_pos)
                occupied[old_pos] = i

    return best_chain


def _try_straight_line(sequence, dimension):
    """Fallback: straight line (always valid)."""
    n = len(sequence)
    return [(i, 0, 0) for i in range(n)]


def fold(sequence, dimension):
    """Best conformation via beam search + local refinement."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(sequence)
    if n <= 1:
        return [[0] * dimension] * n

    best_chain = None
    best_energy = 0  # we want most negative

    # Try multiple beam search runs with different widths and seeds
    beam_configs = [
        (80, 0),
        (50, 1),
        (30, 2),
        (120, 3),
    ]

    for bw, seed_offset in beam_configs:
        if _time_left() < 5.0:
            break
        chain = _beam_fold(sequence, dimension, beam_width=bw)
        if chain is None or len(chain) != n:
            continue
        # Local refine
        refined = _local_refine(sequence, chain, max_steps=1500, seed=42 + seed_offset)
        e = _energy(sequence, refined)
        if e < best_energy:
            best_energy = e
            best_chain = refined

    # If beam search failed entirely, try a simple greedy as fallback
    if best_chain is None:
        best_chain = _try_straight_line(sequence, dimension)

    return [list(p) for p in best_chain]
