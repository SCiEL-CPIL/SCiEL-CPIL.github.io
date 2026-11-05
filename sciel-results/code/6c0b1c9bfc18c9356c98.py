"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90  # Use 90% of 330s budget for safety margin


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE


def _is_deadline_exceeded():
    return time.time() > _get_deadline()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, dimension, rng, time_budget=0.5):
    """Improve a conformation using Monte Carlo crank moves (pivot rotations).

    A crank move picks a pivot index, rotates a sub-chain segment by one
    lattice step, and accepts if the move is non-colliding and lowers energy.
    This escapes local minima of the greedy growth.
    """
    n = len(chain)
    if n < 5:
        return chain, _energy(sequence, chain)

    chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(chain)}
    start = time.time()
    max_time = start + time_budget

    def _is_valid(positions, skip):
        """Check if a list of positions is self-avoiding, skipping 'skip' set of indices."""
        seen = set()
        for i, p in enumerate(positions):
            if i in skip:
                continue
            if p in seen:
                return False
            seen.add(p)
        return True

    for _ in range(20000):
        if time.time() > max_time:
            break
        # Pick a pivot and a segment length
        pivot = rng.randint(1, n - 2)
        max_len = min(n - 1 - pivot, 20)
        seg_len = rng.randint(2, max_len)
        if seg_len > n - 1 - pivot:
            continue

        # The segment is chain[pivot:pivot+seg_len]
        segment = chain[pivot:pivot + seg_len]
        # The pivot's previous point and the segment's last point's next point
        prev_point = chain[pivot - 1]
        tail_point = chain[pivot + seg_len] if pivot + seg_len < n else None

        # Try all 6 rotation directions
        for axis in range(dimension):
            for step in (-1, 1):
                # Rotate segment by one step in (axis, step) direction
                new_segment = []
                valid = True
                for p in segment:
                    np_ = list(p)
                    np_[axis] += step
                    np_ = tuple(np_)
                    new_segment.append(np_)
                # Check the new segment doesn't collide with the rest of the chain
                # and the connection at both ends is unit step
                # Connection at pivot end: new_segment[0] must be adjacent to prev_point
                if abs(new_segment[0][0] - prev_point[0]) + abs(new_segment[0][1] - prev_point[1]) + \
                   abs(new_segment[0][2] - prev_point[2]) != 1:
                    continue
                # Connection at tail end
                if tail_point is not None:
                    if abs(new_segment[-1][0] - tail_point[0]) + abs(new_segment[-1][1] - tail_point[1]) + \
                       abs(new_segment[-1][2] - tail_point[2]) != 1:
                        continue
                # Check collision with the rest of the chain (indices not in segment)
                rest = set()
                for i in range(n):
                    if i < pivot or i >= pivot + seg_len:
                        rest.add(chain[i])
                for np_ in new_segment:
                    if np_ in rest:
                        valid = False
                        break
                if not valid:
                    continue
                # Compute energy change by swapping in the new segment
                old_chain = chain
                chain[pivot:pivot + seg_len] = new_segment
                new_energy = _energy(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    # accept
                else:
                    # reject: restore
                    chain[pivot:pivot + seg_len] = segment

    return chain, best_energy


def fold(sequence, dimension):
    """Best conformation of randomized greedy growths followed by Monte Carlo local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    n = len(sequence)
    # Allocate time per sequence: total budget / 26 sequences, with margin
    per_sequence_time = (330 * _BUDGET_FRACTION) / 26.0
    start_time = time.time()

    # Number of greedy restarts: scale with time available
    # For a 48-residue chain, one greedy growth is fast (~1ms)
    # For a 136-residue chain, one greedy growth is slower
    # Estimate: ~n^2 operations per growth, so time per growth ~ n^2 * c
    # Let's do a fixed number of restarts, then local search
    n_restarts = 100
    if n > 100:
        n_restarts = 50
    elif n > 64:
        n_restarts = 80

    for r in range(n_restarts):
        if _is_deadline_exceeded():
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        # Local search to improve this conformation
        chain_ls, energy_ls = _local_search(sequence, chain, dimension, rng, time_budget=0.3)
        if energy_ls < best_energy:
            best, best_energy = chain_ls, energy_ls

    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
