"""HP lattice protein folding: growth + local search with time guard."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy_fast(seq, chain):
    """Compute energy (negative number of non-adjacent H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == "H":
            x, y = p
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == "H":
                    contacts += 1
    return -contacts


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


def _local_search(sequence, chain, rng, max_steps=500):
    """Improve chain via random segment reversals (u-turns)."""
    n = len(chain)
    if n < 4:
        return chain
    best = chain
    best_e = _energy_fast(sequence, best)
    occupied = {p: i for i, p in enumerate(best)}

    for step in range(max_steps):
        if _remaining() < 5.0:
            break
        # Pick a random segment to reverse
        i = rng.randint(1, n - 3)
        length = rng.randint(2, min(15, n - 1 - i))
        j = i + length  # end index (exclusive for the segment to reverse)
        if j >= n:
            continue

        # The segment to reverse is chain[i+1..j]
        # After reversal, chain[i+1] connects to what was chain[j], etc.
        # Check self-avoidance: the new positions must not collide.
        seg = list(chain[i + 1:j + 1])
        seg.reverse()

        # Check: chain[i] -> seg[0] must be unit step
        ci = chain[i]
        s0 = seg[0]
        if abs(ci[0] - s0[0]) + abs(ci[1] - s0[1]) != 1:
            continue
        # Check: seg[-1] -> chain[j+1] must be unit step (if j+1 < n)
        if j + 1 < n:
            cn = chain[j + 1]
            sl = seg[-1]
            if abs(sl[0] - cn[0]) + abs(sl[1] - cn[1]) != 1:
                continue

        # Check self-avoidance within the new segment and against the rest
        # The positions occupied by the segment are the same set, so no collision within.
        # But we need to check that the new adjacency endpoints don't collide with neighbors.
        # Actually the set of positions is unchanged, so self-avoidance is automatically preserved
        # as long as the unit-step constraints hold. Wait - that's not quite right.
        # The positions are the same set, just the order changes. Self-avoidance (no two residues
        # at the same site) is preserved. The unit-step constraint between consecutive residues
        # is what we need to check, which we did above.
        # But we also need to check that the reversed segment doesn't create a contact with
        # itself that violates... no, self-avoidance is just about positions being unique.
        # Since we're permuting the same set of positions, uniqueness is preserved.

        # Build new chain
        new_chain = chain[:i + 1] + seg + chain[j + 1:]

        # Verify unit steps throughout the modified region
        valid = True
        for k in range(i, min(j + 1, n - 1)):
            if abs(new_chain[k][0] - new_chain[k + 1][0]) + abs(new_chain[k][1] - new_chain[k + 1][1]) != 1:
                valid = False
                break
        if not valid:
            continue

        new_e = _energy_fast(sequence, new_chain)
        if new_e < best_e:
            best = new_chain
            best_e = new_e
            occupied = {p: idx for idx, p in enumerate(best)}

    return best


def fold(sequence, dimension):
    """Best conformation using growth + local search with time guard."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    # Phase 1: Grow multiple conformations
    n_grow = 200
    while n_grow > 0 and _remaining() > 20.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            n_grow -= 1
            continue
        value = _energy_fast(sequence, chain)
        if best is None or value < best_energy:
            best = chain
            best_energy = value
        n_grow -= 1

    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy_fast(sequence, best)

    # Phase 2: Local search on the best conformation
    if _remaining() > 10.0:
        best = _local_search(sequence, best, rng, max_steps=2000)
        best_energy = _energy_fast(sequence, best)

    # Phase 3: If time remains, do more growth + local search cycles
    while _remaining() > 15.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        chain = _local_search(sequence, chain, rng, max_steps=200)
        value = _energy_fast(sequence, chain)
        if value < best_energy:
            best = chain
            best_energy = value

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
