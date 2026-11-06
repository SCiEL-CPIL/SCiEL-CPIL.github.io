"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START = time.time()
# The evaluator kills after 210s wall-clock; use 195s as a safe deadline.
DEADLINE = 195.0


def _time_left():
    return DEADLINE - (time.time() - _START)


def _grow(sequence, n, rng):
    """Randomized greedy chain growth. Prefer sites with new H-H contacts,
    then free space, then a mild turn preference. Returns a list of (x,y) or None."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    last_dir = (1, 0)  # arbitrary initial direction
    for i in range(1, n):
        best_score = None
        best_pos = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if pos in occupied:
                continue
            # Count new H-H contacts this site would form with earlier H residues
            gain = 0
            if sequence[i] == 'H':
                for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((pos[0] + ox, pos[1] + oy))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space around the candidate site
            free = 0
            for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (pos[0] + ox, pos[1] + oy) not in occupied:
                    free += 1
            # Mild preference to continue straight (reduces tight spirals)
            turn_bonus = 0.05 if (dx, dy) == last_dir else 0.0
            score = gain * 3.0 + free * 0.15 + turn_bonus + rng.random()
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
                last_dir = (dx, dy)
        if best_pos is None:
            return None
        chain.append(best_pos)
        occupied[best_pos] = i
    return chain


def _energy_of(chain, sequence):
    """Compute the energy (negative H-H contact count) of a given chain."""
    n = len(chain)
    where = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + ox, y + oy))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _local_search(chain, sequence, rng, time_budget):
    """Improve a conformation using single, double, and triple-move flips.
    Returns the best chain found."""
    n = len(chain)
    if n < 4:
        return chain

    best = list(chain)
    best_e = _energy_of(best, sequence)
    occupied = {best[i]: i for i in range(n)}
    start = time.time()
    stall = 0
    max_stall = 8

    while time.time() - start < time_budget:
        improved = False
        # Try moves from both ends inward
        indices = list(range(1, n - 1))
        # Randomize order a bit but keep ends somewhat prioritized
        rng.shuffle(indices)

        for i in indices:
            if time.time() - start >= time_budget:
                break
            # --- Single move ---
            x, y = best[i]
            prev = best[i - 1]
            nxt = best[i + 1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new = (x + dx, y + dy)
                if new == prev or new == nxt:
                    continue
                if new in occupied:
                    continue
                # Apply move
                del occupied[(x, y)]
                occupied[new] = i
                best[i] = new
                e = _energy_of(best, sequence)
                if e < best_e:
                    best_e = e
                    improved = True
                    stall = 0
                else:
                    # Revert
                    del occupied[new]
                    occupied[(x, y)] = i
                    best[i] = (x, y)
                if e < best_e:
                    break  # accept and stop trying other single moves for this i

            # --- Double move (180-degree turn) ---
            x, y = best[i]
            px, py = best[i - 1]
            nx, ny = best[i + 1]
            # The two positions that form a 180-degree turn at i
            for new_i, new_ip1 in (
                (px + (px - x), py + (py - y)),   # reflect i-1 through i
                (nx + (nx - x), ny + (ny - y)),   # reflect i+1 through i
            ):
                if (new_i, new_ip1) in occupied:
                    continue
                if new_i == new_ip1:
                    continue
                # Check they don't collide with neighbors of i-1 or i+1
                if new_i in (best[i - 2] if i >= 2 else None, best[i + 2] if i + 2 < n else None):
                    continue
                if new_ip1 in (best[i - 2] if i >= 2 else None, best[i + 2] if i + 2 < n else None):
                    continue
                # Apply
                old_i = best[i]
                old_ip1 = best[i + 1]
                del occupied[old_i]
                del occupied[old_ip1]
                occupied[new_i] = i
                occupied[new_ip1] = i + 1
                best[i] = new_i
                best[i + 1] = new_ip1
                e = _energy_of(best, sequence)
                if e < best_e:
                    best_e = e
                    improved = True
                    stall = 0
                else:
                    # Revert
                    del occupied[new_i]
                    del occupied[new_ip1]
                    occupied[old_i] = i
                    occupied[old_ip1] = i + 1
                    best[i] = old_i
                    best[i + 1] = old_ip1

            # --- Triple move (C-shape / U-turn) ---
            if i + 2 < n:
                x0, y0 = best[i]
                x1, y1 = best[i + 1]
                x2, y2 = best[i + 2]
                # Try moving the three residues to form a C-shape
                # The C-shape has the middle residue at the "back" of the C
                # We try all 4 orientations of the C
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    # The three new positions form a U/C shape
                    # Let's try: i goes to (x1+dx, y1+dy), i+1 stays conceptually, i+2 goes to (x0+dx, y0+dy)
                    # Actually, a simpler approach: try translating the 3-residue segment
                    # to a position where it forms a tight C
                    # For now, try moving i and i+2 while keeping i+1, forming a C
                    new_i = (x1 + dx, y1 + dy)
                    new_i2 = (x0 + dx, y0 + dy)
                    if new_i == new_i2:
                        continue
                    if new_i in occupied or new_i2 in occupied:
                        continue
                    # Check against neighbors
                    if new_i == best[i - 1] or new_i == best[i + 1]:
                        continue
                    if new_i2 == best[i + 1] or new_i2 == (best[i + 3] if i + 3 < n else None):
                        continue
                    # Apply
                    old_i, old_i2 = best[i], best[i + 2]
                    del occupied[old_i]
                    del occupied[old_i2]
                    occupied[new_i] = i
                    occupied[new_i2] = i + 2
                    best[i] = new_i
                    best[i + 2] = new_i2
                    e = _energy_of(best, sequence)
                    if e < best_e:
                        best_e = e
                        improved = True
                        stall = 0
                    else:
                        del occupied[new_i]
                        del occupied[new_i2]
                        occupied[old_i] = i
                        occupied[old_i2] = i + 2
                        best[i] = old_i
                        best[i + 2] = old_i2

        if not improved:
            stall += 1
            if stall >= max_stall:
                break
    return best


def fold(sequence, dimension):
    """Best conformation found within the time budget."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    # Fallback: straight line
    fallback = [[i, 0] for i in range(n)]

    # Determine time budget for this sequence
    # We have 10 sequences total, 195s total. Allocate roughly 15s per sequence
    # but use what's actually left.
    seq_budget = min(15.0, _time_left() / 10.0)
    if _time_left() <= 0:
        return fallback

    rng = random.Random(12345 + n)

    # Phase 1: Generate initial conformations via growth
    num_grow = max(10, min(50, int(seq_budget * 3)))
    best = None
    best_e = 0  # energy is negative, start at 0 (no contacts)

    for _ in range(num_grow):
        if _time_left() <= 1.0:
            break
        chain = _grow(sequence, n, rng)
        if chain is None:
            continue
        e = _energy_of(chain, sequence)
        if best is None or e < best_e:
            best = chain
            best_e = e

    if best is None:
        return fallback

    # Phase 2: Local search on the best conformation
    if _time_left() > 2.0:
        ls_budget = min(8.0, _time_left() * 0.6)
        best = _local_search(best, sequence, rng, ls_budget)
        best_e = _energy_of(best, sequence)

    # Phase 3: If time remains, do a few more grow+local_search cycles
    while _time_left() > 5.0:
        chain = _grow(sequence, n, rng)
        if chain is None:
            continue
        e = _energy_of(chain, sequence)
        if e < best_e:
            best = chain
            best_e = e
            # Quick local search
            if _time_left() > 3.0:
                best = _local_search(best, sequence, rng, min(3.0, _time_left() * 0.5))
                best_e = _energy_of(best, sequence)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
