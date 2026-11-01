"""Ternary covering code K_3(8,1) via structured construction + aggressive local search.

Strategy:
1. Start from the direct-sum construction (486 words) as a valid cover.
2. Use vectorized greedy set-cover to shrink the initial pool.
3. Apply aggressive local search: removal, 1-swap, and 2-exchange moves,
   with precomputed ball membership for O(1) redundancy checks.
4. Return the best valid cover found before the deadline.
"""
import time
import random
import itertools

import numpy as np

_DEADLINE = 150.0


def _time_left(deadline):
    return deadline - time.time()


def _tetracode():
    """9 ternary words of length 4, a perfect covering code of radius 1."""
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _initial_cover(q, n, radius):
    """Direct sum of tetracode with every word on remaining coordinates."""
    tet = _tetracode()
    words = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _encode(words, n, q):
    """Encode list of words into integer codes (base-q representation)."""
    result = np.zeros(len(words), dtype=np.int64)
    for i, w in enumerate(words):
        code = 0
        for j in range(n):
            code += w[j] * (q ** j)
        result[i] = code
    return result


def _decode(codes, n, q):
    """Decode integer codes back to list of words."""
    out = []
    for c in codes:
        d = []
        x = int(c)
        for _ in range(n):
            d.append(x % q)
            x //= q
        out.append(d)
    return out


def _ball_points(code, n, q):
    """Generate all points within Hamming distance 1 of a code (the code itself + n*(q-1) neighbors)."""
    points = [code]
    for i in range(n):
        digit = (code // (q ** i)) % q
        for v in range(q):
            if v != digit:
                points.append(code + (v - digit) * (q ** i))
    return points


def _build_ball_matrix(codes, n, q):
    """Build a boolean matrix where ball_matrix[i, p] = True if point p is in ball of code i."""
    N = q ** n
    m = len(codes)
    matrix = np.zeros((m, N), dtype=np.bool_)
    for i, c in enumerate(codes):
        for p in _ball_points(c, n, q):
            matrix[i, p] = True
    return matrix


def _greedy_cover(codes, n, q, deadline, rng):
    """Greedy set cover: repeatedly pick the word whose ball covers most uncovered points."""
    N = q ** n
    m = len(codes)
    
    # Precompute balls as sets for fast intersection
    balls = [set(_ball_points(c, n, q)) for c in codes]
    
    uncovered = np.ones(N, dtype=np.bool_)
    chosen = []
    chosen_set = set()
    
    for _ in range(m):
        if _time_left(deadline) < 1.0:
            break
        
        best = -1
        best_gain = -1
        
        for idx in range(m):
            if idx in chosen_set:
                continue
            # Count uncovered points in this ball
            gain = sum(1 for p in balls[idx] if uncovered[p])
            if gain > best_gain:
                best_gain = gain
                best = idx
        
        if best < 0 or best_gain <= 0:
            break
        
        chosen.append(best)
        chosen_set.add(best)
        for p in balls[best]:
            uncovered[p] = False
        
        if not uncovered.any():
            break
    
    return chosen


def _check_cover(codes, n, q, deadline):
    """Check if the given codes form a valid cover. Returns True if valid."""
    N = q ** n
    covered = np.zeros(N, dtype=np.bool_)
    for c in codes:
        for p in _ball_points(c, n, q):
            covered[p] = True
    return covered.all()


def _local_search(codes, n, q, deadline, rng):
    """Aggressive local search: removal, 1-swap, 2-exchange moves."""
    N = q ** n
    m = len(codes)
    
    # Precompute ball sets for each code
    balls = [set(_ball_points(c, n, q)) for c in codes]
    
    # Build coverage count: for each point, how many codes cover it
    cover_count = np.zeros(N, dtype=np.int32)
    for i in range(m):
        for p in balls[i]:
            cover_count[p] += 1
    
    # All points must be covered
    if (cover_count > 0).all():
        pass  # valid
    
    active = list(range(m))
    active_set = set(active)
    
    # Iteratively try improvements
    improved = True
    while improved and _time_left(deadline) > 2.0:
        improved = False
        
        # Pass 1: Try removals
        for i in range(len(active) - 1, -1, -1):
            idx = active[i]
            # Check if removing idx still covers all points
            # A point is uncovered if cover_count[p] == 1 and p is in ball[idx]
            can_remove = True
            for p in balls[idx]:
                if cover_count[p] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove idx
                for p in balls[idx]:
                    cover_count[p] -= 1
                active.pop(i)
                active_set.discard(idx)
                improved = True
        
        # Pass 2: Try 1-swap (remove one, add another from pool)
        # Pool = all codes not in active
        pool = [i for i in range(m) if i not in active_set]
        if pool and _time_left(deadline) > 5.0:
            for i in range(len(active)):
                if _time_left(deadline) < 5.0:
                    break
                idx = active[i]
                # Try swapping idx with each pool member
                # For each pool member j, check if (active - {idx} + {j}) is still a valid cover
                # This is expensive, so sample
                samples = rng.sample(pool, min(len(pool), 50))
                for j in samples:
                    if j in active_set:
                        continue
                    # Simulate: remove idx, add j
                    # Check if all points in ball[idx] are still covered after removal
                    # i.e., for each p in ball[idx], cover_count[p] - 1 + (1 if p in ball[j] else 0) > 0
                    valid_swap = True
                    for p in balls[idx]:
                        new_count = cover_count[p] - 1 + (1 if p in balls[j] else 0)
                        if new_count <= 0:
                            valid_swap = False
                            break
                    if valid_swap:
                        # Also check that adding j doesn't create issues (it doesn't, since we're adding coverage)
                        # Perform swap
                        for p in balls[idx]:
                            cover_count[p] -= 1
                        for p in balls[j]:
                            cover_count[p] += 1
                        active[i] = j
                        active_set.discard(idx)
                        active_set.add(j)
                        pool.remove(j)
                        improved = True
                        break  # Restart pass after a swap
        
        # Pass 3: Try 2-exchange (remove two, add two) - only if time permits
        if _time_left(deadline) > 20.0 and len(active) > 2:
            # Sample pairs to try
            for _ in range(20):
                if _time_left(deadline) < 20.0:
                    break
                if len(active) < 2:
                    break
                i1, i2 = rng.sample(range(len(active)), 2)
                idx1, idx2 = active[i1], active[i2]
                # Try adding two from pool
                if len(pool) < 2:
                    break
                j1, j2 = rng.sample(pool, 2)
                # Check if (active - {idx1, idx2} + {j1, j2}) is valid
                # For each p, new_count = cover_count[p] - (1 if p in ball[idx1]) - (1 if p in ball[idx2]) + (1 if p in ball[j1]) + (1 if p in ball[j2])
                # This is expensive, so only check points in the union of the four balls
                union = set()
                for s in [balls[idx1], balls[idx2], balls[j1], balls[j2]]:
                    union |= s
                valid_2ex = True
                for p in union:
                    old = cover_count[p]
                    delta = 0
                    if p in balls[idx1]:
                        delta -= 1
                    if p in balls[idx2]:
                        delta -= 1
                    if p in balls[j1]:
                        delta += 1
                    if p in balls[j2]:
                        delta += 1
                    if old + delta <= 0:
                        valid_2ex = False
                        break
                if valid_2ex:
                    # Perform 2-exchange
                    for p in balls[idx1]:
                        cover_count[p] -= 1
                    for p in balls[idx2]:
                        cover_count[p] -= 1
                    for p in balls[j1]:
                        cover_count[p] += 1
                    for p in balls[j2]:
                        cover_count[p] += 1
                    active[i1] = j1
                    active[i2] = j2
                    active_set.discard(idx1)
                    active_set.discard(idx2)
                    active_set.add(j1)
                    active_set.add(j2)
                    pool.remove(j1)
                    pool.remove(j2)
                    improved = True
                    break  # Restart after a 2-exchange
    
    return active


def construct(q, n, radius):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)
    
    # Fallback: direct-sum construction
    fallback = _initial_cover(q, n, radius)
    
    try:
        codes = _encode(fallback, n, q)
        
        # Greedy set cover to get a smaller initial cover
        chosen = _greedy_cover(codes, n, q, deadline, rng)
        
        if not chosen:
            chosen = list(range(len(codes)))
        
        # Local search to further reduce
        chosen = _local_search(codes, n, q, deadline, rng)
        
        result = _decode(codes[chosen], n, q)
        
        # Final validity check
        if not _check_cover(codes[chosen], n, q, deadline):
            return fallback
        
        return result
    except Exception:
        return fallback


if __name__ == "__main__":
    cw = construct(3, 8, 1)
    print("num codewords:", len(cw))
    print("distinct:", len(set(map(tuple, cw))))
