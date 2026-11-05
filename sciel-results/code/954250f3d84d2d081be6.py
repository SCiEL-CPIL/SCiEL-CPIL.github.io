import time
import itertools


def _covers(a, b, q, radius):
    """True if Hamming distance between a and b is <= radius."""
    d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
            if d > radius:
                return False
    return True


def _cover_set(points, radius, q, deadline):
    """Greedy set cover: pick points covering the most uncovered points.
    Returns list of chosen indices."""
    m = len(points)
    # Precompute for each point the list of points it covers (itself + radius neighbors)
    cover = []
    for i in range(m):
        cov = [j for j in range(m) if _covers(points[i], points[j], q, radius)]
        cover.append(cov)
    uncovered = set(range(m))
    chosen = []
    while uncovered:
        if time.time() > deadline:
            break
        # pick point covering most uncovered
        best = -1
        best_cnt = -1
        for i in range(m):
            c = len(uncovered & set(cover[i]))
            if c > best_cnt:
                best_cnt = c
                best = i
        chosen.append(best)
        uncovered -= set(cover[best])
    return chosen


def _cover_set_fast(points, radius, q, deadline):
    """Faster greedy using bitsets for the small subproblem."""
    m = len(points)
    cover_bits = []
    for i in range(m):
        bits = 0
        for j in range(m):
            if _covers(points[i], points[j], q, radius):
                bits |= (1 << j)
        cover_bits.append(bits)
    uncovered = (1 << m) - 1
    chosen = []
    while uncovered:
        if time.time() > deadline:
            break
        best = -1
        best_cnt = -1
        for i in range(m):
            c = bin(uncovered & cover_bits[i]).count('1')
            if c > best_cnt:
                best_cnt = c
                best = i
        chosen.append(best)
        uncovered &= ~cover_bits[best]
    return chosen


def construct(q, n, radius):
    start = time.time()
    deadline = start + 100.0  # safety margin before 120s/150s kill

    # Strategy: product code with per-slice covering.
    # Split n into two parts. For each word of a covering code of length n1,
    # take a covering code of length n2. Total = K(n1)*K(n2).
    # We try to find a good split and use greedy covering for each slice.

    # First, build a covering code for the full space if small enough,
    # otherwise use product of two sub-codes.

    total = q ** n
    if total <= 20000:
        # Direct greedy on full space
        points = [list(w) for w in itertools.product(range(q), repeat=n)]
        idx = _cover_set_fast(points, radius, q, deadline)
        return [points[i] for i in idx]

    # Product approach: split into two halves
    # We want K(n1, r1) * K(n2, r2) where r1+r2 >= radius
    # Simplest: r1 = radius, r2 = 0 for one part? No, r2=0 means perfect.
    # Better: distribute radius. Try splits.

    best_code = None
    best_size = float('inf')

    for n1 in range(1, n):
        n2 = n - n1
        if n1 >= n2:
            break  # avoid duplicates
        # Try different radius splits
        for r1 in range(0, radius + 1):
            r2 = radius - r1
            if r2 < 0:
                continue
            # Build covering code for part 1
            if q ** n1 > 100000:
                continue
            pts1 = [list(w) for w in itertools.product(range(q), repeat=n1)]
            idx1 = _cover_set_fast(pts1, r1, q, deadline)
            code1 = [pts1[i] for i in idx1]
            if time.time() > deadline:
                break

            if q ** n2 > 100000:
                continue
            pts2 = [list(w) for w in itertools.product(range(q), repeat=n2)]
            idx2 = _cover_set_fast(pts2, r2, q, deadline)
            code2 = [pts2[i] for i in idx2]

            size = len(code1) * len(code2)
            if size < best_size:
                best_size = size
                best_code = [c1 + c2 for c1 in code1 for c2 in code2]

    if best_code is not None:
        return best_code

    # Fallback: seed-like tetracode product if applicable
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Last resort: all words
    return [list(w) for w in itertools.product(range(q), repeat=n)]
