"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return a normalized key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    if dy == 0:
        return ('h', y1)
    # Reduce to gcd
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize direction: make dx positive, or if dx==0 make dy positive
    if dx < 0:
        dx = -dx
        dy = -dy
    elif dx == 0 and dy < 0:
        dy = -dy
    # Compute intercept: y = (dy/dx)x + b => b = y1 - (dy/dx)x1
    # Use fraction to avoid floating point issues
    # b = (y1*dx - dy*x1) / dx
    num = y1 * dx - dy * x1
    den = dx
    g2 = math.gcd(abs(num), abs(den))
    if g2 > 0:
        num //= g2
        den //= g2
    if den < 0:
        num = -num
        den = -den
    return ('s', dx, dy, num, den)

def _collinear(p1, p2, p3):
    """Check if three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if not _collinear(points[i], points[j], points[k]):
                    continue
                # Check if any other point is on this line
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    if _collinear(points[i], points[j], points[l]):
                        return False
    return True

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_independent_set_bnb(points, timeout=10.0):
    """
    Find the size of the largest subset with no three collinear.
    Uses branch and bound.
    """
    n = len(points)
    # Precompute which triples are collinear
    collinear = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    collinear[(i, j, k)] = True

    # For each point, precompute which other points form a collinear triple with it
    # i.e., for point i, and another point j, which k's would make (i,j,k) collinear
    conflict = [set() for _ in range(n)]
    for (i, j, k) in collinear:
        conflict[i].add(j)
        conflict[i].add(k)
        conflict[j].add(i)
        conflict[j].add(k)
        conflict[k].add(i)
        conflict[k].add(j)

    best = [0]
    start_time = time.time()

    def bnb(idx, chosen, current_size):
        if time.time() - start_time > timeout:
            return
        if current_size > best[0]:
            best[0] = current_size
        if idx >= n:
            return
        # Upper bound: current_size + (n - idx)
        if current_size + (n - idx) <= best[0]:
            return

        # Try not including point idx
        bnb(idx + 1, chosen, current_size)

        # Try including point idx
        # Check if adding idx creates any collinear triple
        can_add = True
        for j in chosen:
            if j in conflict[idx]:
                # There exists some k such that (idx, j, k) is collinear
                # But we need to check if k is also in chosen
                # Actually, conflict[idx] contains all points j such that there exists
                # some k with (idx, j, k) collinear. But we need to check if the specific
                # triple is formed.
                # Let's re-check: for each j in chosen, is there a k in chosen such that
                # (idx, j, k) is collinear?
                pass

        # More careful check
        for j in chosen:
            for k in chosen:
                if j >= k:
                    continue
                if (min(idx, j, k), min(max(idx, j), max(idx, k)), max(idx, j, k)) in collinear:
                    can_add = False
                    break
            if not can_add:
                break

        if can_add:
            chosen.append(idx)
            bnb(idx + 1, chosen, current_size + 1)
            chosen.pop()

    bnb(0, [], 0)
    return best[0]

def _local_search(points, max_iters=5000, timeout=60.0):
    """
    Local search: try swapping one point with a random point from a pool.
    Accept if no four collinear and score (largest no-three-collinear subset) is reduced.
    """
    n = 20
    start_time = time.time()
    current_points = [list(p) for p in points]

    # Compute initial score
    current_score = _max_independent_set_bnb(current_points, timeout=5.0)

    # Pool of candidate points
    pool = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            pool.append([x, y])
    random.shuffle(pool)

    best_points = [list(p) for p in current_points]
    best_score = current_score

    pool_idx = 0
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break

        # Pick a random point to replace
        i = random.randint(0, n - 1)
        # Pick a random candidate from pool
        candidate = pool[pool_idx % len(pool)]
        pool_idx += 1

        # Skip if candidate is already in current_points
        if candidate in current_points:
            continue

        new_points = [list(p) for p in current_points]
        new_points[i] = list(candidate)

        # Check no four collinear
        if not _no_four_collinear(new_points):
            continue

        # Compute score
        new_score = _max_independent_set_bnb(new_points, timeout=5.0)

        if new_score < best_score:
            best_score = new_score
            best_points = new_points
            current_points = new_points
            current_score = new_score

    return best_points, best_score

def _greedy_construct():
    """
    Greedy construction: build points one by one, preferring positions that
    create many collinear triples while maintaining no-four-collinear.
    """
    n = 20
    points = []
    used = set()

    # Start with a few points that are well-spread
    # Use a grid-based approach
    candidates = []
    for x in range(-50, 51, 5):
        for y in range(-50, 51, 5):
            candidates.append([x, y])
    random.shuffle(candidates)

    for cand in candidates:
        if len(points) >= n:
            break
        if tuple(cand) in used:
            continue

        test_points = points + [cand]
        if not _no_four_collinear(test_points):
            continue

        # Count collinear triples in test_points
        triples = _count_collinear_triples(test_points)

        # Accept if it increases or maintains triple count
        if not points:
            points.append(cand)
            used.add(tuple(cand))
        else:
            old_triples = _count_collinear_triples(points)
            if triples >= old_triples:
                points.append(cand)
                used.add(tuple(cand))

    # If we don't have enough, fill with random
    if len(points) < n:
        for x in range(-100, 101):
            if len(points) >= n:
                break
            for y in range(-100, 101):
                if len(points) >= n:
                    break
                cand = [x, y]
                if tuple(cand) in used:
                    continue
                test_points = points + [cand]
                if _no_four_collinear(test_points):
                    points.append(cand)
                    used.add(tuple(cand))

    return points

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    random.seed(42)

    # Phase 1: Greedy construction
    points = _greedy_construct()

    # Phase 2: Local search with multiple restarts
    best_points = [list(p) for p in points]
    best_score = _max_independent_set_bnb(best_points, timeout=10.0)

    while time.time() - start_time < deadline:
        # Try a few restarts with different random seeds
        for restart in range(3):
            if time.time() - start_time > deadline:
                break

            # Perturb the current best
            perturbed = [list(p) for p in best_points]
            for _ in range(random.randint(1, 3)):
                i = random.randint(0, 19)
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                cand = [new_x, new_y]
                if cand in perturbed:
                    continue
                perturbed[i] = cand

            if not _no_four_collinear(perturbed):
                continue

            # Local search from perturbed
            ls_points, ls_score = _local_search(perturbed, max_iters=2000, timeout=20.0)

            if ls_score < best_score:
                best_score = ls_score
                best_points = ls_points

    return best_points

# EVOLVE-BLOCK-END
