"""Optimized 20-point configuration minimizing largest 3-free subset."""
import time
import random
from itertools import combinations

DEADLINE = 145.0  # safety margin before 150s kill

def _collinear(p, q, r):
    return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]) == 0

def _build_triple_map(points):
    """Map each point index to set of triple indices containing it."""
    n = len(points)
    triples = []
    point_to_triples = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    tid = len(triples)
                    triples.append((i, j, k))
                    point_to_triples[i].add(tid)
                    point_to_triples[j].add(tid)
                    point_to_triples[k].add(tid)
    return triples, point_to_triples

def _max_3free_subset(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    triples, point_to_triples = _build_triple_map(points)
    if not triples:
        return n
    # Greedy upper bound: color points, count max color
    # Use greedy independent set on the "conflict" structure
    # B&B: try to find large 3-free set
    # State: which points are in set, which are forbidden
    best = [0]
    # Order points by number of triples (most constrained first helps prune)
    order = sorted(range(n), key=lambda i: -len(point_to_triples[i]))
    # For B&B we want to include as many as possible
    # Use recursive: at each step, decide to include or exclude next point in order
    # But that's 2^20 = 1M, feasible with pruning
    # Better: branch on whether to include a point that's in many triples
    
    # Use a different approach: find max independent set in the 3-uniform hypergraph
    # where edges are triples. A 3-free set is an independent set (no edge fully contained).
    # B&B with branching: pick a triple, at least one of its 3 points must be excluded.
    
    # Simple recursive B&B:
    # available = set of points not yet decided
    # in_set = set of points we've decided to include
    # We want to maximize |in_set| such that no triple is fully in in_set
    # Pruning: |in_set| + |available| <= best -> prune
    
    # Convert to bitmasks for speed
    triples_mask = [0] * len(triples)
    for t, (i, j, k) in enumerate(triples):
        triples_mask[t] = (1 << i) | (1 << j) | (1 << k)
    
    all_mask = (1 << n) - 1
    best_size = [0]
    
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    # Greedy initial lower bound
    # Sort points by degree, greedily add if no triple completed
    greedy_set = 0
    point_deg = [0] * n
    for tm in triples_mask:
        for i in range(n):
            if tm & (1 << i):
                point_deg[i] += 1
    for i in sorted(range(n), key=lambda x: -point_deg[x]):
        # Check if adding i completes any triple
        ok = True
        for tm in triples_mask:
            if (tm & (1 << i)) and (tm & greedy_set) == (tm & ~(1 << i)):
                # triple would be fully in greedy_set | {i}
                if (tm & greedy_set) == (tm & ~(1 << i)):
                    ok = False
                    break
        if ok:
            greedy_set |= (1 << i)
    best_size[0] = popcount(greedy_set)
    
    # B&B
    def bb(available, in_set, size):
        if size + popcount(available) <= best_size[0]:
            return
        if available == 0:
            if size > best_size[0]:
                best_size[0] = size
            return
        # Find a triple with all 3 points in available
        # Pick the one that gives best pruning
        # Try to find a triple where all 3 are available
        branch_triple = -1
        for tm in triples_mask:
            if (tm & available) == tm:
                branch_triple = tm
                break
        if branch_triple == -1:
            # No triple fully in available, so we can take all available
            if size + popcount(available) > best_size[0]:
                best_size[0] = size + popcount(available)
            return
        # Branch: for each point in the triple, exclude it
        for bit in [1, 2, 4]:
            if branch_triple & bit:
                # Exclude this point
                p = (bit.bit_length() - 1)
                new_avail = available & ~(1 << p)
                bb(new_avail, in_set, size)
    
    bb(all_mask, 0, 0)
    return best_size[0]

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    for l in range(k+1, n):
                        if _collinear(points[i], points[j], points[l]):
                            return True
    return False

def _construct_grid():
    """Start with a grid-like configuration with many collinear triples."""
    # 4x5 grid with some perturbation to avoid 4 collinear
    # Use a 5x4 grid
    pts = []
    # 5 columns, 4 rows
    for r in range(4):
        for c in range(5):
            x = c * 20 - 40  # -40 to 40
            y = r * 20 - 30  # -30 to 30
            pts.append([x, y])
    # Check for 4 collinear - grid has many!
    # Perturb to break 4-collinear while keeping 3-collinear
    return pts

def _perturb_to_break_4(pts, rng):
    """Perturb points to break all 4-collinear while preserving as many 3-collinear as possible."""
    pts = [p[:] for p in pts]
    for _ in range(200):
        if not _has_four_collinear(pts):
            break
        # Find a 4-collinear set and move one point
        n = len(pts)
        fixed = False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if _collinear(pts[i], pts[j], pts[k]):
                        for l in range(k+1, n):
                            if _collinear(pts[i], pts[j], pts[l]):
                                # Move point l slightly
                                pts[l] = [pts[l][0] + rng.randint(-3, 3),
                                          pts[l][1] + rng.randint(-3, 3)]
                                fixed = True
                                break
                        if fixed:
                            break
                    if fixed:
                        break
                if fixed:
                    break
            if fixed:
                break
        if not fixed:
            break
    return pts

def _local_search(pts, rng, time_limit):
    """Local search: move points to reduce max 3-free subset."""
    pts = [p[:] for p in pts]
    best_score = _max_3free_subset(pts)
    current_score = best_score
    start = time.time()
    
    # Try moving individual points
    candidates = []
    for x in range(-100, 101, 5):
        for y in range(-100, 101, 5):
            candidates.append([x, y])
    
    while time.time() - start < time_limit:
        improved = False
        for _ in range(50):
            i = rng.randrange(20)
            old = pts[i][:]
            # Try a random candidate position
            new_pos = rng.choice(candidates)
            if new_pos == old:
                continue
            # Check no duplicate
            if any(pts[j] == new_pos for j in range(20) if j != i):
                continue
            pts[i] = new_pos[:]
            if _has_four_collinear(pts):
                pts[i] = old
                continue
            score = _max_3free_subset(pts)
            if score < current_score:
                current_score = score
                best_score = score
                improved = True
            else:
                pts[i] = old
        if not improved:
            # Try a larger perturbation
            i = rng.randrange(20)
            pts[i] = [rng.randint(-100, 100), rng.randint(-100, 100)]
            if _has_four_collinear(pts):
                pts[i] = old
            else:
                score = _max_3free_subset(pts)
                if score < current_score:
                    current_score = score
                    best_score = score
    return pts, best_score

def construct():
    rng = random.Random(42)
    start = time.time()
    
    # Start from grid, break 4-collinear
    pts = _construct_grid()
    pts = _perturb_to_break_4(pts, rng)
    
    if _has_four_collinear(pts):
        # Fallback: use parabola
        return [[x, x*x] for x in range(-10, 10)]
    
    best_pts = [p[:] for p in pts]
    best_score = _max_3free_subset(pts)
    
    # Local search
    remaining = DEADLINE - (time.time() - start) - 10
    if remaining > 5:
        pts, score = _local_search(pts, rng, remaining)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in pts]
    
    # Verify
    if _has_four_collinear(best_pts):
        return [[x, x*x] for x in range(-10, 10)]
    
    return best_pts
