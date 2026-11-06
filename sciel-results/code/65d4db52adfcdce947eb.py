"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START
def _is_isosceles(p1, p2, p3):
    """Check if three points form an isosceles triangle."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    d12 = (x1 - x2)**2 + (y1 - y2)**2
    d23 = (x2 - x3)**2 + (y2 - y3)**2
    d13 = (x1 - x3)**2 + (y1 - y3)**2
    return d12 == d23 or d12 == d13 or d23 == d13

def _can_add_point(new_pt, existing_pts):
    """Check if adding new_pt to existing_pts creates no isosceles triangles."""
    for i in range(len(existing_pts)):
        for j in range(i + 1, len(existing_pts)):
            if _is_isosceles(new_pt, existing_pts[i], existing_pts[j]):
                return False
    return True

def _greedy_construct(seed=0):
    """Greedy construction with randomized point ordering."""
    rng = random.Random(seed)
    pts = []
    all_pts = [(x, y) for x in range(N) for y in range(N)]
    rng.shuffle(all_pts)
    
    for pt in all_pts:
        if _can_add_point(pt, pts):
            pts.append(pt)
    
    return pts

def _local_search(pts, time_budget=60.0):
    """Local search to improve the set by swapping points."""
    pts = list(pts)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        rng = random.Random()
        # Try to swap a random point with a random unused point
        for _ in range(100):
            if (time.time() - start) >= time_budget:
                break
            i = rng.randint(0, len(pts) - 1)
            # Pick a random candidate from the grid
            candidate = (rng.randint(0, N - 1), rng.randint(0, N - 1))
            if candidate in pts:
                continue
            old_pt = pts[i]
            # Check if we can swap
            remaining = [p for j, p in enumerate(pts) if j != i]
            if _can_add_point(candidate, remaining):
                pts[i] = candidate
                improved = True
    return pts

def _construct_2d_greedy_with_search():
    """2D greedy with local search."""
    best_pts = []
    best_len = 0
    start = time.time()
    
    for seed in range(20):
        if (time.time() - start) > TIME_LIMIT - 10:
            break
        pts = _greedy_construct(seed)
        if len(pts) > best_len:
            best_pts = pts
            best_len = len(pts)
        # Try local search on the best found so far
        if len(pts) >= best_len:
            pts_ls = _local_search(pts, time_budget=min(30.0, TIME_LIMIT - 15 - (time.time() - start)))
            if len(pts_ls) > best_len:
                best_pts = pts_ls
                best_len = len(pts_ls)
    
    return best_pts

def _construct_sidon_1d():
    """Construct points on a line using Sidon sequence."""
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, 0) for x in xs]

def _construct_diagonal():
    """Construct points on diagonal."""
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, x) for x in xs]

def construct():
    start = time.time()
    
    # Start with 1D Sidon
    pts = _construct_sidon_1d()
    best_pts = list(pts)
    best_len = len(pts)
    
    # Try diagonal
    diag_pts = _construct_diagonal()
    if len(diag_pts) > best_len:
        best_pts = diag_pts
        best_len = len(diag_pts)
    
    # Try 2D greedy with local search
    if time.time() - start < TIME_LIMIT - 30:
        pts_2d = _construct_2d_greedy_with_search()
        if len(pts_2d) > best_len:
            best_pts = pts_2d
            best_len = len(pts_2d)
    
    # Final local search on best
    if time.time() - start < TIME_LIMIT - 5:
        remaining_time = TIME_LIMIT - 10 - (time.time() - start)
        if remaining_time > 5:
            best_pts = _local_search(best_pts, time_budget=remaining_time)
    
    return best_pts
# EVOLVE-BLOCK-END
