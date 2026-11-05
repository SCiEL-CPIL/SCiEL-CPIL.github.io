import time
import random
from itertools import combinations

DEADLINE = 118.0  # safety margin before 120s kill

def _are_collinear(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

def _build_masks(points):
    n = len(points)
    masks = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    masks[i] |= (1<<j) | (1<<k)
                    masks[j] |= (1<<i) | (1<<k)
                    masks[k] |= (1<<i) | (1<<j)
    return masks

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if (_are_collinear(points[i],points[j],points[k]) and
                        _are_collinear(points[i],points[j],points[l]) and
                        _are_collinear(points[i],points[k],points[l])):
                        return True
    return False

def _max_3free(points):
    n = len(points)
    masks = _build_masks(points)
    # Fast exact branch and bound for max 3-free subset
    # A set S is 3-free iff for all i,j in S, masks[i] & (1<<j) == 0
    # i.e., no two points in S are collinear with a third point in S.
    best = [0]
    
    def feasible(i, current_mask):
        # Check if point i can be added to current set
        # For each j in current, check if i and j share a collinear third in current
        for j in range(n):
            if current_mask & (1<<j):
                if masks[i] & (1<<j):
                    return False
        return True
    
    def bb(idx, current_mask, size):
        if size + (n - idx) <= best[0]:
            return
        if idx == n:
            if size > best[0]:
                best[0] = size
            return
        # Option: skip point idx
        bb(idx+1, current_mask, size)
        # Option: include point idx if feasible
        if feasible(idx, current_mask):
            bb(idx+1, current_mask | (1<<idx), size+1)
    
    bb(0, 0, 0)
    return best[0]

def _valid(points):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            return False
        if p in seen:
            return False
        seen.add(p)
    return not _has_four_collinear(points)

def construct():
    start = time.time()
    rng = random.Random(42)
    
    # Initial: parabola gives score 20 (no three collinear)
    # We want to MINIMIZE the max 3-free subset.
    # Start with a configuration that has many collinear triples.
    # Try a grid-like start with perturbations.
    
    def random_start():
        pts = []
        for _ in range(20):
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            while (x,y) in pts:
                x = rng.randint(-50, 50)
                y = rng.randint(-50, 50)
            pts.append((x,y))
        return pts
    
    # Try several initial configurations
    candidates = []
    
    # 1) Parabola (score 20, baseline)
    candidates.append([(x, x*x) for x in range(-10, 10)])
    
    # 2) Random configs
    for _ in range(50):
        candidates.append(random_start())
    
    # 3) Structured: two lines with many points (but need no four collinear)
    # Place points on lines with at most 3 per line
    def structured_start():
        pts = []
        # Line y=0: 3 points
        pts.extend([(0,0),(1,0),(2,0)])
        # Line y=1: 3 points
        pts.extend([(0,1),(1,1),(2,1)])
        # Line y=2: 3 points
        pts.extend([(0,2),(1,2),(2,2)])
        # Line x=0 already used... use diagonal
        # Line x=3: 3 points
        pts.extend([(3,0),(3,1),(3,2)])
        # Line y=3: 3 points
        pts.extend([(0,3),(1,3),(2,3)])
        # Line x=4: 3 points
        pts.extend([(4,0),(4,1),(4,2)])
        # Need 2 more, place carefully
        pts.extend([(5,3),(6,3)])
        return pts
    candidates.append(structured_start())
    
    best_score = 999
    best_points = None
    
    for cand in candidates:
        if time.time() - start > DEADLINE:
            break
        if not _valid(cand):
            continue
        s = _max_3free(cand)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in cand]
    
    # Local search: 2-opt moves
    if best_points is None:
        best_points = [list(p) for p in candidates[0]]
        best_score = _max_3free(best_points)
    
    current = [tuple(p) for p in best_points]
    cur_score = best_score
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start < DEADLINE and no_improve < max_no_improve:
        # Pick a random point and perturb it
        i = rng.randint(0, 19)
        x0, y0 = current[i]
        # Try small perturbations
        found = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0+dx, y0+dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                new_pt = (nx, ny)
                if new_pt in [current[j] for j in range(20) if j != i]:
                    continue
                new_points = list(current)
                new_points[i] = new_pt
                if not _valid(new_points):
                    continue
                s = _max_3free(new_points)
                if s < cur_score:
                    cur_score = s
                    current = new_points
                    found = True
                    no_improve = 0
                    break
            if found:
                break
        if not found:
            no_improve += 1
    
    # Update best
    if cur_score < best_score:
        best_score = cur_score
        best_points = [list(p) for p in current]
    
    return best_points
