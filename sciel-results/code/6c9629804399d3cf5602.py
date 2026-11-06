import time
import random
from math import gcd
from itertools import combinations

DEADLINE = 145.0  # Safety margin before 150s kill

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _build_lines(points):
    """
    Returns:
    lines: list of frozensets of indices for each collinear set of size >= 3
    line_masks: list of integer masks for each line
    point_lines: list of lists of line indices for each point
    """
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    lines = []
    line_masks = []
    point_lines = [[] for _ in range(n)]
    
    for s in line_map.values():
        if len(s) >= 3:
            lines.append(frozenset(s))
            mask = 0
            for idx in s:
                mask |= 1 << idx
                point_lines[idx].append(len(lines) - 1)
            line_masks.append(mask)
            
    return lines, line_masks, point_lines

def _check_no_four(points, lines):
    for L in lines:
        if len(L) >= 4:
            return False
    return True

def _max_cap_free(points, lines, line_masks, point_lines, time_limit=5.0):
    """
    Exact max subset with no 3 collinear using Branch and Bound.
    """
    n = len(points)
    start = time.time()
    timed_out = [False]
    
    # Precompute for speed
    # For each point, what are the lines it is on?
    # We need to know if adding a point to S creates a monochromatic triple in S.
    # Actually, the condition is: S has no 3 collinear.
    # This is equivalent to: For every line L, |S intersect L| <= 2.
    
    # State:
    # S: set of points included in the subset (bitmask)
    # We will build S incrementally.
    
    # Optimization: 
    # If we pick point i, we cannot pick 2 other points from any line containing i.
    # This is a packing problem.
    
    # Let's use a simpler B&B:
    # Sort points by degree (number of lines they are on) to prune faster?
    # Or just iterate 0..n-1.
    
    # Upper bound:
    # Current size + remaining points.
    # Better bound: Current size + number of points that can still be added.
    # A point j can be added if for all lines L containing j, |S intersect L| < 2.
    
    best = [0]
    cur = [0]
    
    # To speed up, we can use a bitmask for S.
    # But checking lines is O(deg(j)).
    
    # Let's try a recursive search with symmetry breaking.
    # Fix point 0 in S? No, S can be empty.
    # But we want MAX S.
    
    # We can use the fact that if we don't pick point i, we just move to i+1.
    # If we pick point i, we must ensure no line with i gets a 3rd point.
    
    # Let's maintain counts: line_count[li] = number of points in S on line li.
    # If line_count[li] == 2, then no other point on line li can be added.
    
    line_count = [0] * len(lines)
    
    def dfs(idx):
        if timed_out[0]:
            return
        if time.time() - start > time_limit:
            timed_out[0] = True
            return
            
        if idx == n:
            if cur[0] > best[0]:
                best[0] = cur[0]
            return
            
        # Bound: even if we add all remaining points, can we beat best?
        if cur[0] + (n - idx) <= best[0]:
            return
            
        p = idx
        
        # Option 1: Do not include p
        dfs(idx + 1)
        
        if timed_out[0]:
            return
            
        # Option 2: Include p
        # Check if p can be included
        can_include = True
        touched = []
        for li in point_lines[p]:
            if line_count[li] >= 2:
                can_include = False
                break
            line_count[li] += 1
            touched.append(li)
            
        if can_include:
            cur[0] += 1
            dfs(idx + 1)
            cur[0] -= 1
            for li in touched:
                line_count[li] -= 1
        else:
            # Rollback any partial changes if we broke early?
            # We only increment if we haven't broken.
            # If we broke, we didn't increment for the rest.
            # Wait, the loop breaks on first failure.
            # So we need to rollback only what we incremented.
            # My code above increments inside the loop.
            # If it breaks, some were incremented, some were not.
            # So I must rollback the ones I incremented.
            for li in touched:
                line_count[li] -= 1

    # Symmetry breaking: 
    # If we find a solution of size K, we know the complement has size 20-K.
    # The problem is symmetric? No.
    
    # To improve speed, we can order points by "constraint tightness".
    # Points on more lines are more constrained.
    # Let's reorder indices.
    order = sorted(range(n), key=lambda i: len(point_lines[i]), reverse=True)
    
    # Remap point_lines to new indices?
    # It's easier to just use the original indices in dfs but iterate in 'order'.
    # Let's modify dfs to take an 'idx' into the 'order' list.
    
    line_count = [0] * len(lines)
    
    def dfs_order(idx):
        if timed_out[0]:
            return
        if time.time() - start > time_limit:
            timed_out[0] = True
            return
            
        if idx == n:
            if cur[0] > best[0]:
                best[0] = cur[0]
            return
            
        if cur[0] + (n - idx) <= best[0]:
            return
            
        p = order[idx]
        
        # Option 1: Skip p
        dfs_order(idx + 1)
        
        if timed_out[0]:
            return
            
        # Option 2: Include p
        can_include = True
        touched = []
        for li in point_lines[p]:
            if line_count[li] >= 2:
                can_include = False
                break
            line_count[li] += 1
            touched.append(li)
            
        if can_include:
            cur[0] += 1
            dfs_order(idx + 1)
            cur[0] -= 1
            for li in touched:
                line_count[li] -= 1
        else:
            for li in touched:
                line_count[li] -= 1

    dfs_order(0)
    return best[0]

def _score(points, time_limit=5.0):
    lines, lm, pl = _build_lines(points)
    if not _check_no_four(points, lines):
        return 999
    return _max_cap_free(points, lines, lm, pl, time_limit)

def _valid(points):
    if len(set(tuple(p) for p in points)) != 20:
        return False
    if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
        return False
    lines, _, _ = _build_lines(points)
    return _check_no_four(points, lines)

def _generate_grid_perturbed(seed):
    """
    Generate a configuration starting from a 5x4 grid and perturbing points.
    """
    rng = random.Random(seed)
    pts = []
    # 5x4 grid
    # x in -2..2, y in -1.5..1.5 -> scale to integers
    # Let's use x in -4..4 (9 values), y in -3..3 (7 values)
    # We need 20 points.
    # Take a 5x4 subgrid.
    x_coords = [-4, -2, 0, 2, 4]
    y_coords = [-3, -1, 1, 3]
    
    base_pts = []
    for x in x_coords:
        for y in y_coords:
            base_pts.append([x, y])
            
    # Perturb
    for i in range(20):
        px, py = base_pts[i]
        # Small jitter
        jx = rng.randint(-1, 1)
        jy = rng.randint(-1, 1)
        nx = px + jx
        ny = py + jy
        pts.append([nx, ny])
        
    # Ensure distinct and valid
    seen = set()
    for i in range(20):
        while tuple(pts[i]) in seen or not (-100 <= pts[i][0] <= 100 and -100 <= pts[i][1] <= 100):
            pts[i][0] += 1
            if pts[i][0] > 100:
                pts[i][0] = -100
            if pts[i][1] > 100:
                pts[i][1] = -100
        seen.add(tuple(pts[i]))
        
    # Check no 4 collinear
    # If invalid, apply larger random perturbations
    lines, _, _ = _build_lines(pts)
    if not _check_no_four(pts, lines):
        # Force fix
        for i in range(20):
            for attempt in range(50):
                lines, _, _ = _build_lines(pts)
                if _check_no_four(pts, lines):
                    break
                # Move point i randomly
                pts[i][0] = rng.randint(-100, 100)
                pts[i][1] = rng.randint(-100, 100)
                
    return pts

def _local_search(points, time_budget, rng):
    """
    Local search with 1-move and 2-swap moves.
    """
    start = time.time()
    cur = [p[:] for p in points]
    lines, lm, pl = _build_lines(cur)
    if not _check_no_four(cur, lines):
        return cur, 999
    cur_score = _max_cap_free(cur, lines, lm, pl, 3.0)
    best = [p[:] for p in cur]
    best_score = cur_score
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start < time_budget and no_improve < max_no_improve:
        improved = False
        # Try random moves
        for _ in range(50):
            if time.time() - start > time_budget:
                break
            
            move_type = rng.choice(['move', 'swap'])
            
            if move_type == 'move':
                idx = rng.randrange(20)
                new_pt = [rng.randint(-100, 100), rng.randint(-100, 100)]
                # Check collision
                collides = False
                for j in range(20):
                    if j != idx and cur[j] == new_pt:
                        collides = True
                        break
                if collides:
                    continue
                
                old_pt = cur[idx][:]
                cur[idx] = new_pt
                new_lines, new_lm, new_pl = _build_lines(cur)
                if _check_no_four(cur, new_lines):
                    new_score = _max_cap_free(cur, new_lines, new_lm, new_pl, 2.0)
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in cur]
                        improved = True
                        no_improve = 0
                    elif new_score == best_score and rng.random() < 0.05:
                        # Accept neutral
                        pass
                    else:
                        cur[idx] = old_pt
                else:
                    cur[idx] = old_pt
                    
            else: # swap
                i, j = rng.sample(range(20), 2)
                # Swap coordinates? Or swap positions with random points?
                # Let's swap point i with a random new point, and point j with a random new point.
                # Actually, just moving one point is simpler and effective.
                # Let's do a "2-opt" like swap of two points with two other random points.
                p1 = cur[i][:]
                p2 = cur[j][:]
                new_p1 = [rng.randint(-100, 100), rng.randint(-100, 100)]
                new_p2 = [rng.randint(-100, 100), rng.randint(-100, 100)]
                
                # Check collisions
                valid = True
                for k in range(20):
                    if k != i and k != j:
                        if cur[k] == new_p1 or cur[k] == new_p2:
                            valid = False
                            break
                if new_p1 == new_p2:
                    valid = False
                if not valid:
                    continue
                    
                cur[i] = new_p1
                cur[j] = new_p2
                new_lines, new_lm, new_pl = _build_lines(cur)
                if _check_no_four(cur, new_lines):
                    new_score = _max_cap_free(cur, new_lines, new_lm, new_pl, 2.0)
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in cur]
                        improved = True
                        no_improve = 0
                    else:
                        cur[i] = p1
                        cur[j] = p2
                else:
                    cur[i] = p1
                    cur[j] = p2
                    
        if not improved:
            no_improve += 1
            # If stuck, try a random restart of a few points
            if no_improve % 10 == 0:
                for _ in range(5):
                    idx = rng.randrange(20)
                    cur[idx] = [rng.randint(-100, 100), rng.randint(-100, 100)]
                # Re-evaluate
                lines, lm, pl = _build_lines(cur)
                if _check_no_four(cur, lines):
                    s = _max_cap_free(cur, lines, lm, pl, 2.0)
                    if s < best_score:
                        best_score = s
                        best = [p[:] for p in cur]
                        no_improve = 0

    return best, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 999
    
    # Multi-start
    num_starts = 20
    time_per_start = (DEADLINE - 10) / num_starts
    
    for s in range(num_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        pts = _generate_grid_perturbed(s)
        if not _valid(pts):
            continue
            
        rng = random.Random(1000 + s)
        refined, rs = _local_search(pts, time_per_start, rng)
        
        if rs < best_score:
            best_score = rs
            best_points = refined
            
        if best_score <= 12:
            break
            
    if best_points is None:
        # Fallback
        best_points = [[x, x*x//4] for x in range(-10, 10)]
        
    return best_points
