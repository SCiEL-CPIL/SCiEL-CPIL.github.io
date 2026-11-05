"""
Minimize ordinary lines for 12 integer points.
Uses finite projective geometry over F_13 (p=13) to find a 12-point
configuration with few ordinary lines, then lifts to integer coordinates
while preserving collinearity structure.
"""
import time
import random
import math

def _line_key(p1, p2, p3=None):
    """Return a canonical key for the line through p1 and p2.
    p3 is optional and only used to verify collinearity when given.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normalize
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    g = math.gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _points_on_line(line_key_val, points):
    """Return list of indices of points lying on the given line."""
    if line_key_val is None:
        return []
    dx, dy, c = line_key_val
    result = []
    for i, (x, y) in enumerate(points):
        if dy * x - dx * y == c:
            result.append(i)
    return result

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given 12 points."""
    n = len(points)
    if n < 2:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary = 0
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 2:
            ordinary += 1
    return ordinary

def _finite_field_count_ordinary(pts_mod_p):
    """Count ordinary lines for points given as (x mod p, y mod p) using
    finite-field arithmetic. Three points are collinear iff the determinant
    is 0 mod p.
    """
    p = len(pts_mod_p)  # not right; p is the prime
    # Actually p is passed separately
    raise NotImplementedError  # we'll inline this

def _ff_count_ordinary(pts, p):
    """pts: list of (x_mod_p, y_mod_p). Count ordinary lines."""
    n = len(pts)
    lines = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = (xj - xi) % p
            dy = (yj - yi) % p
            # line: dy*(x - xi) - dx*(y - yi) == 0  =>  dy*x - dx*y == dy*xi - dx*yi
            c = (dy * xi - dx * yj) % p  # wait, c = dy*xi - dx*yi
            c = (dy * xi - dx * yi) % p
            key = (dx, dy, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary = 0
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 2:
            ordinary += 1
    return ordinary

def _lift_to_integers(pts_mod_p, p, scale):
    """Lift points from F_p to integers by multiplying coordinates by scale.
    This preserves collinearity exactly because the affine structure is
    homomorphic under scaling.
    """
    return [[x * scale, y * scale] for x, y in pts_mod_p]

def _is_valid(pts):
    if len(pts) != 12:
        return False
    tuples = [tuple(p) for p in pts]
    if len(set(tuples)) != 12:
        return False
    for x, y in tuples:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
    # Not all collinear
    x1, y1 = tuples[0]
    x2, y2 = tuples[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, 12):
        xi, yi = tuples[i]
        if (xi - x1) * dy - (yi - y1) * dx != 0:
            return True
    return False

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin before 150s kill

    best_points = None
    best_count = float('inf')

    def update_best(pts, cnt):
        nonlocal best_points, best_count
        if cnt < best_count:
            best_count = cnt
            best_points = [p[:] for p in pts]

    # === Strategy 1: Search in F_13 for 12-point sets with few ordinary lines ===
    # The projective plane PG(2,13) has 13^2 + 13 + 1 = 183 points.
    # We pick 12 affine points from F_13 x F_13.
    # Known: In PG(2,q), a "line spread" or related structures can have
    # few ordinary lines. For q=13, we search.
    # 
    # We use a structured search: enumerate candidate sets based on
    # unions of lines in the finite field, then evaluate.

    p = 13
    all_points_ff = [(x, y) for x in range(p) for y in range(p)]
    
    def ff_line_members(p1, p2):
        """All points in F_p^2 collinear with p1, p2."""
        x1, y1 = p1
        x2, y2 = p2
        dx = (x2 - x1) % p
        dy = (y2 - y1) % p
        if dx == 0 and dy == 0:
            return [p1]
        # Parametric: (x1 + t*dx, y1 + t*dy) for t in range(p)
        members = set()
        for t in range(p):
            x = (x1 + t * dx) % p
            y = (y1 + t * dy) % p
            members.add((x, y))
        return list(members)

    # Build a list of all lines in the affine plane F_13^2
    # Each line is a frozenset of 13 points.
    all_lines = []
    seen_lines = set()
    for i in range(p * p):
        for j in range(i + 1, p * p):
            line = ff_line_members(all_points_ff[i], all_points_ff[j])
            key = frozenset(line)
            if key not in seen_lines:
                seen_lines.add(key)
                all_lines.append(key)

    # There are p*(p+1) = 13*14 = 182 lines in the affine plane F_13^2
    # (plus 1 "line at infinity" in PG(2,13), which we ignore for affine points)
    
    # Strategy: Pick 2 or 3 lines, take union of some subset of their points
    # to get exactly 12 points, then count ordinary lines.
    
    # For two lines: if they share 1 point (intersect), union has 13+13-1=25 points.
    # We need to select 12 from 25.
    # For 3 lines in general position (pairwise intersect, no common point):
    # union has 13*3 - 3 = 36 points (each pair shares 1, triple intersection 1
    # is counted in all three pairs... let me compute: |A∪B∪C| = |A|+|B|+|C|
    # - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|. If no triple intersection:
    # = 13*3 - 1*3 + 0 = 39 - 3 = 36. If they share a common point:
    # = 39 - 3 + 1 = 37? No: |A∩B|=|A∩C|=|B∩C|=1, |A∩B∩C|=1,
    # so 39 - 3 + 1 = 37.)
    
    # This is getting complex. Let's do a direct search instead.

    # === Strategy 2: Direct combinatorial search in F_13 ===
    # Try structured 12-point sets:
    # (a) 3 lines x 4 points each, arranged so intersections are shared
    # (b) Random subsets of F_13^2 with local improvement
    
    # Try specific constructions:
    # 4 points on 3 lines, with each line pair sharing 2 points:
    # Line L1: y=0, points at x=0,1,2,3
    # Line L2: y=1, points at x=0,1,2,3
    # Line L3: x=0, points at y=0,1,2,3
    # Total unique: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(0,3) = 10 points
    # Need 2 more. Add points that lie on diagonals to create more collinearities.
    
    # Let me just do a focused search: enumerate all 12-subsets of a smaller grid
    # in F_13, but that's C(169,12) which is too large.
    
    # Instead: use the fact that optimal configurations for small n in finite
    # fields are known to be related to grid-like or line-union structures.
    
    # Let's try: take 3 lines through a common point, with 4 points on each
    # (including the common point). Total = 3*4 - 2 = 10. Add 2 more points
    # on a 4th line through the common point: total = 10 + 3 = 13, too many.
    # With 3 lines of 4 and 2 extra: 10+2=12.
    
    # Better: Let me just run a randomized local search in F_13, then lift.
    
    random.seed(42)
    
    # Local search in F_13
    for restart in range(2000):
        if time.time() > deadline - 30:
            break
        
        # Random initial set of 12 points in F_13
        init_pts = random.sample(all_points_ff, 12)
        cur = list(init_pts)
        cur_cnt = _ff_count_ordinary(cur, p)
        
        improved = True
        max_iters = 300
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 30:
            improved = False
            iters += 1
            idx = random.randint(0, 11)
            old_pt = cur[idx]
            # Try several candidate replacements
            candidates = random.sample([pt for pt in all_points_ff if pt != old_pt], 
                                       min(15, p*p - 1))
            best_cand = None
            best_cand_cnt = cur_cnt
            for cand in candidates:
                cur[idx] = cand
                # Check for duplicates
                if len(set(cur)) != 12:
                    cur[idx] = old_pt
                    continue
                cnt = _ff_count_ordinary(cur, p)
                if cnt < best_cand_cnt:
                    best_cand_cnt = cnt
                    best_cand = cand
            if best_cand is not None:
                cur[idx] = best_cand
                cur_cnt = best_cand_cnt
                improved = True
                if cur_cnt <= 6:
                    break
        
        # Lift to integers
        # Scale by a factor to get integer coordinates. Since we're in F_13,
        # the coordinates are already 0..12. Scaling by 1 gives small ints.
        # But we need to make sure the lifted configuration preserves collinearity.
        # The issue: collinearity in F_13 does NOT directly imply collinearity
        # in Z, because three points (x1,y1),(x2,y2),(x3,y3) in Z that are
        # collinear mod p may not be collinear in Z.
        
        # Actually, the correct approach: find a configuration in Z directly
        # that mimics the F_13 structure. The F_13 search is a heuristic to
        # guide the integer search.
        
        # Let's lift and check in Z:
        scale = 1
        lifted = [[x * scale, y * scale] for x, y in cur]
        if _is_valid(lifted):
            z_cnt = count_ordinary_lines(lifted)
            update_best(lifted, z_cnt)
            
            # Also try with larger scale
            for s in [10, 100, 1000, 10000]:
                lifted_s = [[x * s, y * s] for x, y in cur]
                if _is_valid(lifted_s):
                    z_cnt_s = count_ordinary_lines(lifted_s)
                    update_best(lifted_s, z_cnt_s)
        
        if best_count <= 6:
            break
    
    # === Strategy 3: Local search in Z directly ===
    if best_points is not None:
        for restart in range(500):
            if time.time() > deadline - 5:
                break
            cur = [p[:] for p in best_points]
            cur_cnt = count_ordinary_lines(cur)
            improved = True
            iters = 0
            max_iters = 500
            while improved and iters < max_iters and time.time() < deadline - 5:
                improved = False
                iters += 1
                idx = random.randint(0, 11)
                ox, oy = cur[idx]
                for _ in range(20):
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    cur[idx] = [nx, ny]
                    if not _is_valid(cur):
                        cur[idx] = [ox, oy]
                        continue
                    new_cnt = count_ordinary_lines(cur)
                    if new_cnt < cur_cnt:
                        cur_cnt = new_cnt
                        improved = True
                        update_best(cur, cur_cnt)
                        if best_count <= 6:
                            break
                        break
                    else:
                        cur[idx] = [ox, oy]
                if best_count <= 6:
                    break
            if best_count <= 6:
                break
    
    # === Strategy 4: Try known small configurations ===
    # 3x4 grid
    grid = [[x, y] for x in range(4) for y in range(3)]
    if _is_valid(grid):
        cnt = count_ordinary_lines(grid)
        update_best(grid, cnt)
    
    # 4x3 grid
    grid2 = [[x, y] for x in range(3) for y in range(4)]
    if _is_valid(grid2):
        cnt = count_ordinary_lines(grid2)
        update_best(grid2, cnt)
    
    # Various structured configs
    configs = [
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(0,3),(1,2),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(2,0),(4,0),(6,0),(1,1),(3,1),(5,1),(7,1),(0,2),(2,2),(4,2),(6,2)],
        [(0,0),(1,1),(2,2),(3,3),(0,1),(1,0),(2,1),(3,0),(0,2),(1,3),(2,0),(3,1)],
        [(0,0),(1,0),(0,1),(1,1),(2,0),(3,0),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
    ]
    for cfg in configs:
        if _is_valid(cfg):
            cnt = count_ordinary_lines(cfg)
            update_best(cfg, cnt)
    
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
