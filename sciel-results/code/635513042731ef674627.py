"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _get_lines(points):
    """Return dict mapping line_key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    return line_map

def _smart_local_search(points, max_iters=30000, deadline=None):
    """
    Smart local search: for each point, try placing it on an existing line
    (one that already has 2+ other points) to create a 3+ point line,
    which reduces ordinary line count.
    """
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        lines = _get_lines(best)
        
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            xi, yi = best[idx]
            
            # Strategy 1: Try to place on an existing line with 2+ other points
            # For each line not containing idx, if it has >= 2 points, try extending it
            for line_key, pts_on_line in lines.items():
                if idx in pts_on_line:
                    continue
                if len(pts_on_line) < 2:
                    continue
                
                # Get two points on this line to determine direction
                pts_list = list(pts_on_line)
                p1 = best[pts_list[0]]
                p2 = best[pts_list[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                g = _gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                # Find integer points on this line near the current point
                # Line: dy_n * x - dx_n * y = c
                c = dy_n * p1[0] - dx_n * p1[1]
                
                # Try a few positions along this line
                base_x = p1[0]
                base_y = p1[1]
                for t in range(-5, 6):
                    new_x = base_x + t * dx_n
                    new_y = base_y + t * dy_n
                    # Check if this is on the line
                    if dy_n * new_x - dx_n * new_y != c:
                        continue
                    new_pt = [new_x, new_y]
                    # Check bounds and distinctness
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    valid = True
                    for j in range(n):
                        if j != idx and best[j] == new_pt:
                            valid = False
                            break
                    if not valid:
                        continue
                    best[idx] = new_pt
                    cnt = _count_ordinary_lines(best)
                    if cnt < best_cnt:
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        # Rebuild lines after improvement
                        lines = _get_lines(best)
                    else:
                        best[idx] = [xi, yi]
            
            # Strategy 2: Try nearby lattice points
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
                    if abs(new_pt[0]) > 100 or abs(new_pt[1]) > 100:
                        continue
                    valid = True
                    for j in range(n):
                        if j != idx and best[j] == new_pt:
                            valid = False
                            break
                    if not valid:
                        continue
                    best[idx] = new_pt
                    cnt = _count_ordinary_lines(best)
                    best[idx] = [xi, yi]
                    if cnt < best_cnt:
                        best[idx] = new_pt
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        lines = _get_lines(best)
            
            # Strategy 3: Random moves
            for _ in range(5):
                if deadline and time.time() > deadline:
                    break
                rx = random.randint(-50, 50)
                ry = random.randint(-50, 50)
                new_pt = [rx, ry]
                valid = True
                for j in range(n):
                    if j != idx and best[j] == new_pt:
                        valid = False
                        break
                if not valid:
                    continue
                best[idx] = new_pt
                cnt = _count_ordinary_lines(best)
                best[idx] = [xi, yi]
                if cnt < best_cnt:
                    best[idx] = new_pt
                    best_cnt = cnt
                    improved = True
                    iters += 1
                    lines = _get_lines(best)
        
        iters += 1
    return best, best_cnt

def _pair_swap_search(points, max_iters=10000, deadline=None):
    """Try swapping pairs of points with random new positions."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        i, j = random.sample(range(n), 2)
        old_i, old_j = best[i][:], best[j][:]
        for _ in range(5):
            ni = [random.randint(-30, 30), random.randint(-30, 30)]
            nj = [random.randint(-30, 30), random.randint(-30, 30)]
            if ni == nj:
                continue
            valid = True
            for k in range(n):
                if k != i and k != j:
                    if best[k] == ni or best[k] == nj:
                        valid = False
                        break
                if k == i and ni == old_j:
                    valid = False
                if k == j and nj == old_i:
                    valid = False
            if not valid:
                continue
            best[i] = ni
            best[j] = nj
            cnt = _count_ordinary_lines(best)
            best[i] = old_i
            best[j] = old_j
            if cnt < best_cnt:
                best[i] = ni
                best[j] = nj
                best_cnt = cnt
    return best, best_cnt

def _make_grid(w, h, offset=(0, 0)):
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append([x + offset[0], y + offset[1]])
    return pts

def _projective_transform(points_3d):
    """
    Apply a projective transformation to 3D homogeneous coordinates.
    Returns 2D points if possible (all w != 0).
    """
    # Try several projective transformations
    transforms = [
        # Identity-like with small perturbation
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        # Shear in x
        [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
        # Shear in y
        [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
        # Combined shear
        [[1, 0, 0], [0, 1, 0], [1, 1, 1]],
        # Scale + shear
        [[2, 0, 0], [0, 2, 0], [1, 1, 2]],
        # Another variant
        [[1, 0, 0], [0, 1, 0], [2, 1, 1]],
        [[1, 0, 0], [0, 1, 0], [1, 2, 1]],
        # More aggressive
        [[3, 0, 0], [0, 3, 0], [1, 1, 3]],
        [[2, 1, 0], [0, 2, 0], [1, 0, 2]],
        [[1, 0, 0], [1, 2, 0], [0, 1, 2]],
    ]
    
    for M in transforms:
        result = []
        valid = True
        for p in points_3d:
            x, y, w = p
            # Apply transform
            nx = M[0][0]*x + M[0][1]*y + M[0][2]*w
            ny = M[1][0]*x + M[1][1]*y + M[1][2]*w
            nw = M[2][0]*x + M[2][1]*y + M[2][2]*w
            if nw == 0:
                valid = False
                break
            result.append([nx, ny, nw])
        if not valid:
            continue
        
        # Convert to affine
        pts = []
        valid = True
        seen = set()
        for x, y, w in result:
            # Check if we can get integer coordinates
            # We need x/w and y/w to be integers
            if w == 0:
                valid = False
                break
            if x % w != 0 or y % w != 0:
                # Try scaling - multiply by common factor
                # Actually, let's just use rational and check later
                valid = False
                break
            px, py = x // w, y // w
            if (px, py) in seen:
                valid = False
                break
            seen.add((px, py))
            if abs(px) > 10**9 or abs(py) > 10**9:
                valid = False
                break
            pts.append([px, py])
        if valid and len(pts) == len(points_3d):
            return pts
    
    return None

def _make_projective_grid():
    """
    Create a 3x3 grid in homogeneous coordinates, then apply projective transform.
    The 3x3 grid has 9 points: 3 horizontal, 3 vertical, 2 diagonal lines with 3 points each.
    That's 8 lines with 3+ points.
    Total pairs: C(9,2) = 36
    Pairs on 3-point lines: 8*3 = 24 (each 3-point line has C(3,2)=3 pairs)
    But we need to be careful about double counting.
    
    Actually for 9 points in 3x3 grid:
    Lines with 3 points: 3 horizontal + 3 vertical + 2 diagonal = 8 lines
    Each such line accounts for 3 pairs
    Total pairs accounted: 8*3 = 24
    Remaining pairs: 36 - 24 = 12
    These 12 pairs form 12 ordinary lines (each with exactly 2 points)
    
    So 3x3 grid has 12 ordinary lines. Not optimal for 9 points.
    
    For 12 points, we want to minimize ordinary lines.
    Let's try different configurations.
    """
    # 3x3 grid as homogeneous points (w=1)
    pts_3d = []
    for x in range(3):
        for y in range(3):
            pts_3d.append([x, y, 1])
    
    result = _projective_transform(pts_3d)
    if result:
        return result
    
    # Fallback to regular grid
    return _make_grid(3, 3)

def _make_pascal_like():
    """
    Try to create a configuration inspired by Pascal's theorem or other
    projective geometry constructions that maximize collinearities.
    
    For 12 points, aim for many lines with 3+ points.
    """
    # Start with a hexagon and add points on diagonals
    # Or use a known configuration
    
    # Try: 4 points on each of 3 parallel lines, arranged to create
    # additional collinear triples
    
    pts = []
    # Line 1: y=0
    for x in [0, 3, 6, 9]:
        pts.append([x, 0])
    # Line 2: y=3  
    for x in [1, 4, 7, 10]:
        pts.append([x, 3])
    # Line 3: y=6
    for x in [2, 5, 8, 11]:
        pts.append([x, 6])
    
    return pts

def _make_complete_quadrilateral():
    """
    Complete quadrilateral: 4 lines, 6 intersection points.
    Then add 6 more points strategically.
    """
    # 4 lines in general position create 6 intersection points
    # Use lines: x=0, y=0, x+y=1, x-y=1
    # Intersections:
    # (0,0), (0,1), (0,-1), (1,0), (-1,0), (1,1)... 
    # Let me compute properly
    
    # Line 1: x = 0
    # Line 2: y = 0  
    # Line 3: x + y = 2
    # Line 4: x - y = 0
    
    # Intersections:
    # L1∩L2: (0,0)
    # L1∩L3: (0,2)
    # L1∩L4: (0,0) - same as L1∩L2! Not good.
    
    # Try different lines:
    # L1: y = 0
    # L2: y = 1
    # L3: x = 0
    # L4: x = 1
    # Intersections: (0,0), (1,0), (0,1), (1,1) - only 4 points
    
    # For 4 lines in general position (no two parallel, no three concurrent):
    # L1: y = 0
    # L2: y = x
    # L3: y = 2x + 1
    # L4: y = -x + 3
    
    # L1∩L2: (0,0)
    # L1∩L3: (-0.5, 0) - not integer
    # Let me use integer-friendly lines
    
    # L1: y = 0
    # L2: y = x
    # L3: y = 2x - 1
    # L4: y = -x + 2
    
    # L1∩L2: (0,0)
    # L1∩L3: (0.5, 0) - not integer
    
    # Use:
    # L1: y = 0
    # L2: y = 2x
    # L3: y = 3x - 3
    # L4: y = -x + 4
    
    # L1∩L2: (0,0)
    # L1∩L3: (1,0)
    # L1∩L4: (4,0)
    # L2∩L3: 2x = 3x-3 => x=3, y=6 => (3,6)
    # L2∩L4: 2x = -x+4 => 3x=4 => x=4/3 - not integer
    
    # This is getting complicated. Let me just use a known good configuration.
    
    # 3x4 grid
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def _make_harmonic_config():
    """
    Configuration based on harmonic bundles.
    Try to create many 3-point lines.
    """
    # Use a 3x3 grid plus 3 points at infinity (in projective sense)
    # Represented as large coordinates
    
    pts = []
    # 3x3 grid
    for x in range(3):
        for y in range(3):
            pts.append([x, y])
    
    # Add 3 more points to create additional collinearities
    # Try to place them on lines that currently have 2 points
    
    # Current 3x3 grid lines with 2 points:
    # Many diagonal-like lines
    
    # Add points to extend some of these
    pts.append([4, 2])   # extends line through (0,0),(2,1) direction
    pts.append([2, 4])   # extends line through (1,2),(2,3)? No, (1,2) to (2,4) is (1,2)
    pts.append([4, 4])   # corner
    
    return pts

def _fast_count_ordinary(points):
    """Faster count using line grouping with normalized representation."""
    n = len(points)
    if n < 2:
        return 0
    
    # For each pair, compute line key
    line_map = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            key = (dx_n, dy_n, c)
            if key not in line_map:
                line_map[key] = 0
            line_map[key] += 1
    
    # A line with k points has C(k,2) pairs
    # We need to find which lines have exactly 2 points
    # But we only have pair counts, not point counts
    # We need a different approach
    
    # Actually, let's use the same approach but optimized
    line_pts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            key = (dx_n, dy_n, c)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    ordinary = 0
    for key, pts in line_pts.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _try_move_to_line(points, idx, lines, deadline=None):
    """Try to move point idx onto an existing line with 2+ other points."""
    xi, yi = points[idx]
    n = len(points)
    
    for line_key, pts_on_line in lines.items():
        if idx in pts_on_line:
            continue
        if len(pts_on_line) < 2:
            continue
        
        # Get direction from two points on this line
        pts_list = list(pts_on_line)
        p1_idx, p2_idx = pts_list[0], pts_list[1]
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            continue
        g = _gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        
        c = dy_n * p1[0] - dx_n * p1[1]
        
        # Try positions along this line near current point
        base_x, base_y = p1[0], p1[1]
        for t in range(-10, 11):
            new_x = base_x + t * dx_n
            new_y = base_y + t * dy_n
            if dy_n * new_x - dx_n * new_y != c:
                continue
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            # Check distinctness
            valid = True
            for j in range(n):
                if j != idx and points[j] == [new_x, new_y]:
                    valid = False
                    break
            if not valid:
                continue
            
            old = points[idx]
            points[idx] = [new_x, new_y]
            yield new_x, new_y, old
            points[idx] = old

def _smart_local_search_v2(points, max_iters=20000, deadline=None):
    """
    Improved local search with better line placement strategy.
    """
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _fast_count_ordinary(best)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        lines = _get_lines(best)
        
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            xi, yi = best[idx]
            
            # Strategy 1: Place on existing 2+ point lines
            for line_key, pts_on_line in lines.items():
                if idx in pts_on_line:
                    continue
                if len(pts_on_line) < 2:
                    continue
                
                pts_list = list(pts_on_line)
                p1 = best[pts_list[0]]
                p2 = best[pts_list[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                g = _gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                c = dy_n * p1[0] - dx_n * p1[1]
                
                base_x, base_y = p1[0], p1[1]
                for t in range(-8, 9):
                    new_x = base_x + t * dx_n
                    new_y = base_y + t * dy_n
                    if dy_n * new_x - dx_n * new_y != c:
                        continue
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    valid = True
                    for j in range(n):
                        if j != idx and best[j] == [new_x, new_y]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    best[idx] = [new_x, new_y]
                    cnt = _fast_count_ordinary(best)
                    if cnt < best_cnt:
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        lines = _get_lines(best)
                    else:
                        best[idx] = [xi, yi]
            
            # Strategy 2: Small local moves
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
                    if abs(new_pt[0]) > 200 or abs(new_pt[1]) > 200:
                        continue
                    valid = True
                    for j in range(n):
                        if j != idx and best[j] == new_pt:
                            valid = False
                            break
                    if not valid:
                        continue
                    best[idx] = new_pt
                    cnt = _fast_count_ordinary(best)
                    best[idx] = [xi, yi]
                    if cnt < best_cnt:
                        best[idx] = new_pt
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        lines = _get_lines(best)
            
            # Strategy 3: Random jumps to encourage exploration
            for _ in range(3):
                if deadline and time.time() > deadline:
                    break
                rx = random.randint(-100, 100)
                ry = random.randint(-100, 100)
                new_pt = [rx, ry]
                valid = True
                for j in range(n):
                    if j != idx and best[j] == new_pt:
                        valid = False
                        break
                if not valid:
                    continue
                best[idx] = new_pt
                cnt = _fast_count_ordinary(best)
                best[idx] = [xi, yi]
                if cnt < best_cnt:
                    best[idx] = new_pt
                    best_cnt = cnt
                    improved = True
                    iters += 1
                    lines = _get_lines(best)
        
        iters += 1
    return best, best_cnt

def _multi_point_swap(points, max_iters=5000, deadline=None):
    """Try moving 2-3 points simultaneously for better escape from local minima."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _fast_count_ordinary(best)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Choose 2 or 3 points to move
        k = random.choice([2, 3])
        indices = random.sample(range(n), k)
        old_pts = [best[i][:] for i in indices]
        
        for _ in range(8):
            new_pts = []
            valid = True
            for idx in indices:
                np = [random.randint(-50, 50), random.randint(-50, 50)]
                # Check against all other points and other new points
                for j in range(n):
                    if j not in indices:
                        if best[j] == np:
                            valid = False
                            break
                for other_np in new_pts:
                    if other_np == np:
                        valid = False
                        break
                if not valid:
                    break
                new_pts.append(np)
            
            if not valid:
                continue
            
            for idx, np in zip(indices, new_pts):
                best[idx] = np
            
            cnt = _fast_count_ordinary(best)
            
            for idx, op in zip(indices, old_pts):
                best[idx] = op
            
            if cnt < best_cnt:
                for idx, np in zip(indices, new_pts):
                    best[idx] = np
                best_cnt = cnt
    
    return best, best_cnt

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    random.seed(42)
    
    # Generate diverse initial configurations
    seeds = []
    
    # 3x4 grid
    seeds.append(_make_grid(4, 3))
    seeds.append(_make_grid(3, 4))
    
    # Projective grid
    proj_grid = _make_projective_grid()
    if proj_grid and len(proj_grid) >= 9:
        # Extend to 12 points
        while len(proj_grid) < 12:
            proj_grid.append([random.randint(-10, 10), random.randint(-10, 10)])
        seeds.append(proj_grid[:12])
    
    # Pascal-like
    seeds.append(_make_pascal_like())
    
    # Complete quadrilateral based
    seeds.append(_make_complete_quadrilateral())
    
    # Harmonic config
    seeds.append(_make_harmonic_config())
    
    # 4x3 with perturbations
    for _ in range(3):
        pts = []
        for x in range(4):
            for y in range(3):
                pts.append([x + random.randint(-1, 1), y + random.randint(-1, 1)])
        # Ensure distinct
        unique = []
        seen = set()
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique.append(p)
        while len(unique) < 12:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique.append(p)
        seeds.append(unique[:12])
    
    # Random configurations
    for _ in range(10):
        pts = []
        seen = set()
        while len(pts) < 12:
            p = [random.randint(-30, 30), random.randint(-30, 30)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                pts.append(p)
        seeds.append(pts)
    
    best_pts = None
    best_cnt = float('inf')
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Ensure exactly 12 distinct points
        unique = []
        seen = set()
        for p in seed:
            pt = [int(p[0]), int(p[1])]
            if tuple(pt) not in seen and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                seen.add(tuple(pt))
                unique.append(pt)
        while len(unique) < 12:
            p = [random.randint(-50, 50), random.randint(-50, 50)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique.append(p)
        seed = unique[:12]
        
        cnt = _fast_count_ordinary(seed)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in seed]
        
        # Smart local search v2
        pts, cnt = _smart_local_search_v2(seed, max_iters=10000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        # Multi-point swap
        if best_pts and time.time() < deadline - 10:
            pts, cnt = _multi_point_swap(best_pts, max_iters=2000, deadline=deadline)
            if cnt < best_cnt:
                best_cnt = cnt
                best_pts = pts[:]
    
    # Final refinement with more iterations
    if best_pts and time.time() < deadline - 5:
        pts, cnt = _smart_local_search_v2(best_pts, max_iters=30000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
    
    # Additional multi-point swap on best
    if best_pts and time.time() < deadline - 2:
        pts, cnt = _multi_point_swap(best_pts, max_iters=3000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
    
    if best_pts is None:
        best_pts = _make_grid(4, 3)
    
    # Verify distinctness and bounds
    seen = set()
    final = []
    for p in best_pts:
        pt = [int(p[0]), int(p[1])]
        if tuple(pt) not in seen and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
            seen.add(tuple(pt))
            final.append(pt)
    
    while len(final) < 12:
        p = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    return final
# EVOLVE-BLOCK-END
