"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from fractions import Fraction

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

def _try_place_on_line(idx, points, line_key, line_map, max_search=20):
    """
    Try to place point idx on the given line (identified by line_key) such that
    it lands on an integer point not already occupied. Returns (new_point, new_count) 
    if improvement found, else (None, None).
    """
    n = len(points)
    pts_on_line = line_map[line_key]
    if idx in pts_on_line or len(pts_on_line) < 2:
        return None, None
    
    # Get direction from the line key
    dx_n, dy_n, c = line_key
    
    # Find two points on the line to get a base
    pts_list = list(pts_on_line)
    p1 = points[pts_list[0]]
    p2 = points[pts_list[1]]
    
    # The line is: dy_n * x - dx_n * y = c
    # We want integer solutions (x, y) to this equation that are not in points
    
    # General solution: if (x0, y0) is a particular solution, then
    # x = x0 + t * dx_n, y = y0 + t * dy_n for integer t
    
    # Find a particular solution
    # We can use p1 as a particular solution since it's on the line
    x0, y0 = p1[0], p1[1]
    
    # Try values of t around the existing points
    # Existing t values for points on the line
    t_values = set()
    for pidx in pts_on_line:
        px, py = points[pidx]
        # px = x0 + t * dx_n => t = (px - x0) / dx_n if dx_n != 0
        if dx_n != 0:
            if (px - x0) % dx_n == 0:
                t_values.add((px - x0) // dx_n)
        elif dy_n != 0:
            if (py - y0) % dy_n == 0:
                t_values.add((py - y0) // dy_n)
    
    # Also check all other points to avoid collisions
    existing_points = set(tuple(p) for p in points)
    
    # Try t values near the existing t values
    candidate_ts = set()
    for t in t_values:
        for dt in range(-max_search, max_search + 1):
            candidate_ts.add(t + dt)
    
    # Also try t values near 0 and near the current point
    xi, yi = points[idx]
    if dx_n != 0:
        if (xi - x0) % dx_n == 0:
            t_cur = (xi - x0) // dx_n
            for dt in range(-max_search, max_search + 1):
                candidate_ts.add(t_cur + dt)
    elif dy_n != 0:
        if (yi - y0) % dy_n == 0:
            t_cur = (yi - y0) // dy_n
            for dt in range(-max_search, max_search + 1):
                candidate_ts.add(t_cur + dt)
    
    # Sort candidates by distance to current point
    def dist_to_current(t):
        nx = x0 + t * dx_n
        ny = y0 + t * dy_n
        return (nx - xi) ** 2 + (ny - yi) ** 2
    
    candidate_ts = sorted(candidate_ts, key=dist_to_current)
    
    for t in candidate_ts[:50]:  # Try closest 50
        nx = x0 + t * dx_n
        ny = y0 + t * dy_n
        
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            continue
        
        new_pt = (nx, ny)
        if new_pt in existing_points:
            continue
        
        # Check if this is on the line
        if dy_n * nx - dx_n * ny != c:
            continue
        
        # Try this move
        old_pt = points[idx][:]
        points[idx] = [nx, ny]
        cnt = _count_ordinary_lines(points)
        points[idx] = old_pt
        
        return [nx, ny], cnt
    
    return None, None

def _smart_local_search(points, max_iters=50000, deadline=None):
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
            for line_key, pts_on_line in lines.items():
                if idx in pts_on_line:
                    continue
                if len(pts_on_line) < 2:
                    continue
                
                new_pt, cnt = _try_place_on_line(idx, best, line_key, lines, max_search=10)
                if new_pt is not None and cnt < best_cnt:
                    best[idx] = new_pt
                    best_cnt = cnt
                    improved = True
                    iters += 1
                    lines = _get_lines(best)
                    break
            
            # Strategy 2: Try nearby lattice points
            for dx in range(-5, 6):
                for dy in range(-5, 6):
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
            for _ in range(10):
                if deadline and time.time() > deadline:
                    break
                rx = random.randint(-30, 30)
                ry = random.randint(-30, 30)
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

def _pair_swap_search(points, max_iters=15000, deadline=None):
    """Try swapping pairs of points with random new positions."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        i, j = random.sample(range(n), 2)
        old_i, old_j = best[i][:], best[j][:]
        for _ in range(8):
            ni = [random.randint(-40, 40), random.randint(-40, 40)]
            nj = [random.randint(-40, 40), random.randint(-40, 40)]
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

def _make_three_parallel_lines():
    """
    4 points on each of 3 parallel lines.
    This gives 3 lines with 4 points each.
    Plus cross-lines between points on different lines.
    """
    pts = []
    for y in [0, 1, 2]:
        for x in [0, 1, 2, 3]:
            pts.append([x, y])
    return pts

def _make_tetrahedral():
    """
    Points arranged to maximize collinear triples.
    Use a structure based on projective plane ideas.
    """
    pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
    pts.append([2, 0])    
    pts.append([0, 2])    
    pts.append([2, 1])    
    pts.append([1, 2])    
    pts.append([2, 2])    
    pts.append([2, -1])   
    return pts

def _make_optimal_structure():
    """
    Try to construct a configuration close to optimal.
    """
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def _make_hexagonal():
    pts_hex = []
    for r in range(4):
        for c in range(3):
            x = c + (r % 2)
            y = r
            pts_hex.append([x, y])
    return pts_hex

def _make_near_pencil():
    pts_pencil = [[i, 0] for i in range(11)]
    pts_pencil.append([5, 1])
    return pts_pencil

def _make_staggered():
    pts = []
    for i in range(4):
        for j in range(3):
            x = i + (j % 2) * 0.5  # This won't work for integers, let's use different approach
            # Use integer staggering
            x = i * 2 + (j % 2)
            y = j * 2
            pts.append([x, y])
    return pts

def _make_diagonal_grid():
    pts = []
    for i in range(3):
        for j in range(4):
            pts.append([i + j, i])
    return pts

def _scale_points(pts, scale=10):
    return [[x * scale, y * scale] for x, y in pts]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    random.seed(42)
    
    # Generate diverse initial configurations
    seeds = []
    seeds.append(_make_grid(4, 3))
    seeds.append(_make_grid(3, 4))
    seeds.append(_make_three_parallel_lines())
    seeds.append(_make_tetrahedral())
    seeds.append(_make_optimal_structure())
    seeds.append(_make_hexagonal())
    seeds.append(_make_near_pencil())
    seeds.append(_make_staggered())
    seeds.append(_make_diagonal_grid())
    
    # Scaled versions
    seeds.append(_scale_points(_make_grid(4, 3), 5))
    seeds.append(_scale_points(_make_three_parallel_lines(), 5))
    
    # Random configurations
    for _ in range(12):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-25, 25), random.randint(-25, 25)]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)
    
    # More structured random
    for _ in range(8):
        pts = []
        base_x = random.randint(-10, 10)
        base_y = random.randint(-10, 10)
        for i in range(12):
            x = base_x + random.randint(-8, 8)
            y = base_y + random.randint(-8, 8)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        while len(pts) < 12:
            p = [random.randint(-30, 30), random.randint(-30, 30)]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)
    
    best_pts = None
    best_cnt = float('inf')
    
    for seed in seeds:
        if time.time() > deadline:
            break
        cnt = _count_ordinary_lines(seed)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in seed]
        
        # Smart local search
        pts, cnt = _smart_local_search(seed, max_iters=20000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        # Pair swap
        if best_pts and time.time() < deadline:
            pts, cnt = _pair_swap_search(best_pts, max_iters=5000, deadline=deadline)
            if cnt < best_cnt:
                best_cnt = cnt
                best_pts = pts[:]
    
    # Final refinement
    if best_pts and time.time() < deadline - 5:
        pts, cnt = _smart_local_search(best_pts, max_iters=100000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        pts, cnt = _pair_swap_search(best_pts, max_iters=20000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
    
    if best_pts is None:
        best_pts = _make_grid(4, 3)
    
    # Verify distinctness
    seen = set()
    final = []
    for p in best_pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    while len(final) < 12:
        p = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    return final
# EVOLVE-BLOCK-END
