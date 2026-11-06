"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines_fast(points):
    """Count lines containing exactly 2 of the given points using efficient method."""
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
            if g == 0:
                continue
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
            if g == 0:
                continue
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

def _count_triples(points):
    """Count number of collinear triples (i,j,k) with i<j<k."""
    n = len(points)
    line_map = _get_lines(points)
    triples = 0
    for key, pts in line_map.items():
        k = len(pts)
        if k >= 3:
            triples += k * (k - 1) * (k - 2) // 6
    return triples

def _evaluate_combined(points):
    """Combined score: primary is ordinary lines, secondary is -triples."""
    ordinary = _count_ordinary_lines(points)
    triples = _count_triples(points)
    return (ordinary, -triples)

def _try_place_on_line(idx, points, line_key, line_map, max_search=30):
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
    
    # Use p1 as a particular solution
    x0, y0 = p1[0], p1[1]
    
    # Find existing t values for points on the line
    t_values = set()
    for pidx in pts_on_line:
        px, py = points[pidx]
        if dx_n != 0:
            if (px - x0) % dx_n == 0:
                t_values.add((px - x0) // dx_n)
        elif dy_n != 0:
            if (py - y0) % dy_n == 0:
                t_values.add((py - y0) // dy_n)
    
    existing_points = set(tuple(p) for p in points)
    
    # Generate candidate t values
    candidate_ts = set()
    for t in t_values:
        for dt in range(-max_search, max_search + 1):
            candidate_ts.add(t + dt)
    
    # Also try t values near 0
    for dt in range(-max_search, max_search + 1):
        candidate_ts.add(dt)
    
    # Sort candidates by distance to current point
    xi, yi = points[idx]
    def dist_to_current(t):
        nx = x0 + t * dx_n
        ny = y0 + t * dy_n
        return (nx - xi) ** 2 + (ny - yi) ** 2
    
    candidate_ts = sorted(candidate_ts, key=dist_to_current)
    
    for t in candidate_ts[:100]:  # Try closest 100
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
                
                new_pt, cnt = _try_place_on_line(idx, best, line_key, lines, max_search=15)
                if new_pt is not None and cnt < best_cnt:
                    best[idx] = new_pt
                    best_cnt = cnt
                    improved = True
                    iters += 1
                    lines = _get_lines(best)
                    break
            
            # Strategy 2: Try nearby lattice points with larger radius
            for dx in range(-8, 9):
                for dy in range(-8, 9):
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
                    cnt = _count_ordinary_lines(best)
                    best[idx] = [xi, yi]
                    if cnt < best_cnt:
                        best[idx] = new_pt
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        lines = _get_lines(best)
            
            # Strategy 3: Random moves with wider range
            for _ in range(15):
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
        for _ in range(10):
            ni = [random.randint(-60, 60), random.randint(-60, 60)]
            nj = [random.randint(-60, 60), random.randint(-60, 60)]
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

def _projective_transform(points_3x3, a, b, c, d, e, f, g, h, i):
    """Apply projective transformation to 3x3 grid points."""
    result = []
    for (x, y) in points_3x3:
        # Homogeneous: [x, y, 1]
        X = a*x + b*y + c
        Y = d*x + e*y + f
        Z = g*x + h*y + i
        if Z == 0:
            return None  # Point at infinity
        px = X / Z
        py = Y / Z
        # Check if integer (or very close)
        if abs(px - round(px)) > 1e-9 or abs(py - round(py)) > 1e-9:
            return None
        result.append([int(round(px)), int(round(py))])
    return result

def _make_projective_3x3():
    """
    Generate 3x3 grid configurations via projective transforms.
    A 3x3 grid has 8 ordinary lines (the 9 diagonal lines minus 1).
    We search for transforms that produce integer coordinates.
    """
    base_3x3 = [(i, j) for i in range(3) for j in range(3)]
    results = []
    seen = set()
    
    # Try various projective transformations
    # The identity gives the standard 3x3 grid
    # We want transforms where all 9 points map to integers
    
    # General approach: pick 8 points, determine the 9th via projective geometry
    # For a 3x3 grid, the 4 corner points determine the transform up to some freedom
    
    # Try small integer projective matrices
    for a in range(-5, 6):
        for b in range(-5, 6):
            for c in range(-5, 6):
                for d in range(-5, 6):
                    for e in range(-5, 6):
                        for f in range(-5, 6):
                            # Keep it simple: affine transforms (g=0, h=0, i=1)
                            # X = a*x + b*y + c, Y = d*x + e*y + f
                            pts = []
                            valid = True
                            for (x, y) in base_3x3:
                                px = a*x + b*y + c
                                py = d*x + e*y + f
                                pts.append([px, py])
                            
                            # Check distinctness
                            if len(set(tuple(p) for p in pts)) != 9:
                                continue
                            
                            # Check bounds
                            if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in pts):
                                continue
                            
                            key = tuple(tuple(p) for p in sorted(pts))
                            if key in seen:
                                continue
                            seen.add(key)
                            results.append(pts)
                            if len(results) >= 20:
                                break
                        if len(results) >= 20:
                            break
                    if len(results) >= 20:
                        break
                if len(results) >= 20:
                    break
            if len(results) >= 20:
                break
        if len(results) >= 20:
            break
    
    return results

def _make_three_parallel_lines():
    """
    4 points on each of 3 parallel lines.
    """
    pts = []
    for y in [0, 1, 2]:
        for x in [0, 1, 2, 3]:
            pts.append([x, y])
    return pts

def _make_tetrahedral():
    """
    Points arranged to maximize collinear triples.
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

def _make_hesse_like():
    """
    Inspired by Hesse configuration: 9 points with 12 lines of 3 points.
    For 12 points, we extend this idea.
    """
    pts = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [3, 0], [3, 1], [3, 2]
    ]
    return pts

def _make_boroczky_like():
    """
    Boroczky-type construction: points on a conic + points creating many collinearities.
    For 12 points, try to get close to 6 ordinary lines.
    
    Strategy: 6 points on a parabola y=x^2, and 6 points placed to create
    many 3-point lines.
    """
    # 6 points on parabola y = x^2
    para_pts = [[x, x*x] for x in range(6)]
    
    # Place 6 more points to maximize collinearities
    # The lines through pairs of parabola points have various slopes
    # We want to find 6 additional points that lie on many of these lines
    
    # For parabola y=x^2, the line through (a,a^2) and (b,b^2) has equation:
    # y = (a+b)x - ab
    # So for each pair (a,b), the line is y = (a+b)x - ab
    
    # Let's try placing points at strategic locations
    # Point (s, t) lies on line through (a,a^2) and (b,b^2) iff t = (a+b)s - ab
    
    # Try s values and find t that maximizes collinearities
    best_extra = []
    best_score = -1
    
    # Precompute all lines through pairs of parabola points
    lines = []
    for a in range(6):
        for b in range(a+1, 6):
            slope = a + b
            intercept = -a*b
            lines.append((slope, intercept, a, b))
    
    # For each candidate x position, count how many lines pass through (x, t) for integer t
    for sx in range(-5, 15):
        # For each line, compute what y would be
        y_values = []
        for (slope, intercept, a, b) in lines:
            ty = slope * sx + intercept
            y_values.append(ty)
        
        # Find the y value that appears most often (most collinearities)
        from collections import Counter
        y_counts = Counter(y_values)
        best_y, best_count = y_counts.most_common(1)[0]
        
        # Score: number of lines this point lies on
        if best_count > best_score:
            best_score = best_count
            best_extra = [[sx, best_y]]
    
    # This is getting complex. Let's just try a few hand-crafted configurations
    # Configuration 1: 4 points on each of 3 parallel lines, staggered
    config1 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [1, 1], [2, 1], [3, 1], [4, 1],
        [2, 2], [3, 2], [4, 2], [5, 2]
    ]
    
    # Configuration 2: Points forming a complete quadrilateral + extras
    # 4 lines in general position create 6 intersection points
    # Plus 6 more points on these lines
    config2 = [
        [0, 0], [2, 0], [1, 1],  # on y=0 and diagonals
        [0, 2], [2, 2], [1, 3],  # on y=2 and diagonals  
        [0, 1], [2, 1], [1, 2],  # middle
        [3, 0], [3, 2], [1, 0]
    ]
    
    # Configuration 3: 3x4 grid with one point moved
    config3 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2],
        [1, 3]  # moved from [3,2] to create diagonal
    ]
    
    # Configuration 4: Hexagonal lattice subset
    config4 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1],
        [1, 2], [2, 2], [3, 2],
        [0, 3], [4, 1]
    ]
    
    return [config1, config2, config3, config4]

def _make_cross_pattern():
    """
    Cross pattern with extra points to create more collinearities.
    """
    pts = [
        [0, 0], [1, 0], [-1, 0], [2, 0],
        [0, 1], [0, -1], [0, 2],
        [1, 1], [-1, -1], [1, -1],
        [-1, 1], [2, 1]
    ]
    return pts

def _make_staggered_parabola():
    """
    Points on/near a parabola to create many 3-point lines.
    For y=x^2: line through (a,a^2) and (b,b^2) is y=(a+b)x-ab
    """
    # 6 points on parabola
    pts = [[x, x*x] for x in range(6)]
    
    # Add 6 points that lie on multiple parabola-pair lines
    # Line through (0,0) and (1,1): y=x
    # Line through (0,0) and (2,4): y=2x
    # Line through (1,1) and (2,4): y=3x-2
    # Line through (0,0) and (3,9): y=3x
    # Line through (1,1) and (3,9): y=4x-3
    # Line through (2,4) and (3,9): y=5x-6
    
    # Point (4, 4) is on y=x (through 0,1) - but (4,4) is not on parabola
    # Let's find points on multiple lines
    # (3,3): on y=x (0,1) and y=3x-6? No. y=3x gives (3,9) not (3,3)
    # Let me just try specific points and evaluate
    
    candidates = [
        [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[3,3],[4,4],[2,2],[5,5],[1,0],[0,1]],
        [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[6,36],[2,2],[3,3],[4,4],[5,5],[1,0]],
        [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[1,0],[0,1],[2,2],[3,3],[4,4],[5,5]],
    ]
    return candidates

def _make_lattice_subgrid():
    """
    Subgrid of a larger lattice with strategic point removal.
    """
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [1, 3]
    ]
    return pts

def _scale_points(pts, scale=10):
    return [[x * scale, y * scale] for x, y in pts]

def _generate_lattice_candidates(size=20):
    """Generate diverse candidate configurations from a lattice."""
    candidates = []
    
    # Random subsets of a grid
    for _ in range(20):
        pts = []
        while len(pts) < 12:
            x = random.randint(-size//2, size//2)
            y = random.randint(-size//2, size//2)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
    
    # Structured subsets
    for _ in range(10):
        pts = []
        base_x = random.randint(-size//4, size//4)
        base_y = random.randint(-size//4, size//4)
        for i in range(12):
            x = base_x + random.randint(-size//3, size//3)
            y = base_y + random.randint(-size//3, size//3)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        while len(pts) < 12:
            p = [random.randint(-size, size), random.randint(-size, size)]
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
    
    return candidates

def _make_staggered_parabola():
    """Points on/near a parabola to create many 3-point lines."""
    candidates = [
        [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[3,3],[4,4],[2,2],[5,5],[1,0],[0,1]],
        [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[6,36],[2,2],[3,3],[4,4],[5,5],[1,0]],
        [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[1,0],[0,1],[2,2],[3,3],[4,4],[5,5]],
        [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25],[0,1],[1,0],[2,2],[3,3],[4,4],[6,3]],
    ]
    return candidates

def _make_complete_quad():
    """Complete quadrilateral: 4 lines, 6 vertices, plus 6 more on the lines."""
    # 4 lines: x=0, y=0, x+y=3, x-y=1
    # Intersections: (0,0), (0,3), (0,1), (3,0), (2,1), (1,2)
    # Add 6 more points on these lines
    pts = [
        [0, 0], [0, 3], [0, 1],
        [3, 0], [2, 1], [1, 2],
        [1, 0], [0, 2], [3, 1],
        [2, 0], [1, 3], [3, 3]
    ]
    return pts

def _make_pappus_like():
    """Pappus configuration: 9 points on 9 lines of 3, extend to 12."""
    # Pappus: two lines with 3 points each, 6 cross-connections
    # Line 1: (0,0), (1,0), (2,0)
    # Line 2: (0,2), (1,2), (2,2)
    # Cross lines create 3 more points
    # Extend with 3 more points
    pts = [
        [0, 0], [1, 0], [2, 0],
        [0, 2], [1, 2], [2, 2],
        [1, 1],  # center
        [0, 1], [2, 1],
        [3, 0], [3, 2], [1, 3]
    ]
    return pts

def _aggressive_local_search(points, deadline=None):
    """
    Aggressive local search: try all possible line placements for each point,
    plus random restarts within the search.
    """
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    
    for iteration in range(200):
        if deadline and time.time() > deadline:
            break
        
        lines = _get_lines(best)
        improved = False
        
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            
            xi, yi = best[idx]
            best_move = None
            best_move_cnt = best_cnt
            
            # Strategy 1: Place on existing lines with 2+ points
            for line_key, pts_on_line in lines.items():
                if idx in pts_on_line:
                    continue
                if len(pts_on_line) < 2:
                    continue
                
                dx_n, dy_n, c = line_key
                pts_list = list(pts_on_line)
                p1 = best[pts_list[0]]
                x0, y0 = p1[0], p1[1]
                
                # Try t values near existing points
                t_values = set()
                for pidx in pts_on_line:
                    px, py = best[pidx]
                    if dx_n != 0:
                        if (px - x0) % dx_n == 0:
                            t_values.add((px - x0) // dx_n)
                    elif dy_n != 0:
                        if (py - y0) % dy_n == 0:
                            t_values.add((py - y0) // dy_n)
                
                existing_points = set(tuple(p) for p in best)
                
                # Generate candidate t values
                candidates = set()
                for t in t_values:
                    for dt in range(-5, 6):
                        candidates.add(t + dt)
                for dt in range(-10, 11):
                    candidates.add(dt)
                
                for t in candidates:
                    nx = x0 + t * dx_n
                    ny = y0 + t * dy_n
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_pt = (nx, ny)
                    if new_pt in existing_points:
                        continue
                    if dy_n * nx - dx_n * ny != c:
                        continue
                    
                    best[idx] = [nx, ny]
                    cnt = _count_ordinary_lines(best)
                    best[idx] = [xi, yi]
                    
                    if cnt < best_move_cnt:
                        best_move_cnt = cnt
                        best_move = [nx, ny]
            
            # Strategy 2: Small random moves
            for _ in range(20):
                if deadline and time.time() > deadline:
                    break
                rx = random.randint(-15, 15)
                ry = random.randint(-15, 15)
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
                if cnt < best_move_cnt:
                    best_move_cnt = cnt
                    best_move = new_pt
            
            # Strategy 3: Moves near current position
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
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
                    if cnt < best_move_cnt:
                        best_move_cnt = cnt
                        best_move = new_pt
            
            if best_move is not None:
                best[idx] = best_move
                best_cnt = best_move_cnt
                improved = True
        
        if not improved:
            # Try a random restart of one point
            idx = random.randint(0, n-1)
            best[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
            new_cnt = _count_ordinary_lines(best)
            if new_cnt < best_cnt:
                best_cnt = new_cnt
            else:
                break  # No improvement possible
    
    return best, best_cnt

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
    seeds.append(_make_hesse_like())
    seeds.append(_make_cross_pattern())
    seeds.append(_make_lattice_subgrid())
    seeds.append(_make_complete_quad())
    seeds.append(_make_pappus_like())
    
    # Parabola-based configurations
    for cfg in _make_staggered_parabola():
        seeds.append(cfg)
    
    # Scaled versions
    seeds.append(_scale_points(_make_grid(4, 3), 5))
    seeds.append(_scale_points(_make_three_parallel_lines(), 5))
    seeds.append(_scale_points(_make_hesse_like(), 3))
    seeds.append(_scale_points(_make_complete_quad(), 3))
    
    # Random configurations from lattice
    seeds.extend(_generate_lattice_candidates(size=25))
    
    # More random with wider range
    for _ in range(10):
        pts = []
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)
    
    best_pts = None
    best_cnt = float('inf')
    
    # First pass: evaluate all seeds quickly
    for seed in seeds:
        if time.time() > deadline:
            break
        cnt = _count_ordinary_lines(seed)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in seed]
    
    # Second pass: local search on top seeds
    seed_sorted = []
    for seed in seeds:
        cnt = _count_ordinary_lines(seed)
        seed_sorted.append((cnt, seed))
    seed_sorted.sort()
    
    for cnt, seed in seed_sorted:
        if time.time() > deadline:
            break
        if cnt > best_cnt + 5:
            break  # Skip seeds too far from best
        
        # Aggressive local search
        pts, cnt2 = _aggressive_local_search(seed, deadline=deadline)
        if cnt2 < best_cnt:
            best_cnt = cnt2
            best_pts = pts[:]
        
        # Also try original smart local search
        pts, cnt3 = _smart_local_search(seed, max_iters=5000, deadline=deadline)
        if cnt3 < best_cnt:
            best_cnt = cnt3
            best_pts = pts[:]
    
    # Final refinement
    if best_pts and time.time() < deadline - 15:
        pts, cnt = _aggressive_local_search(best_pts, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        pts, cnt = _smart_local_search(best_pts, max_iters=20000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        pts, cnt = _pair_swap_search(best_pts, max_iters=5000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
    
    # Final polish
    if best_pts and time.time() < deadline - 5:
        n = len(best_pts)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            lines = _get_lines(best_pts)
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                xi, yi = best_pts[idx]
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [xi + dx, yi + dy]
                        valid = True
                        for j in range(n):
                            if j != idx and best_pts[j] == new_pt:
                                valid = False
                                break
                        if not valid:
                            continue
                        best_pts[idx] = new_pt
                        cnt = _count_ordinary_lines(best_pts)
                        best_pts[idx] = [xi, yi]
                        if cnt < best_cnt:
                            best_pts[idx] = new_pt
                            best_cnt = cnt
                            improved = True
    
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
