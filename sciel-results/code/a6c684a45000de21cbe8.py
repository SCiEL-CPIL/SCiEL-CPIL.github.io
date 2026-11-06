import time
import random
from math import gcd
from collections import defaultdict

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_map = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            if dx_n == 0:
                key = ('v', x1)
            else:
                c = dy_n * x1 - dx_n * y1
                key = ('s', dx_n, dy_n, c)
            line_map[key].add(i)
            line_map[key].add(j)
    
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y1) != dy * (x - x1):
            return True
    return False

def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    if dx_n == 0:
        return ('v', x1)
    else:
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _solve_line_intersection(p1, p2, p3, p4):
    """Find intersection of line(p1,p2) and line(p3,p4). Return None if parallel or same line."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    # Line 1: a1*x + b1*y = c1 where a1=y2-y1, b1=x1-x2, c1=a1*x1+b1*y1
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    # Line 2: a2*x + b2*y = c2
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    # Check if integer
    if abs(x - round(x)) > 1e-9 or abs(y - round(y)) > 1e-9:
        return None
    xi, yi = int(round(x)), int(round(y))
    # Verify on both lines
    if a1 * xi + b1 * yi != c1 or a2 * xi + b2 * yi != c2:
        return None
    return [xi, yi]

def _generate_line_based_configs():
    """Generate configurations by specifying line patterns and solving for points."""
    configs = []
    
    # Pattern 1: 4 lines of 3 points, arranged in a quadrilateral
    # Lines: L1 (horizontal), L2 (horizontal), L3 (diagonal), L4 (diagonal)
    # Each pair of non-parallel lines intersects at one point
    # 4 lines: 2 parallel horizontal, 2 parallel diagonal
    # Points: 4 intersections + 8 on-line points = 12
    
    # Pattern: 3 parallel lines x 4 points each, with 2 transversals
    # 3 horizontal lines at y=0,1,2 with 4 points each
    # 2 diagonal transversals each passing through 3 of the 12 points
    for h_spacing in [1, 2, 3]:
        for w_spacing in [1, 2, 3]:
            pts = []
            for row in range(3):
                for col in range(4):
                    pts.append([col * w_spacing, row * h_spacing])
            configs.append(pts)
    
    # Pattern: 4 lines of 3, forming a complete quadrilateral
    # 4 lines, each pair intersects, 6 intersection points, plus 6 more on the lines
    # Try: 4 lines through specific integer points
    # Line A: y=0, points at x=0,1,2
    # Line B: y=3, points at x=0,1,2  
    # Line C: x=0, points at y=0,3,6
    # Line D: x=3, points at y=0,3,6
    # Intersections: (0,0),(3,0),(0,3),(3,3) - 4 points
    # On A: (0,0),(1,0),(2,0),(3,0) - adds (1,0),(2,0)
    # On B: (0,3),(1,3),(2,3),(3,3) - adds (1,3),(2,3)
    # On C: (0,0),(0,3),(0,6),(0,-3) - adds (0,6),(0,-3)
    # On D: (3,0),(3,3),(3,6),(3,-3) - adds (3,6),(3,-3)
    # Total: 4 + 2+2+2+2 = 12
    quad = [[0,0],[1,0],[2,0],[3,0],[0,3],[1,3],[2,3],[3,3],[0,6],[0,-3],[3,6],[3,-3]]
    configs.append(quad)
    
    # Pattern: 3 lines of 4 points, each pair intersects
    # 3 lines, 3 intersection points, 9 more on lines = 12
    # Line 1: y=0, x in {0,1,2,3}
    # Line 2: y=x, points at (0,0),(1,1),(2,2),(3,3)
    # Line 3: y=-x+3, points at (0,3),(1,2),(2,1),(3,0)
    # Intersections: (0,0), (1.5,1.5)-not int, (3,0)
    # Let me try different: 
    # Line 1: y=0: (0,0),(1,0),(2,0),(3,0)
    # Line 2: y=2x: (0,0),(1,2),(2,4),(3,6)
    # Line 3: y=-2x+6: (0,6),(1,4),(2,2),(3,0)
    # Intersections: L1&L2: (0,0), L1&L3: (3,0), L2&L3: solve 2x=-2x+6, x=1.5 - not int
    # Try: L2: y=3x, L3: y=-3x+9
    # L2&L3: 3x=-3x+9, x=1.5 - still not
    # Try: L1: y=0, L2: y=x, L3: y=-x+4
    # L1&L2: (0,0), L1&L3: (4,0), L2&L3: x=-x+4, x=2, (2,2)
    # L1: (0,0),(1,0),(2,0),(4,0) - 4 pts
    # L2: (0,0),(1,1),(2,2),(3,3) - 4 pts
    # L3: (0,4),(1,3),(2,2),(4,0) - 4 pts
    # Union: (0,0),(1,0),(2,0),(4,0),(1,1),(2,2),(3,3),(0,4),(1,3) = 9, need 3 more
    # Add (3,0) to L1, (3,3) already there, add (2,4) to L3? No, (2,2) is on L3
    # Let me just use the 9 and add 3 strategic points
    tri3 = [[0,0],[1,0],[2,0],[4,0],[1,1],[2,2],[3,3],[0,4],[1,3]]
    # Add (3,0) to extend L1, (2,4) on L3? L3: y=-x+4, (2,2) yes. (3,1) on L3? -3+4=1 yes
    # Add (3,0) to L1, (3,1) to L3, and (0,2) somewhere
    tri3 += [[3,0],[3,1],[0,2]]
    configs.append(tri3)
    
    # Pattern: Pencil + line (11 on one line + 1 off) - known bad but include
    pencil = [[i, 0] for i in range(11)] + [[5, 1]]
    configs.append(pencil)
    
    # Pattern: 2 lines of 6
    two_lines = [[i, 0] for i in range(6)] + [[i, 3] for i in range(6)]
    configs.append(two_lines)
    
    # Pattern: 6 lines of 2 with many near-collinearities
    # 3x4 grid
    grid34 = [[x, y] for x in range(4) for y in range(3)]
    configs.append(grid34)
    
    # Pattern: 4x3 grid
    grid43 = [[x, y] for x in range(3) for y in range(4)]
    configs.append(grid43)
    
    # Pattern: 3x3 grid + 3 diagonal
    grid33_diag = [[x, y] for x in range(3) for y in range(3)] + [[3,3],[0,3],[3,0]]
    configs.append(grid33_diag)
    
    # Pattern: Hesse configuration approximation
    # 9 points from 3x3, add 3 to create more 3-point lines
    hese = [[x, y] for x in range(3) for y in range(3)]
    hese += [[0, 3], [3, 0], [1, 4]]
    configs.append(hese)
    
    # Pattern: 3 lines of 4, no common point, pairwise intersecting
    # L1: y=0: (0,0),(2,0),(4,0),(6,0)
    # L2: y=2: (1,2),(3,2),(5,2),(7,2)
    # L3: y=4: (0,4),(2,4),(4,4),(6,4)
    # Intersections: L1&L2 parallel, L1&L3 parallel, L2&L3 parallel - no good
    # Try: L1: y=0, L2: y=x, L3: y=2x+2
    # L1&L2: (0,0), L1&L3: (-1,0), L2&L3: x=2x+2, x=-2, (-2,-2)
    # L1: (-1,0),(0,0),(1,0),(2,0)
    # L2: (-2,-2),(-1,-1),(0,0),(1,1)
    # L3: (-2,-2),(-1,0),(0,2),(1,4)
    # Union: (-1,0),(0,0),(1,0),(2,0),(-2,-2),(-1,-1),(1,1),(0,2),(1,4) = 9
    # Add 3 more: (2,2) on L2? No. (3,0) on L1, (-3,-3) on L2, (-3,-4) on L3
    tri3b = [[-1,0],[0,0],[1,0],[2,0],[-2,-2],[-1,-1],[1,1],[0,2],[1,4],[3,0],[-3,-3],[-3,-4]]
    configs.append(tri3b)
    
    # Pattern: 5 lines of 3 with shared points
    # Star: 5 lines through origin, 3 points each
    # But origin counted multiple times
    # L1: y=0: (-1,0),(0,0),(1,0)
    # L2: y=x: (-1,-1),(0,0),(1,1)
    # L3: y=-x: (-1,1),(0,0),(1,-1)
    # L4: y=2x: (-1,-2),(0,0),(1,2)
    # L5: y=-2x: (-1,2),(0,0),(1,-2)
    # Union: 1 + 4*2 = 9, need 3 more
    star5 = [[-1,0],[0,0],[1,0],[-1,-1],[1,1],[-1,1],[1,-1],[-1,-2],[1,2],[-1,2],[1,-2]]
    # Add (2,0) to L1, (2,2) to L2, (2,-2) to L5
    star5 += [[2,0],[2,2],[2,-2]]
    configs.append(star5)
    
    # Pattern: 4 lines of 3, 2 pairs parallel
    # L1: y=0: (0,0),(1,0),(2,0)
    # L2: y=1: (0,1),(1,1),(2,1)
    # L3: x=0: (0,0),(0,1),(0,2)
    # L4: x=2: (2,0),(2,1),(2,2)
    # Union: (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(2,2) = 8, need 4 more
    # Add (1,2) to make L5: y=2, (3,0),(3,1),(3,2)
    four_par = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[2,2],[1,2],[3,0],[3,1],[3,2]]
    configs.append(four_par)
    
    # Pattern: 3 lines of 4, 1 pair parallel, 1 transversal
    # L1: y=0: (0,0),(1,0),(2,0),(3,0)
    # L2: y=2: (0,2),(1,2),(2,2),(3,2)
    # L3: y=x+1: (-1,0),(0,1),(1,2),(2,3)
    # Union: (0,0),(1,0),(2,0),(3,0),(0,2),(1,2),(2,2),(3,2),(-1,0),(0,1),(2,3) = 11
    # Add (1,3) to L3? No, L3: y=x+1, (1,2) already. Add (3,3)
    three_par = [[0,0],[1,0],[2,0],[3,0],[0,2],[1,2],[2,2],[3,2],[-1,0],[0,1],[2,3],[3,3]]
    configs.append(three_par)
    
    # Pattern: 6 lines of 2, arranged as edges of a hexagon
    # Hexagon vertices
    hex_pts = [[1,0],[0,1],[-1,1],[-1,0],[0,-1],[1,-1]]
    # Midpoints of edges
    hex_mids = [[1,1],[-1,1],[-1,-1],[1,-1],[0,0],[0,0]]
    # That gives duplicates. Use different hexagon
    hex2 = [[2,0],[1,1],[-1,1],[-2,0],[-1,-1],[1,-1]]
    hex_mids2 = [[3,1],[0,2],[-3,1],[-3,-1],[0,-2],[3,-1]]
    configs.append(hex2 + hex_mids2)
    
    # Pattern: 4 lines of 3, complete quadrilateral with 6 intersections
    # 4 lines, C(4,2)=6 intersection points, 6 more on lines
    # L1: y=0: (0,0),(3,0),(6,0)
    # L2: y=3: (0,3),(3,3),(6,3)
    # L3: x=0: (0,0),(0,3),(0,6)
    # L4: x=6: (6,0),(6,3),(6,6)
    # Intersections: (0,0),(6,0),(0,3),(6,3) - 4 points
    # On L1: (0,0),(3,0),(6,0) - adds (3,0)
    # On L2: (0,3),(3,3),(6,3) - adds (3,3)
    # On L3: (0,0),(0,3),(0,6) - adds (0,6)
    # On L4: (6,0),(6,3),(6,6) - adds (6,6)
    # Total: 4 + 4 = 8, need 4 more
    # Add (3,6) to L2, (0,9) to L3, (6,9) to L4, (3,-3) to L1
    quad6 = [[0,0],[3,0],[6,0],[0,3],[3,3],[6,3],[0,6],[6,6],[3,6],[0,9],[6,9],[3,-3]]
    configs.append(quad6)
    
    # Pattern: 12 points as vertices of a 4x3 grid with one row shifted
    for shift in [0, 1, 2]:
        shifted = []
        for row in range(3):
            for col in range(4):
                shifted.append([col + (row % 2) * shift, row])
        configs.append(shifted)
    
    return configs

def _local_search_line_based(points, deadline, max_radius=4):
    """Local search using line-map representation for efficient evaluation."""
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point
        for idx in range(12):
            if time.time() > deadline:
                break
            old_x, old_y = current[idx]
            
            # Try nearby positions
            for dx in range(-max_radius, max_radius + 1):
                if time.time() > deadline:
                    break
                for dy in range(-max_radius, max_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check distinctness
                    is_dup = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    current[idx] = [new_x, new_y]
                    if not _not_all_collinear(current):
                        current[idx] = [old_x, old_y]
                        continue
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        break
                    else:
                        current[idx] = [old_x, old_y]
                else:
                    continue
                break
            if improved:
                break
        
        if not improved:
            # Try random moves to escape local minima
            idx = random.randint(0, 11)
            old_x, old_y = current[idx]
            found = False
            for _ in range(80):
                new_x = old_x + random.randint(-30, 30)
                new_y = old_y + random.randint(-30, 30)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                is_dup = False
                for j in range(12):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                current[idx] = [new_x, new_y]
                if not _not_all_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                new_score = _count_ordinary(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    found = True
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
            if not found:
                break
    
    return best, best_score

def _collinearity_repair(points, deadline):
    """Try to move points onto existing lines to create 3+ point lines."""
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        
        # Build line map
        line_map = defaultdict(list)
        for i in range(12):
            x1, y1 = current[i]
            for j in range(i+1, 12):
                x2, y2 = current[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                if dx_n == 0:
                    key = ('v', x1)
                else:
                    c = dy_n * x1 - dx_n * y1
                    key = ('s', dx_n, dy_n, c)
                line_map[key].append((i, j))
        
        # Find 2-point lines (ordinary lines)
        ordinary_lines = [(k, v) for k, v in line_map.items() if len(v) == 1]
        
        if not ordinary_lines:
            break
        
        # Try to move a point onto an ordinary line to make it 3-point
        for key, pairs in ordinary_lines:
            if time.time() > deadline:
                break
            i, j = pairs[0]
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Try to move each other point k onto this line
            for k in range(12):
                if k == i or k == j:
                    continue
                if time.time() > deadline:
                    break
                xk, yk = current[k]
                
                # Try t values along the line
                for t in range(-15, 16):
                    if time.time() > deadline:
                        break
                    new_x = x1 + t * dx_n
                    new_y = y1 + t * dy_n
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if new_x == xk and new_y == yk:
                        continue
                    # Check distinctness
                    is_dup = False
                    for m in range(12):
                        if m != k and current[m][0] == new_x and current[m][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    current[k] = [new_x, new_y]
                    if not _not_all_collinear(current):
                        current[k] = [xk, yk]
                        continue
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        break
                    else:
                        current[k] = [xk, yk]
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best, best_score

def _random_configs(n_configs=30):
    """Generate random configurations with structure."""
    configs = []
    for _ in range(n_configs):
        pts = []
        attempts = 0
        while len(pts) < 12 and attempts < 500:
            # Mix of random and structured
            if random.random() < 0.5:
                x = random.randint(-15, 15)
                y = random.randint(-15, 15)
            else:
                # Try to be on a line with existing points
                if len(pts) >= 2:
                    i, j = random.sample(range(len(pts)), 2)
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    g = gcd(abs(dx), abs(dy)) if g != 0 else 1
                    if g != 0:
                        dx_n = dx // g
                        dy_n = dy // g
                        t = random.randint(-5, 5)
                        x = x1 + t * dx_n
                        y = y1 + t * dy_n
                    else:
                        x = random.randint(-15, 15)
                        y = random.randint(-15, 15)
                else:
                    x = random.randint(-15, 15)
                    y = random.randint(-15, 15)
            if [x, y] not in pts:
                pts.append([x, y])
            attempts += 1
        if len(pts) == 12 and _not_all_collinear(pts):
            configs.append(pts)
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Generate all candidate configurations
    candidates = _generate_line_based_configs()
    candidates.extend(_random_configs(30))
    
    # Evaluate all candidates
    best_config = None
    best_score = float('inf')
    
    for cfg in candidates:
        if time.time() > deadline - 20:
            break
        if len(cfg) != 12:
            continue
        if len(set(tuple(p) for p in cfg)) != 12:
            continue
        if not _not_all_collinear(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_config = [p[:] for p in cfg]
    
    if best_config is None:
        return [[x, y] for x in range(4) for y in range(3)]
    
    # Phase 1: Collinearity repair (30% of time)
    repair_deadline = time.time() + (deadline - time.time()) * 0.3
    repair_best, repair_score = _collinearity_repair(best_config, repair_deadline)
    if repair_score < best_score:
        best_score = repair_score
        best_config = repair_best
    
    # Phase 2: Local search (40% of time)
    local_deadline = time.time() + (deadline - time.time()) * 0.4
    best_config, best_score = _local_search_line_based(best_config, local_deadline)
    
    # Phase 3: Try from other promising configs
    for cfg in candidates:
        if time.time() > deadline - 15:
            break
        if len(cfg) != 12:
            continue
        score = _count_ordinary(cfg)
        if score <= best_score + 3:
            sub_deadline = min(deadline - 5, time.time() + (deadline - time.time()) * 0.1)
            cfg_repair, cfg_repair_score = _collinearity_repair(cfg, sub_deadline)
            if cfg_repair_score < score:
                cfg = cfg_repair
                score = cfg_repair_score
            cfg_best, cfg_score = _local_search_line_based(cfg, deadline - 5)
            if cfg_score < best_score:
                best_score = cfg_score
                best_config = cfg_best
    
    # Phase 4: Final polish
    if time.time() < deadline - 3:
        final_best, final_score = _local_search_line_based(best_config, deadline - 2)
        if final_score < best_score:
            best_score = final_score
            best_config = final_best
    
    return best_config
