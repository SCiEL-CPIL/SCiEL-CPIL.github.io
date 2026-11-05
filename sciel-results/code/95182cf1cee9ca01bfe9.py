import time
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 140.0

def _line_key(x1, y1, x2, y2):
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_ordinary(points):
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)

def _collinear_check(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)

def _build_line_data(points):
    """Return (line_counts dict, point_lines: dict point_index -> set of line keys)."""
    n = len(points)
    line_counts = {}
    point_lines = [set() for _ in range(n)]
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
            point_lines[i].add(key)
            point_lines[j].add(key)
    return line_counts, point_lines

def _score_from_line_counts(line_counts):
    return sum(1 for c in line_counts.values() if c == 1)

def _greedy_complete_lines(points, max_rounds=500):
    """
    Greedy strategy: repeatedly find a 2-point line and try to add a point 
    on it (by moving another point onto it) to make it a 3-point line.
    """
    cur = [p[:] for p in points]
    if not _valid(cur):
        return cur, _count_ordinary(cur)
    
    best = [p[:] for p in cur]
    best_score = _count_ordinary(cur)
    
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        
        # Build line data
        line_counts, point_lines = _build_line_data(cur)
        cur_score = _score_from_line_counts(line_counts)
        
        if cur_score < best_score:
            best_score = cur_score
            best = [p[:] for p in cur]
        
        if best_score <= 6:
            break
        
        # Find all 2-point lines
        two_point_lines = [key for key, cnt in line_counts.items() if cnt == 2]
        if not two_point_lines:
            break
        
        # For each 2-point line, try to move a point onto it
        # Score improvement: converting a 2-pt line to 3-pt line saves 1 ordinary line,
        # but the moved point's old lines may become 2-pt (cost) or 3-pt->2-pt (cost)
        
        best_move = None
        best_move_score = cur_score
        
        # For each 2-point line, find the two points on it
        for line_key in two_point_lines:
            A, B, C = line_key
            # Find which points are on this line
            pts_on_line = []
            for idx, (px, py) in enumerate(cur):
                if A * px + B * py + C == 0:
                    pts_on_line.append(idx)
            
            if len(pts_on_line) != 2:
                continue
            
            p1_idx, p2_idx = pts_on_line
            
            # Try moving each other point onto this line
            for move_idx in range(12):
                if move_idx in (p1_idx, p2_idx):
                    continue
                
                # Try to find a position on this line for the moved point
                # The line has equation A*x + B*y + C = 0
                # We need integer solutions. Parametrize:
                # If A != 0: x = -B*t - C/A ... need A|C for integer, or use param
                # Better: find integer point on line near the moved point's current position
                
                mx, my = cur[move_idx]
                
                # Find integer points on the line
                # A*x + B*y + C = 0
                # If gcd(A,B) divides C, there are solutions
                g_ab = gcd(abs(A), abs(B))
                if g_ab == 0:
                    continue
                if C % g_ab != 0:
                    continue
                
                # Particular solution
                A1 = A // g_ab
                B1 = B // g_ab
                C1 = C // g_ab
                # A1*x + B1*y = -C1
                # Find one solution
                # Extended Euclidean for A1*x + B1*y = 1
                def egcd(a, b):
                    if b == 0:
                        return (a, 1, 0)
                    g, x1, y1 = egcd(b, a % b)
                    return (g, y1, x1 - (a // b) * y1)
                
                g, x0, y0 = egcd(abs(A1), abs(B1))
                if A1 < 0:
                    x0 = -x0
                if B1 < 0:
                    y0 = -y0
                # x0*A1 + y0*B1 = 1
                # Particular: x_p = -C1*x0, y_p = -C1*y0
                xp = -C1 * x0
                yp = -C1 * y0
                # General: x = xp + B1*t, y = yp - A1*t
                # Find t closest to making (x,y) near (mx,my)
                # Minimize (xp + B1*t - mx)^2 + (yp - A1*t - my)^2
                # d/dt: 2*B1*(xp + B1*t - mx) - 2*A1*(yp - A1*t - my) = 0
                # B1*(xp - mx) + B1^2*t - A1*(yp - my) + A1^2*t = 0
                # t*(A1^2 + B1^2) = A1*(yp - my) - B1*(xp - mx)
                denom = A1 * A1 + B1 * B1
                if denom == 0:
                    continue
                num = A1 * (yp - my) - B1 * (xp - mx)
                t_center = num / denom
                t1 = int(round(t_center))
                t2 = t1 + 1
                t3 = t1 - 1
                
                best_pos = None
                best_dist = float('inf')
                for t in [t1, t2, t3]:
                    nx = xp + B1 * t
                    ny = yp - A1 * t
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check not already occupied
                    if (nx, ny) in set(tuple(p) for p in cur):
                        continue
                    dist = (nx - mx)**2 + (ny - my)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nx, ny)
                
                if best_pos is None:
                    continue
                
                # Evaluate: move point move_idx to best_pos
                new_pts = [p[:] for p in cur]
                new_pts[move_idx] = list(best_pos)
                
                if not _valid(new_pts):
                    continue
                
                new_score = _count_ordinary(new_pts)
                if new_score < best_move_score:
                    best_move_score = new_score
                    best_move = (move_idx, best_pos)
        
        if best_move is None:
            # Try a different strategy: swap two points
            # Or try moving a point to create a new 3-point line
            # For now, try random perturbations
            import random
            rng = random.Random(rnd)
            improved = False
            for attempt in range(20):
                move_idx = rng.randint(0, 11)
                mx, my = cur[move_idx]
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = mx + dx, my + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                if (nx, ny) in set(tuple(p) for p in cur):
                    continue
                new_pts = [p[:] for p in cur]
                new_pts[move_idx] = [nx, ny]
                if not _valid(new_pts):
                    continue
                new_score = _count_ordinary(new_pts)
                if new_score < cur_score:
                    cur = new_pts
                    improved = True
                    break
            if not improved:
                break
        else:
            move_idx, new_pos = best_move
            cur[move_idx] = list(new_pos)
    
    if _count_ordinary(cur) < best_score:
        best = [p[:] for p in cur]
        best_score = _count_ordinary(cur)
    
    return best, best_score

def _try_structured_configs():
    """Generate structured configurations known to have few ordinary lines."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid  
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 3x3 grid + 3 points on extensions
    base33 = [[x, y] for x in range(3) for y in range(3)]
    # Add (3,0),(3,1),(3,2) -> 4x3
    configs.append(base33 + [[3, 0], [3, 1], [3, 2]])
    # Add (0,3),(1,3),(2,3) -> 3x4
    configs.append(base33 + [[0, 3], [1, 3], [2, 3]])
    # Add (3,3),(3,0),(0,3)
    configs.append(base33 + [[3, 3], [3, 0], [0, 3]])
    # Add (3,3),(-1,-1),(3,-1)
    configs.append(base33 + [[3, 3], [-1, -1], [3, -1]])
    # Add diagonal extensions
    configs.append(base33 + [[3, 3], [-1, -1], [3, -1]])
    configs.append(base33 + [[-1, 3], [3, -1], [3, 3]])
    
    # Pappus configuration variants
    # Two lines with 4 points each + 4 more
    configs.append([[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[0, 2], [1, 2], [2, 2], [3, 2]])
    
    # Triangle with 4 points per side
    configs.append([
        [0,0],[1,0],[2,0],[3,0],
        [3,1],[3,2],[3,3],
        [2,3],[1,3],[0,3],
        [1,1],[2,2]
    ])
    
    # 2x2 blocks of 3
    configs.append([
        [0,0],[1,0],[2,0],
        [0,1],[1,1],[2,1],
        [0,2],[1,2],[2,2],
        [3,0],[3,1],[3,2]
    ])
    
    # Non-uniform: use larger gaps to create more collinearities
    configs.append([[0,0],[1,0],[3,0],[6,0],
                    [0,1],[1,1],[3,1],[6,1],
                    [0,2],[1,2],[3,2],[6,2]])
    
    # Parabolic: (t, t^2)
    configs.append([[t, t*t] for t in range(12)])
    configs.append([[t, t*t] for t in range(-5, 7)])
    
    # Cubic: (t, t^3) mod something... just use small
    configs.append([[t, t*t*t] for t in range(12)])
    
    # Hexagonal lattice subset
    configs.append([
        [0,0],[1,0],[2,0],
        [0,1],[1,1],[2,1],
        [0,2],[1,2],[2,2],
        [3,1],[3,2],[3,3]
    ])
    
    # 3 concurrent lines with 4 points each (sharing center)
    # Line 1: y=0: (0,0),(1,0),(2,0),(3,0)
    # Line 2: x=0: (0,0),(0,1),(0,2),(0,3)
    # Line 3: y=x: (0,0),(1,1),(2,2),(3,3)
    # Distinct: 1+3+3+3 = 10, need 2 more
    configs.append([
        [0,0],[1,0],[2,0],[3,0],
        [0,1],[0,2],[0,3],
        [1,1],[2,2],[3,3],
        [4,0],[0,4]
    ])
    
    # 4 lines through 2 pairs of parallel lines
    configs.append([
        [0,0],[1,0],[2,0],[3,0],
        [0,1],[1,1],[2,1],[3,1],
        [0,2],[1,2],[2,2],[3,2]
    ])
    
    # Try: 12 points as vertices of a regular-ish polygon (approximate with integers)
    # Just use a circle approximation
    import math
    pts_circle = []
    for i in range(12):
        angle = 2 * math.pi * i / 12
        x = round(10 * math.cos(angle))
        y = round(10 * math.sin(angle))
        pts_circle.append([x, y])
    # Deduplicate
    seen = set()
    unique_pts = []
    for p in pts_circle:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_pts.append(p)
    if len(unique_pts) >= 12:
        configs.append(unique_pts[:12])
    
    return configs

def _local_search(points, max_iters=100000):
    """Fast local search with greedy line completion."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    if time.time() > _DEADLINE - 5:
        return best, best_score
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        if best_score <= 6:
            break
        
        # Build line data
        line_counts, point_lines = _build_line_data(cur)
        
        # Find 2-point lines and try to complete them
        two_lines = [k for k, c in line_counts.items() if c == 2]
        if not two_lines:
            break
        
        best_move = None
        best_move_score = cur_score
        
        for lk in two_lines:
            A, B, C = lk
            pts_on = [i for i, (px, py) in enumerate(cur) if A*px + B*py + C == 0]
            if len(pts_on) != 2:
                continue
            p1, p2 = pts_on
            
            for mi in range(12):
                if mi in (p1, p2):
                    continue
                mx, my = cur[mi]
                
                # Find integer point on line A*x+B*y+C=0 near (mx,my)
                g_ab = gcd(abs(A), abs(B))
                if g_ab == 0 or C % g_ab != 0:
                    continue
                A1, B1, C1 = A//g_ab, B//g_ab, C//g_ab
                
                def egcd(a, b):
                    if b == 0:
                        return (a, 1, 0)
                    g, x1, y1 = egcd(b, a % b)
                    return (g, y1, x1 - (a//b)*y1)
                
                g, x0, y0 = egcd(abs(A1), abs(B1))
                if A1 < 0: x0 = -x0
                if B1 < 0: y0 = -y0
                xp, yp = -C1*x0, -C1*y0
                denom = A1*A1 + B1*B1
                if denom == 0:
                    continue
                num = A1*(yp - my) - B1*(xp - mx)
                tc = num / denom
                t1 = int(round(tc))
                
                occupied = set(tuple(p) for p in cur)
                best_pos = None
                best_d = float('inf')
                for t in [t1-1, t1, t1+1]:
                    nx, ny = xp + B1*t, yp - A1*t
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    if (nx, ny) in occupied:
                        continue
                    d = (nx-mx)**2 + (ny-my)**2
                    if d < best_d:
                        best_d = d
                        best_pos = (nx, ny)
                
                if best_pos is None:
                    continue
                
                new_pts = [p[:] for p in cur]
                new_pts[mi] = list(best_pos)
                if not _valid(new_pts):
                    continue
                ns = _count_ordinary(new_pts)
                if ns < best_move_score:
                    best_move_score = ns
                    best_move = (mi, best_pos)
        
        if best_move is None:
            # Try random small moves
            import random
            rng = random.Random(it)
            moved = False
            for _ in range(10):
                mi = rng.randint(0, 11)
                dx, dy = rng.randint(-2, 2), rng.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cur[mi][0]+dx, cur[mi][1]+dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                if (nx, ny) in set(tuple(p) for p in cur):
                    continue
                new_pts = [p[:] for p in cur]
                new_pts[mi] = [nx, ny]
                if not _valid(new_pts):
                    continue
                ns = _count_ordinary(new_pts)
                if ns < cur_score:
                    cur = new_pts
                    cur_score = ns
                    moved = True
                    if ns < best_score:
                        best_score = ns
                        best = [p[:] for p in cur]
                    break
            if not moved:
                break
        else:
            mi, pos = best_move
            cur[mi] = list(pos)
            cur_score = best_move_score
            if cur_score < best_score:
                best_score = cur_score
                best = [p[:] for p in cur]
    
    return best, best_score

def construct():
    best_pts = None
    best_score = float('inf')
    
    # Try all structured configs
    configs = _try_structured_configs()
    for cfg in configs:
        if not _valid(cfg):
            continue
        s = _count_ordinary(cfg)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in cfg]
    
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Run local search from best config
    if time.time() < _DEADLINE - 10:
        pts, score = _local_search(best_pts, max_iters=50000)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    # Run local search from other good configs
    if time.time() < _DEADLINE - 15:
        tried = set()
        for cfg in configs:
            if not _valid(cfg):
                continue
            key = tuple(sorted(tuple(p) for p in cfg))
            if key in tried:
                continue
            tried.add(key)
            if time.time() >= _DEADLINE - 10:
                break
            pts, score = _local_search(cfg, max_iters=30000)
            if score < best_score:
                best_pts = pts
                best_score = score
            if best_score <= 6:
                break
    
    # Final greedy completion pass
    if time.time() < _DEADLINE - 5:
        pts, score = _greedy_complete_lines(best_pts, max_rounds=200)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
