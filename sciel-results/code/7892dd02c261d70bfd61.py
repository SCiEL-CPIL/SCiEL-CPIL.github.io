"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, pts[i][0])
            elif dy == 0:
                key = (0, 1, pts[i][1])
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                c = dy * pts[i][0] - dx * pts[i][1]
                key = (1, dx, dy, c)
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    for j in range(1, n):
        for k in range(j + 1, n):
            x0, y0 = points[0]
            x1, y1 = points[j]
            x2, y2 = points[k]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return True
    return False

def _parabola_config(x_vals, a=1, b=0, c=0, off_pt=None):
    """Create 11 points on parabola y=a*x^2+b*x+c plus one off-point."""
    pts = []
    for x in x_vals:
        y = a * x * x + b * x + c
        pts.append([x, y])
    if off_pt is not None:
        pts.append(list(off_pt))
    return pts

def _find_best_off_point(pts11, search_range=15):
    """Find the best 12th point to minimize ordinary lines."""
    best_score = float('inf')
    best_pt = None
    existing = set((p[0], p[1]) for p in pts11)
    
    for ox in range(-search_range, search_range + 1):
        for oy in range(-search_range, search_range + 1):
            if (ox, oy) in existing:
                continue
            full = pts11 + [[ox, oy]]
            if not _not_all_collinear(full):
                continue
            score = _count_ordinary_lines(full)
            if score < best_score:
                best_score = score
                best_pt = [ox, oy]
    return best_pt, best_score

def _try_parabola_families():
    """Try various parabola configurations and find best overall."""
    best_points = None
    best_score = float('inf')
    
    # Standard y=x^2 with symmetric x-values
    x_sets = [
        list(range(-5, 6)),
        list(range(-6, 6)),
        list(range(-4, 7)),
        list(range(-7, 5)),
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6],
        [-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        [-5, -4, -3, -2, -1, 0, 1, 2, 4, 5, 6],
        [-5, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
        [-6, -5, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        [-5, -4, -3, -1, 0, 1, 2, 3, 4, 5, 6],
    ]
    
    for x_vals in x_sets:
        if len(x_vals) != 11:
            continue
        pts11 = _parabola_config(x_vals)
        off_pt, score = _find_best_off_point(pts11, search_range=20)
        if off_pt is not None and score < best_score:
            best_score = score
            best_points = pts11 + [off_pt]
    
    # Try scaled parabolas
    for a in [1, 2, 3, 4]:
        for x_vals in [list(range(-4, 7)), list(range(-5, 6))]:
            pts11 = _parabola_config(x_vals, a=a)
            off_pt, score = _find_best_off_point(pts11, search_range=20)
            if off_pt is not None and score < best_score:
                best_score = score
                best_points = pts11 + [off_pt]
    
    # Try shifted parabolas
    for b in [-2, -1, 1, 2]:
        for x_vals in [list(range(-5, 6))]:
            pts11 = _parabola_config(x_vals, b=b)
            off_pt, score = _find_best_off_point(pts11, search_range=20)
            if off_pt is not None and score < best_score:
                best_score = score
                best_points = pts11 + [off_pt]
    
    return best_points, best_score

def _try_grid_configs():
    """Try various grid configurations."""
    best_points = None
    best_score = float('inf')
    
    grids = [
        [[x, y] for x in range(4) for y in range(3)],
        [[x, y] for x in range(3) for y in range(4)],
        [[x, y] for x in range(6) for y in range(2)],
        [[x, y] for x in range(2) for y in range(6)],
        [[x + y, y] for x in range(4) for y in range(3)],
        [[x, y + x] for x in range(3) for y in range(4)],
        [[2*x, y] for x in range(4) for y in range(3)],
        [[x, 2*y] for x in range(3) for y in range(4)],
    ]
    
    for grid in grids:
        if len(grid) != 12:
            continue
        seen = set()
        valid = True
        for p in grid:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid or not _not_all_collinear(grid):
            continue
        score = _count_ordinary_lines(grid)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in grid]
    
    return best_points, best_score

def _try_hybrid_configs():
    """Try hybrid configurations mixing different structures."""
    best_points = None
    best_score = float('inf')
    
    # 3 points on one line, 4 on another, 5 on a curve
    configs = [
        # Collinear triplets
        [[0,0],[1,1],[2,2], [0,1],[1,3],[2,5],[3,7], [0,2],[1,4],[2,6],[3,8],[4,10],[5,12]],
        # Two lines with 4 points each + 4 others
        [[0,0],[1,0],[2,0],[3,0], [0,1],[1,1],[2,1],[3,1], [0,2],[1,2],[0,3],[1,3],[2,3],[3,3]],
        # Star pattern
        [[0,0],[3,0],[6,0],[9,0], [0,3],[3,3],[6,3],[9,3], [0,6],[3,6],[6,6],[9,6]],
    ]
    
    for cfg in configs:
        if len(cfg) != 12:
            continue
        seen = set()
        valid = True
        for p in cfg:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid or not _not_all_collinear(cfg):
            continue
        score = _count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
    
    return best_points, best_score

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill
    
    n = 12
    
    # Try multiple configuration families
    best_points = None
    best_score = float('inf')
    
    # 1. Parabola-based (primary strategy)
    para_pts, para_score = _try_parabola_families()
    if para_pts is not None:
        best_points = para_pts
        best_score = para_score
    
    # 2. Grid-based
    if time.time() < deadline:
        grid_pts, grid_score = _try_grid_configs()
        if grid_pts is not None and grid_score < best_score:
            best_points = grid_pts
            best_score = grid_score
    
    # 3. Hybrid
    if time.time() < deadline:
        hyb_pts, hyb_score = _try_hybrid_configs()
        if hyb_pts is not None and hyb_score < best_score:
            best_points = hyb_pts
            best_score = hyb_score
    
    # 4. Local search to improve
    if best_points is None:
        best_points = [[i, i*i] for i in range(12)]
        best_score = _count_ordinary_lines(best_points)
    
    rng = random.Random(42)
    
    def _local_search(points, max_time=60):
        """Improve configuration via local search."""
        current = [p[:] for p in points]
        current_score = _count_ordinary_lines(current)
        best_local = [p[:] for p in current]
        best_local_score = current_score
        deadline_ls = time.time() + max_time
        
        no_improve = 0
        max_no_improve = 20
        
        while no_improve < max_no_improve and time.time() < deadline_ls:
            improved = False
            
            for idx in range(n):
                if time.time() > deadline_ls:
                    break
                
                old_x, old_y = current[idx]
                existing = [(current[i][0], current[i][1]) for i in range(n) if i != idx]
                occupied = set(existing)
                
                # Generate candidate positions
                candidates = []
                
                # Positions on lines through pairs of existing points
                for i in range(len(existing)):
                    for j in range(i+1, len(existing)):
                        x1, y1 = existing[i]
                        x2, y2 = existing[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        if dx == 0 and dy == 0:
                            continue
                        g = gcd(abs(dx), abs(dy))
                        if g == 0:
                            continue
                        dxn = dx // g
                        dyn = dy // g
                        for t in range(-12, 13):
                            if t == 0 or t == 1:
                                continue
                            nx = x1 + t * dxn
                            ny = y1 + t * dyn
                            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                                if (nx, ny) not in occupied:
                                    candidates.append((nx, ny))
                
                # Random positions
                cx = sum(p[0] for p in existing) // len(existing)
                cy = sum(p[1] for p in existing) // len(existing)
                for _ in range(100):
                    nx = cx + rng.randint(-25, 25)
                    ny = cy + rng.randint(-25, 25)
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9 and (nx, ny) not in occupied:
                        candidates.append((nx, ny))
                
                rng.shuffle(candidates)
                
                for nx, ny in candidates[:200]:
                    if time.time() > deadline_ls:
                        break
                    if nx == old_x and ny == old_y:
                        continue
                    current[idx] = [nx, ny]
                    if not _not_all_collinear(current):
                        current[idx] = [old_x, old_y]
                        continue
                    score = _count_ordinary_lines(current)
                    if score < current_score:
                        current_score = score
                        improved = True
                        if score < best_local_score:
                            best_local_score = score
                            best_local = [p[:] for p in current]
                        break
                    else:
                        current[idx] = [old_x, old_y]
                
                if improved:
                    break
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        return best_local, best_local_score
    
    # Run local search
    if time.time() < deadline:
        improved_pts, improved_score = _local_search(best_points, max_time=min(90, deadline - time.time()))
        if improved_score < best_score:
            best_points = improved_pts
            best_score = improved_score
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    seen = set()
    for p in best_points:
        assert (p[0], p[1]) not in seen
        seen.add((p[0], p[1]))
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
