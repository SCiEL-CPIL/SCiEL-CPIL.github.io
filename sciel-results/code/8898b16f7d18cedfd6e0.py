"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter
import random

def _score(pts):
    """Compute the max multiplicity of any single positive doubled area."""
    n = len(pts)
    areas = []
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i+1, n):
            xj, yj = pts[j]
            dx1 = xj - xi
            dy1 = yj - yi
            for k in range(j+1, n):
                xk, yk = pts[k]
                det = abs(dx1*(yk - yi) - dy1*(xk - xi))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _grid_init(cols, rows, x0, y0):
    return [(x0 + c, y0 + r) for r in range(rows) for c in range(cols)]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    inits = [
        _grid_init(6, 5, -2, -2),
        _grid_init(5, 6, -2, -2),
        _grid_init(6, 5, -3, -2),
        _grid_init(5, 6, -2, -3),
        _grid_init(6, 5, -2, -3),
        _grid_init(5, 6, -3, -2),
        _grid_init(10, 3, -4, -1),
        _grid_init(3, 10, -1, -4),
    ]
    
    best_points = inits[0]
    best_score = _score(best_points)
    
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        pts = list(init_pts)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(30):
                if time.time() > deadline:
                    break
                old_pt = pts[idx]
                candidates = set()
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = old_pt[0] + dx, old_pt[1] + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            candidates.add((nx, ny))
                for _ in range(30):
                    candidates.add((random.randint(-30, 30), random.randint(-30, 30)))
                
                occupied = set(pts)
                occupied.discard(old_pt)
                candidates = [c for c in candidates if c not in occupied]
                
                best_cand = None
                best_cand_score = score
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    new_pts = list(pts)
                    new_pts[idx] = cand
                    cand_score = _score(new_pts)
                    if cand_score > best_cand_score:
                        best_cand_score = cand_score
                        best_cand = cand
                
                if best_cand is not None:
                    pts[idx] = best_cand
                    score = best_cand_score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_points = list(pts)
    
    return best_points
# EVOLVE-BLOCK-END
