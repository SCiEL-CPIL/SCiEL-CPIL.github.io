import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute all triplets
_tri_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIS = len(_tri_indices)

def _clamp_to_triangle(pts):
    """Clamp points to be inside the triangle using barycentric coords."""
    x = pts[:, 0]
    y = pts[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    # Clamp
    eps = 1e-12
    a = np.clip(a, eps, 1.0)
    b = np.clip(b, eps, 1.0)
    c = np.clip(c, eps, 1.0)
    s = a + b + c
    a, b, c = a/s, b/s, c/s
    x_new = b * 1.0 + c * 0.5
    y_new = c * H
    return np.column_stack([x_new, y_new])

def _all_areas(pts):
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    return float(np.min(_all_areas(pts)))

def _init_structured():
    """Structured initialization: vertices, midpoints, centroid, and symmetric ring."""
    pts = []
    # Vertices
    pts.append(V0.copy())
    pts.append(V1.copy())
    pts.append(V2.copy())
    # Midpoints
    pts.append((V0 + V1) / 2)
    pts.append((V1 + V2) / 2)
    pts.append((V2 + V0) / 2)
    # Centroid
    pts.append((V0 + V1 + V2) / 3)
    # 4 points in a small ring around centroid
    cx, cy = 0.5, H/3
    r = 0.12
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append(np.array([px, py]))
    return np.array(pts)

def _init_random(seed):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        pts.append(np.array([r1 + 0.5*r2, H*r2]))
    return np.array(pts)

def _optimize_triplet(pts, tri_idx, deadline, time_left):
    """
    Optimize the 3 points in tri_idx to maximize their triangle area,
    keeping other points fixed. Uses L-BFGS-B with bounds.
    Returns updated pts.
    """
    if time_left() < 2.0:
        return pts
    
    i, j, k = tri_idx
    fixed_indices = [m for m in range(N) if m not in (i, j, k)]
    
    # Objective: maximize area of triangle (i, j, k)
    # Variables: 6 coords for points i, j, k
    def obj(flat):
        p_i = flat[0:2]
        p_j = flat[2:4]
        p_k = flat[4:6]
        dx1 = p_j[0] - p_i[0]
        dy1 = p_j[1] - p_i[1]
        dx2 = p_k[0] - p_i[0]
        dy2 = p_k[1] - p_i[1]
        area = 0.5 * abs(dx1*dy2 - dy1*dx2)
        return -area
    
    # Initial values
    x0 = np.concatenate([pts[i], pts[j], pts[k]])
    
    # Bounds: each coord in [0, 1] (roughly, will clamp after)
    bounds = [(0.0, 1.0)] * 6
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success:
            new_pts = pts.copy()
            new_pts[i] = res.x[0:2]
            new_pts[j] = res.x[2:4]
            new_pts[k] = res.x[4:6]
            new_pts = _clamp_to_triangle(new_pts)
            return new_pts
    except Exception:
        pass
    return pts

def _critical_triplet_search(pts, deadline, time_left, max_rounds=50):
    """
    Iteratively find the minimum area triplet and optimize it.
    """
    current = pts.copy()
    current_score = _min_area(current)
    best = current.copy()
    best_score = current_score
    
    for round_num in range(max_rounds):
        if time_left() < 5.0:
            break
        
        areas = _all_areas(current)
        min_idx = np.argmin(areas)
        min_val = areas[min_idx]
        
        # If min area is very small, we're stuck; try a different strategy
        if min_val < 1e-8:
            break
        
        # Optimize this triplet
        tri = _tri_indices[min_idx]
        new_pts = _optimize_triplet(current, tri, deadline, time_left)
        
        if new_pts is not None:
            new_score = _min_area(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
            else:
                # No improvement from this triplet; try a few others
                # Find next few smallest areas
                sorted_idx = np.argsort(areas)
                improved = False
                for idx in sorted_idx[1:10]:
                    if time_left() < 3.0:
                        break
                    tri2 = _tri_indices[idx]
                    new_pts2 = _optimize_triplet(current, tri2, deadline, time_left)
                    if new_pts2 is not None:
                        new_score2 = _min_area(new_pts2)
                        if new_score2 > current_score:
                            current = new_pts2
                            current_score = new_score2
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best = current.copy()
                            break
                if not improved:
                    # Stagnation: break out
                    break
    
    return best, best_score

def _nelder_mead_polish(pts, deadline, time_left):
    """Final polish using Nelder-Mead on the hard objective."""
    if time_left() < 10.0:
        return pts, _min_area(pts)
    
    def obj(flat):
        p = flat.reshape(-1, 2)
        p = _clamp_to_triangle(p)
        return -_min_area(p)
    
    try:
        res = minimize(obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        new_pts = _clamp_to_triangle(res.x.reshape(-1, 2))
        return new_pts, _min_area(new_pts)
    except Exception:
        return pts, _min_area(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Multiple initializations
    inits = [_init_structured()]
    for seed in [42, 123, 456, 789, 101, 202]:
        if time_left() < 20.0:
            break
        inits.append(_init_random(seed))
    
    for init in inits:
        if time_left() < 15.0:
            break
        
        try:
            # Phase 1: Critical triplet search
            pts1, score1 = _critical_triplet_search(init, deadline, time_left, max_rounds=30)
            
            # Phase 2: Nelder-Mead polish
            pts2, score2 = _nelder_mead_polish(pts1, deadline, time_left)
            
            if score2 > best_score:
                best_score = score2
                best_pts = pts2.copy()
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _init_structured()
        best_score = _min_area(best_pts)
    
    # Final check and clamp
    best_pts = _clamp_to_triangle(best_pts)
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area(pts)}")
