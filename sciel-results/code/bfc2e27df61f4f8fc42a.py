import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 145.0  # Safety margin before 150s kill

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas."""
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    i_idx = np.array([c[0] for c in idx])
    j_idx = np.array([c[1] for c in idx])
    k_idx = np.array([c[2] for c in idx])
    v1 = pts[j_idx] - pts[i_idx]
    v2 = pts[k_idx] - pts[i_idx]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _soft_min(areas, tau):
    """Soft minimum via log-sum-exp (numerically stable)."""
    m = np.min(areas)
    shifted = areas - m
    shifted = np.clip(shifted, 0, 50.0 / tau)
    sum_exp = np.sum(np.exp(-tau * shifted))
    return m - (1.0 / tau) * np.log(sum_exp)

def _min_triangle_area(pts):
    """Minimum triangle area."""
    return np.min(_triangle_areas(pts))

def _hull_area(pts):
    """Convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    ma = _min_triangle_area(pts)
    return ma / ha

def _normalize(pts):
    """Center and scale to unit hull area."""
    pts = pts - np.mean(pts, axis=0)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return pts
    return pts / np.sqrt(ha)

def _neg_objective(flat):
    """Negative score after normalization."""
    pts = flat.reshape(-1, 2)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    return -_min_triangle_area(pts)

def _smooth_objective(flat, tau=50.0):
    """Smooth surrogate: minimize -soft_min(areas)/hull_area."""
    pts = flat.reshape(-1, 2)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, tau)
    return -sm / ha

def _local_refine(pts, max_iters=100, base_step=0.005):
    """Local refinement: perturb points involved in worst triangles (faster)."""
    pts = pts.copy()
    current = _score(pts)
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        # Find points in triangles with area close to minimum
        worst_tris = np.where(areas < min_a * 1.001)[0]
        worst_pts = set()
        for ti in worst_tris[:10]:
            for p in idx[ti]:
                worst_pts.add(p)
        
        improved = False
        step = base_step * (0.7 ** (it // 30))
        for pi in sorted(worst_pts):
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[pi, 0] += dx
                    new_pts[pi, 1] += dy
                    s = _score(new_pts)
                    if s > current + 1e-12:
                        pts = new_pts
                        current = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return pts, current

def construct():
    start_time = time.time()
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial configurations (reduced count)
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(x0.copy())
    
    # 2. Regular 14-gon with small noise
    configs.append(x0 + 0.03 * np.random.randn(N, 2))
    
    # 3. Two concentric rings: 7+7
    ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = ang1 + np.pi / 7
    inner = 0.55 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    configs.append(np.vstack([outer, inner]))
    
    # 4. Three rings: 5+5+4
    ang1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = ang1 + np.pi / 5
    r2 = 0.65 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    ang3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    r3 = 0.35 * np.column_stack([np.cos(ang3), np.sin(ang3)])
    configs.append(np.vstack([r1, r2, r3]))
    
    # 5. Four rings: 4+4+3+3
    ang1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    r1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = ang1 + np.pi / 4
    r2 = 0.7 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    ang3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
    r3 = 0.45 * np.column_stack([np.cos(ang3), np.sin(ang3)])
    ang4 = ang3 + np.pi / 3
    r4 = 0.2 * np.column_stack([np.cos(ang4), np.sin(ang4)])
    configs.append(np.vstack([r1, r2, r3, r4]))
    
    # 6. Random uniform in unit disk (reduced)
    for _ in range(4):
        r = np.sqrt(np.random.rand(N))
        th = np.random.rand(N) * 2 * np.pi
        configs.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    
    # 7. Random Gaussian (reduced)
    for _ in range(2):
        configs.append(np.random.randn(N, 2))
    
    # 8. Perturbed regular polygon with varying noise
    for scale in [0.05, 0.1]:
        configs.append(x0 + scale * np.random.randn(N, 2))
    
    for x_init in configs:
        if time.time() > _DEADLINE - 10:
            break
        
        try:
            x_flat = x_init.flatten()
            
            # Phase 1: Smooth optimization with Powell
            res1 = minimize(_smooth_objective, x_flat, args=(80.0,),
                          method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
            
            if time.time() > _DEADLINE - 10:
                break
            
            p1 = res1.x.reshape(-1, 2)
            p1 = _normalize(p1)
            
            # Phase 2: Exact optimization with Nelder-Mead
            res2 = minimize(_neg_objective, p1.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > _DEADLINE - 10:
                break
            
            p2 = res2.x.reshape(-1, 2)
            p2 = _normalize(p2)
            
            # Phase 3: Local refinement
            if time.time() < _DEADLINE - 15:
                p3, s3 = _local_refine(p2, max_iters=80, base_step=0.005)
            else:
                p3, s3 = p2, _score(p2)
            
            s = _score(p3)
            if s > best_score:
                best_score = s
                best_points = p3.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        best_points = _normalize(x0.copy())
    
    # Final global polish
    if best_points is not None and time.time() < _DEADLINE - 5:
        try:
            res_final = minimize(_neg_objective, best_points.flatten(),
                              method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            pf = res_final.x.reshape(-1, 2)
            pf = _normalize(pf)
            sf = _score(pf)
            if sf > best_score:
                best_points = pf
                best_score = sf
        except Exception:
            pass
    
    return best_points
