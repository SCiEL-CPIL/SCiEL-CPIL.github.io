"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar, differential_evolution

# Pre-compute indices for N=14 for fast triangle area calculation
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, beta):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - (1.0 / beta) * log_sum

def _two_ring_score(theta_inner, theta_outer, r_inner):
    """
    Compute score for a two-ring configuration.
    theta_inner: angles for inner ring (len 7)
    theta_outer: angles for outer ring (len 7)
    r_inner: radius of inner ring (outer is 1.0)
    """
    inner_pts = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    pts = np.vstack([inner_pts, outer_pts])
    return _score(pts)

def _three_ring_points(r1, r2, theta1, theta2, theta3):
    """
    Three-ring configuration with 5, 4, 5 points.
    r1, r2: radii (outer is 1.0)
    theta1: angles for innermost ring (5 points)
    theta2: angles for middle ring (4 points)
    theta3: angles for outer ring (5 points)
    """
    pts1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    pts2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    pts3 = np.column_stack([np.cos(theta3), np.sin(theta3)])
    return np.vstack([pts1, pts2, pts3])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = -np.inf
    best_pts = None
    
    # Helper to update best
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(reg_pts)
    
    # 2. Two-ring optimization (7+7)
    def neg_two_ring_score(r):
        theta_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        theta_outer = theta_inner + np.pi / 7
        inner_pts = np.column_stack([r * np.cos(theta_inner), r * np.sin(theta_inner)])
        outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
        pts = np.vstack([inner_pts, outer_pts])
        return -_score(pts)
    
    if time.time() < deadline - 30.0:
        try:
            res = minimize_scalar(neg_two_ring_score, bounds=(0.1, 0.9), method='bounded',
                                   options={'xatol': 1e-6})
            r_opt = res.x
            theta_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            theta_outer = theta_inner + np.pi / 7
            inner_pts = np.column_stack([r_opt * np.cos(theta_inner), r_opt * np.sin(theta_inner)])
            outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
            pts = np.vstack([inner_pts, outer_pts])
            update_best(pts)
        except Exception:
            pass
    
    # 3. Three-ring optimization (5+4+5)
    if time.time() < deadline - 35.0:
        def neg_three_ring_opt(x):
            r1 = x[0]
            r2 = x[1]
            theta1 = x[2:7]
            theta2 = x[7:11]
            theta3 = x[11:16]
            pts = _three_ring_points(r1, r2, theta1, theta2, theta3)
            return -_score(pts)
        
        # Bounds
        bounds = [(0.2, 0.7), (0.5, 0.9)] + [(0, 2 * np.pi)] * 14
        
        # Symmetric initial guess
        theta1_init = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        theta2_init = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        theta3_init = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        x0 = np.concatenate([[0.4, 0.7], theta1_init, theta2_init, theta3_init])
        
        try:
            res = differential_evolution(
                neg_three_ring_opt, 
                bounds, 
                seed=42, 
                maxiter=150, 
                tol=1e-6,
                polish=True
            )
            if res.fun < 0:
                r1_opt = res.x[0]
                r2_opt = res.x[1]
                theta1_opt = res.x[2:7]
                theta2_opt = res.x[7:11]
                theta3_opt = res.x[11:16]
                pts = _three_ring_points(r1_opt, r2_opt, theta1_opt, theta2_opt, theta3_opt)
                update_best(pts)
        except Exception:
            pass
    
    # 4. Local refinement using L-BFGS-B on the full 28-dim space
    if best_pts is not None and time.time() < deadline - 20.0:
        def neg_score_full(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        def neg_soft_score_full(x, beta):
            pts = x.reshape(-1, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-9:
                return 1e10
            soft_min = _soft_min_area(pts, beta)
            return -(soft_min / hull_area)
        
        current_pts = best_pts.copy()
        current_score = best_score
        
        # Continuation with multiple betas
        betas = [5.0, 20.0, 100.0, 500.0]
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    neg_soft_score_full,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < -current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = -res.fun
                    update_best(current_pts)
            except Exception:
                pass
        
        # Final Powell refinement
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    neg_score_full,
                    current_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-14}
                )
                if res.fun < -current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = -res.fun
                    update_best(current_pts)
            except Exception:
                pass
    
    # 5. Random restarts with local optimization
    if time.time() < deadline - 25.0:
        rng = np.random.RandomState(999)
        for restart in range(5):
            if time.time() > deadline - 15.0:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Near regular polygon
                noise = rng.normal(0, 0.05, size=(N, 2))
                init_pts = reg_pts + noise
            elif restart == 1:
                # Two-ring with random inner radius
                r = rng.uniform(0.3, 0.7)
                theta_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(-0.1, 0.1, 7)
                theta_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7 + rng.uniform(-0.1, 0.1, 7)
                inner_pts = np.column_stack([r * np.cos(theta_inner), r * np.sin(theta_inner)])
                outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
                init_pts = np.vstack([inner_pts, outer_pts])
            elif restart == 2:
                # Three-ring random
                r1 = rng.uniform(0.2, 0.5)
                r2 = rng.uniform(0.5, 0.8)
                theta1 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rng.uniform(-0.2, 0.2, 5)
                theta2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + rng.uniform(-0.2, 0.2, 4)
                theta3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rng.uniform(-0.2, 0.2, 5)
                init_pts = _three_ring_points(r1, r2, theta1, theta2, theta3)
            else:
                # Random convex points
                angles = np.sort(rng.uniform(0, 2 * np.pi, N))
                radii = rng.uniform(0.5, 1.0, N)
                init_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            update_best(init_pts)
            
            # Quick local refinement
            if time.time() < deadline - 10.0:
                try:
                    res = minimize(
                        neg_score_full,
                        init_pts.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-12}
                    )
                    if res.fun < -best_score:
                        update_best(res.x.reshape(-1, 2))
                except Exception:
                    pass
    
    if best_pts is None:
        return reg_pts
        
    return best_pts
# EVOLVE-BLOCK-END
