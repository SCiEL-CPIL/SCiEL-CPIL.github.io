import numpy as np
import scipy.spatial as spat
import time
from scipy.optimize import minimize
import itertools

N = 14

def _hull_area(points):
    """Calculate area of convex hull."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    n = len(points)
    # Get all combinations of 3 indices
    # For N=14, C(14,3) = 364
    i, j, k = np.array(list(itertools.combinations(range(n), 3))).T
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Smooth approximation of min triangle area / hull area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    n = len(points)
    i, j, k = np.array(list(itertools.combinations(range(n), 3))).T
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    a = -beta * areas
    m = np.max(a)
    
    try:
        soft_min_val = -1.0 / beta * (m + np.log(np.sum(np.exp(a - m))))
    except:
        return 0.0
    
    return soft_min_val / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Pre-compute indices for speed
    n = N
    idx_arr = np.array(list(itertools.combinations(range(n), 3)))
    i_idx = idx_arr[:, 0]
    j_idx = idx_arr[:, 1]
    k_idx = idx_arr[:, 2]

    def fast_score(points):
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas) / hull_area

    def objective(x):
        """
        Objective for L-BFGS-B.
        x: shape (14, 2) flattened to 28.
        Returns negative of soft min score to maximize.
        """
        points = x.reshape(n, 2)
        points_centered = points - np.mean(points, axis=0)
        
        # Calculate soft min score
        beta = 50.0
        score_val = _soft_min_score(points_centered, beta)
        
        # Small regularization to keep points near origin (scale invariance)
        penalty = 0.01 * np.sum(points_centered**2)
        
        return -score_val + penalty

    # Initial candidates
    candidates = []
    
    # 1. Regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles_reg), np.sin(angles_reg)))
    candidates.append(pts_reg.flatten())
    
    # 2. Two-heptagon structure (outer and inner ring)
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    best_r = 0.5
    best_phi = 0.0
    best_score = 0.0
    best_pts_hept = None
    for r in np.linspace(0.4, 0.9, 15):
        for phi in np.linspace(0, np.pi/7, 15):
            pts_outer = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
            pts_inner = r * np.column_stack((np.cos(base_angles + phi), np.sin(base_angles + phi)))
            pts = np.vstack((pts_outer, pts_inner))
            s = fast_score(pts)
            if s > best_score:
                best_score = s
                best_r = r
                best_phi = phi
                best_pts_hept = pts
    
    if best_pts_hept is not None:
        candidates.append(best_pts_hept.flatten())
        
    # 3. Random perturbations of regular 14-gon
    np.random.seed(42)
    for _ in range(5):
        noise = np.random.normal(0, 0.1, (N, 2))
        pts_noise = pts_reg + noise
        candidates.append(pts_noise.flatten())
    
    # 4. Random points in unit disk
    for _ in range(5):
        r = np.sqrt(np.random.uniform(0, 1, N))
        theta = np.random.uniform(0, 2*np.pi, N)
        pts_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        candidates.append(pts_rand.flatten())

    best_points = None
    best_score = 0.0

    # Evaluate initial candidates
    for x in candidates:
        pts = x.reshape(N, 2)
        s = fast_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Optimize from top candidates
    scored_candidates = []
    for x in candidates:
        pts = x.reshape(N, 2)
        s = fast_score(pts)
        scored_candidates.append((s, x))
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    
    top_candidates = [x for s, x in scored_candidates[:10]]
    
    for x0 in top_candidates:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if res.success:
                pts_opt = res.x.reshape(N, 2)
                pts_opt = pts_opt - np.mean(pts_opt, axis=0)
                
                s_opt = fast_score(pts_opt)
                
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = pts_opt.copy()
                    
        except Exception as e:
            pass

    # Final refinement with higher beta (sharper min) if time permits
    if time.time() < deadline - 30 and best_points is not None:
        def objective_sharp(x):
            points = x.reshape(n, 2)
            points_centered = points - np.mean(points, axis=0)
            score_val = _soft_min_score(points_centered, beta=200.0)
            penalty = 0.01 * np.sum(points_centered**2)
            return -score_val + penalty

        try:
            res = minimize(objective_sharp, best_points.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-9})
            if res.success:
                pts_final = res.x.reshape(N, 2)
                pts_final = pts_final - np.mean(pts_final, axis=0)
                s_final = fast_score(pts_final)
                if s_final > best_score:
                    best_score = s_final
                    best_points = pts_final.copy()
        except:
            pass

    if best_points is None:
        best_points = pts_reg

    return best_points
