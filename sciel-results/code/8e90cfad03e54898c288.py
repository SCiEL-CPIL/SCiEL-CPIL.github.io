import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

def _vertex_scores(pts):
    n = len(pts)
    c = pts.mean(axis=0)
    p = pts - c
    dcent = np.linalg.norm(p, axis=1)
    mean_d = dcent.mean()
    if mean_d <= 0.01:
        return 0.0
    p = p / mean_d
    scores = []
    for i in range(n):
        diff = p - p[i]
        dists = np.linalg.norm(diff, axis=1)
        dists = np.delete(dists, i)
        dists = np.sort(dists)
        if dists.min() < 0.05:
            scores.append(0.0)
            continue
        gaps = np.diff(dists)
        k = int(np.argmin(gaps))
        D = 0.5*(dists[k] + dists[k+1])
        if D < 0.1:
            scores.append(0.0)
            continue
        if k == 0:
            four = dists[0:4]
        elif k == 17:
            four = dists[15:19]
        else:
            four = dists[k-1:k+3]
        ratios = np.maximum(four/D, D/four)
        raw = ratios.mean()
        scores.append(min(1.0, 1.0/raw))
    return min(scores)

def _crosses(pts):
    n = len(pts)
    cr = []
    for i in range(n):
        a = pts[i]; b = pts[(i+1)%n]; c = pts[(i+2)%n]
        v1 = b-a; v2 = c-b
        cr.append(v1[0]*v2[1]-v1[1]*v2[0])
    return np.array(cr)

def _convex_violation(pts):
    cr = _crosses(pts)
    if np.any(np.abs(cr) < 1e-7):
        return 1.0 + np.abs(cr).min()
    # mixed signs?
    if np.all(cr > 0) or np.all(cr < 0):
        return 0.0
    # magnitude of inconsistency
    s = np.sign(cr[0])
    bad = cr * s
    return 1.0 + np.abs(bad[bad < 0]).max()

def _pts_from_params(params, M=7):
    n = 20
    theta = 2*np.pi*np.arange(n)/n
    r = np.ones(n)
    idx = 0
    for m in range(1, M+1):
        a = params[idx]; b = params[idx+1]; idx += 2
        r = r + a*np.cos(m*theta) + b*np.sin(m*theta)
    x = r*np.cos(theta); y = r*np.sin(theta)
    return np.stack([x,y],axis=1)

def _objective(params, M=7):
    pts = _pts_from_params(params, M)
    # repeated points check
    for i in range(20):
        for j in range(i+1,20):
            if np.linalg.norm(pts[i]-pts[j]) < 1e-6:
                return 5.0
    viol = _convex_violation(pts)
    score = _vertex_scores(pts)
    if viol > 0:
        return 1.0 + 10.0*viol + (1.0-score)
    return 1.0 - score

def _local_refine(pts, max_iter=5000):
    """Local refinement by directly optimizing vertex positions with convexity constraint."""
    n = 20
    
    def _obj_pos(params):
        p = params.reshape(20, 2)
        # Check for repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(p[i]-p[j]) < 1e-6:
                    return 10.0
        viol = _convex_violation(p)
        if viol > 0:
            return 1.0 + 5.0*viol
        s = _vertex_scores(p)
        return 1.0 - s
    
    best_score = _vertex_scores(pts)
    best_pts = pts.copy()
    
    # Try multiple random perturbations
    for attempt in range(10):
        np.random.seed(attempt * 123)
        perturbation = 0.01 * np.random.randn(20, 2)
        init = (pts + perturbation).flatten()
        
        try:
            res = minimize(_obj_pos, init, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12})
            test_pts = res.x.reshape(20, 2)
            if _convex_violation(test_pts) == 0:
                s = _vertex_scores(test_pts)
                if s > best_score:
                    best_score = s
                    best_pts = test_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    M = 7
    best_pts = None
    best_score = 0.0
    
    # Regular polygon as baseline
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    reg_score = _vertex_scores(reg_pts)
    best_pts = reg_pts
    best_score = reg_score
    
    # Try multiple DE runs with different seeds and increasing M
    for M_try in [3, 5, 7, 9]:
        for seed in [0, 1, 42, 123, 999]:
            np.random.seed(seed)
            # Smaller bounds for higher M to maintain convexity
            bound = 0.2 / np.sqrt(M_try)
            bounds = [(-bound, bound)] * (2 * M_try)
            
            def obj_local(params):
                pts = _pts_from_params(params, M_try)
                viol = _convex_violation(pts)
                if viol > 0:
                    return 1.0 + 10 * viol
                return 1.0 - _vertex_scores(pts)
            
            try:
                res = differential_evolution(
                    obj_local, bounds, seed=seed, maxiter=80,
                    popsize=25, tol=1e-6, polish=True,
                    init='latinhypercube', mutation=(0.4, 1.0),
                    recombination=0.7
                )
                test_pts = _pts_from_params(res.x, M_try)
                if _convex_violation(test_pts) == 0:
                    s = _vertex_scores(test_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = test_pts.copy()
            except Exception:
                pass
        
        # Local polish on best found using position-based optimization
        if best_score > 0:
            refined_pts, refined_score = _local_refine(best_pts, max_iter=3000)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
    
    # Final refinement with more iterations
    if best_score > 0:
        refined_pts, refined_score = _local_refine(best_pts, max_iter=8000)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Final fallback
    if _convex_violation(best_pts) != 0:
        best_pts = reg_pts
    
    return best_pts
