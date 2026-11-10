import numpy as np
from scipy.optimize import differential_evolution, minimize

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
    for M_try in [5, 7, 9]:
        for seed in [0, 1, 42]:
            np.random.seed(seed)
            # Smaller bounds for higher M to maintain convexity
            bound = 0.15 / np.sqrt(M_try)
            bounds = [(-bound, bound)] * (2 * M_try)
            
            def obj_local(params):
                pts = _pts_from_params(params, M_try)
                viol = _convex_violation(pts)
                if viol > 0:
                    return 1.0 + 10 * viol
                return 1.0 - _vertex_scores(pts)
            
            try:
                res = differential_evolution(
                    obj_local, bounds, seed=seed, maxiter=60,
                    popsize=20, tol=1e-5, polish=True,
                    init='latinhypercube', mutation=(0.3, 1.0),
                    recombination=0.8
                )
                test_pts = _pts_from_params(res.x, M_try)
                if _convex_violation(test_pts) == 0:
                    s = _vertex_scores(test_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = test_pts.copy()
            except Exception:
                pass
        
        # Local polish on best found
        if best_score > 0:
            def obj_polish(params):
                pts = _pts_from_params(params, M)
                viol = _convex_violation(pts)
                if viol > 0:
                    return 1.0 + 10 * viol
                return 1.0 - _vertex_scores(pts)
            
            # Find params that give current best_pts by trying small random inits
            for init_seed in range(5):
                np.random.seed(init_seed)
                bound = 0.15 / np.sqrt(M)
                init_params = np.random.uniform(-bound, bound, 2*M)
                try:
                    lm = minimize(obj_polish, init_params, method='Nelder-Mead',
                                  options={'maxiter':3000, 'xatol':1e-10, 'fatol':1e-10})
                    if _convex_violation(_pts_from_params(lm.x, M)) == 0:
                        s2 = _vertex_scores(_pts_from_params(lm.x, M))
                        if s2 > best_score:
                            best_score = s2
                            best_pts = _pts_from_params(lm.x, M).copy()
                except Exception:
                    pass
    
    # Final fallback
    if _convex_violation(best_pts) != 0:
        best_pts = reg_pts
    
    return best_pts
