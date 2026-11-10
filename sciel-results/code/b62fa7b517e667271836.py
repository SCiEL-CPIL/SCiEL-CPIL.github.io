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
    if np.all(cr > 0) or np.all(cr < 0):
        return 0.0
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
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    reg_score = _vertex_scores(reg_pts)
    best_pts = reg_pts.copy()
    best_score = reg_score
    
    M = 7
    bound = 0.12 / np.sqrt(M)
    bounds = [(-bound, bound)] * (2 * M)
    
    def obj_de(params):
        pts = _pts_from_params(params, M)
        viol = _convex_violation(pts)
        if viol > 0:
            return 1.0 + 10 * viol
        return 1.0 - _vertex_scores(pts)
    
    seeds = [0, 1, 7, 13, 42, 99, 123, 256, 777, 1024]
    for seed in seeds:
        np.random.seed(seed)
        try:
            res = differential_evolution(
                obj_de, bounds, seed=seed, maxiter=80,
                popsize=25, tol=1e-6, polish=True,
                init='latinhypercube', mutation=(0.4, 1.2),
                recombination=0.85
            )
            test_pts = _pts_from_params(res.x, M)
            if _convex_violation(test_pts) == 0:
                s = _vertex_scores(test_pts)
                if s > best_score:
                    best_score = s
                    best_pts = test_pts.copy()
        except Exception:
            pass
    
    # Local polish with Nelder-Mead from multiple starting points
    def obj_polish(params):
        pts = _pts_from_params(params, M)
        viol = _convex_violation(pts)
        if viol > 0:
            return 1.0 + 10 * viol
        return 1.0 - _vertex_scores(pts)
    
    for init_seed in range(10):
        np.random.seed(init_seed)
        init_params = np.random.uniform(-bound*0.8, bound*0.8, 2*M)
        try:
            lm = minimize(obj_polish, init_params, method='Nelder-Mead',
                          options={'maxiter':5000, 'xatol':1e-11, 'fatol':1e-11,
                                   'adaptive': True})
            if _convex_violation(_pts_from_params(lm.x, M)) == 0:
                s2 = _vertex_scores(_pts_from_params(lm.x, M))
                if s2 > best_score:
                    best_score = s2
                    best_pts = _pts_from_params(lm.x, M).copy()
        except Exception:
            pass
    
    # Try higher harmonic content for finer adjustments
    M2 = 9
    bound2 = 0.08 / np.sqrt(M2)
    bounds2 = [(-bound2, bound2)] * (2 * M2)
    
    def obj_de2(params):
        pts = _pts_from_params(params, M2)
        viol = _convex_violation(pts)
        if viol > 0:
            return 1.0 + 10 * viol
        return 1.0 - _vertex_scores(pts)
    
    for seed in [0, 42, 777]:
        np.random.seed(seed)
        try:
            res = differential_evolution(
                obj_de2, bounds2, seed=seed, maxiter=60,
                popsize=20, tol=1e-6, polish=True,
                init='latinhypercube', mutation=(0.3, 1.0),
                recombination=0.8
            )
            test_pts = _pts_from_params(res.x, M2)
            if _convex_violation(test_pts) == 0:
                s = _vertex_scores(test_pts)
                if s > best_score:
                    best_score = s
                    best_pts = test_pts.copy()
        except Exception:
            pass
    
    if _convex_violation(best_pts) != 0:
        best_pts = reg_pts
    
    return best_pts
