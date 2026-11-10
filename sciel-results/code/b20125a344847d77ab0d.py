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

def _convexity_penalty(pts):
    n = len(pts)
    edges = np.roll(pts, -1, axis=0) - pts
    # Cross products of consecutive edges
    # v1 x v2 = v1[0]*v2[1] - v1[1]*v2[0]
    cr = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    # For a strictly convex CCW polygon, all cr should be > 0
    # We penalize negative or near-zero cross products
    # Using sum of squares of negative parts for smoothness
    neg_cr = np.minimum(cr, 0)
    return np.sum(neg_cr**2) + np.sum(np.maximum(0, 1e-5 - np.abs(cr))**2) # Penalize near-zero too? No, just negative is enough for validity, but we want strict.
    # Actually, the task says "every turn ... nonzero". So |cr| must be > 0.
    # Let's just penalize cr <= 0.
    # A better penalty: sum of (min(cr, 0))^2
    return np.sum(np.minimum(cr, 0)**2)

def _pts_from_params(params):
    n = 20
    M = 2 # Reduced M for stability and faster convergence
    theta = 2*np.pi*np.arange(n)/n
    r = np.ones(n)
    idx = 0
    for m in range(1, M+1):
        a = params[idx]; b = params[idx+1]; idx += 2
        r = r + a*np.cos(m*theta) + b*np.sin(m*theta)
    # Ensure r is positive to avoid self-intersection near origin
    # If r becomes <= 0, points are invalid. We can clip or penalize.
    # For now, let the optimizer handle it via convexity penalty (which will fail if r is small/negative causing cr issues)
    x = r*np.cos(theta); y = r*np.sin(theta)
    return np.stack([x,y],axis=1)

def _objective(params):
    pts = _pts_from_params(params)
    # Check for repeated points
    # If any distance is too small, it's likely invalid or scores 0
    min_dist = np.inf
    for i in range(20):
        for j in range(i+1,20):
            d = np.linalg.norm(pts[i]-pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-5:
        return 10.0 + 1.0/min_dist

    pen = _convexity_penalty(pts)
    if pen > 1e-8:
        return 1.0 + 10.0 * pen
    
    score = _vertex_scores(pts)
    return 1.0 - score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    M = 2
    np.random.seed(42)
    # Bounds: small perturbations
    bounds = [(-0.15, 0.15)]*(2*M)
    
    res = differential_evolution(_objective, bounds, seed=42, maxiter=100,
                                  popsize=30, tol=1e-8, polish=False,
                                  init='latinhypercube', mutation=(0.5, 1.0),
                                  recombination=0.8)
    
    best_params = res.x
    best_pts = _pts_from_params(best_params)
    best_score = _vertex_scores(best_pts) if _convexity_penalty(best_pts) < 1e-8 else 0.0

    # Local polish
    def obj_local(params):
        pts = _pts_from_params(params)
        pen = _convexity_penalty(pts)
        if pen > 1e-8:
            return 1.0 + 10*pen
        return 1.0 - _vertex_scores(pts)
        
    try:
        lm = minimize(obj_local, best_params, method='Nelder-Mead',
                      options={'maxiter':5000,'xatol':1e-10,'fatol':1e-10})
        if _convexity_penalty(_pts_from_params(lm.x)) < 1e-8:
            s2 = _vertex_scores(_pts_from_params(lm.x))
            if s2 > best_score:
                best_pts = _pts_from_params(lm.x)
                best_score = s2
    except Exception:
        pass

    # Final check and fallback
    if _convexity_penalty(best_pts) >= 1e-8:
        angles = np.linspace(0,2*np.pi,n,endpoint=False)
        best_pts = np.stack([np.cos(angles),np.sin(angles)],axis=1)
        
    return best_pts
