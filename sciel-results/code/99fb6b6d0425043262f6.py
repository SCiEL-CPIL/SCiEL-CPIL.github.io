"""Stella octangula (two interlocking tetrahedra) + 2 extra points, optimized via DE + polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_dists_sq(P):
    """Return matrix of squared pairwise distances (n, n)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sum(diff * diff, axis=2)

def _score_from_d2(D2):
    """Score = (max/min)^2 from squared-distance matrix."""
    mask = ~np.eye(N, dtype=bool)
    vals = D2[mask]
    if len(vals) == 0:
        return np.inf
    dmin = vals.min()
    dmax = vals.max()
    if dmin <= 1e-12:
        return np.inf
    r = dmax / dmin
    return r * r

def _make_points(params):
    """
    Build 14 points from two regular tetrahedra (stella octangula) + 2 extra points.
    params = [s1, s2, tx, ty, tz, rx, ry, rz, ex1, ey1, ez1, ex2, ey2, ez2]
    where:
      s1, s2: scale of tetra1, tetra2
      tx, ty, tz: translation of tetra2 relative to tetra1
      rx, ry, rz: rotation angles of tetra2 (Euler XYZ)
      ex1..ez2: positions of 2 extra points
    """
    # Regular tetrahedron vertices (unit scale, centered at origin)
    # Using the standard tetrahedron: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) / sqrt(3)
    tetra_base = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float) / np.sqrt(3.0)
    
    s1 = params[0]
    s2 = params[1]
    tx, ty, tz = params[2], params[3], params[4]
    rx, ry, rz = params[5], params[6], params[7]
    
    # Tetra 1: scaled by s1
    P1 = tetra_base * s1
    
    # Tetra 2: scaled by s2, rotated, translated
    # Rotation matrix (XYZ Euler angles)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    P2 = (tetra_base * s2) @ R.T + np.array([tx, ty, tz])
    
    # Extra points
    P3 = np.array([params[8:11], params[11:14]], dtype=float)
    
    return np.vstack([P1, P2, P3])

def _objective(params):
    P = _make_points(params)
    D2 = _pairwise_dists_sq(P)
    return _score_from_d2(D2)

def _smooth_objective(params, alpha=10.0):
    """Smooth surrogate using log-sum-exp for max/min."""
    P = _make_points(params)
    D2 = _pairwise_dists_sq(P)
    mask = ~np.eye(N, dtype=bool)
    vals = D2[mask]
    if np.any(vals <= 1e-12):
        return 1e10
    log_vals = np.log(vals)
    soft_max = np.exp(np.logaddexp.reduce(log_vals * alpha)) ** (1.0/alpha)
    soft_min = np.exp(-np.logaddexp.reduce(-log_vals * alpha)) ** (1.0/alpha)
    if soft_min <= 1e-12:
        return 1e10
    r = soft_max / soft_min
    return r * r

def _polish(P0, time_budget):
    """Local polish using L-BFGS-B on free coordinates."""
    deadline = time.time() + time_budget
    P = P0.copy()
    best_P = P.copy()
    best_score = _score_from_d2(_pairwise_dists_sq(P))
    
    def obj(x):
        P = x.reshape(N, D)
        D2 = _pairwise_dists_sq(P)
        mask = ~np.eye(N, dtype=bool)
        vals = D2[mask]
        if np.any(vals <= 1e-12):
            return 1e10
        dmin = vals.min()
        dmax = vals.max()
        if dmin <= 1e-12:
            return 1e10
        r = dmax / dmin
        return r * r
    
    x0 = P.flatten()
    res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    if res.fun < best_score:
        P = res.x.reshape(N, D)
        s = _score_from_d2(_pairwise_dists_sq(P))
        if s < best_score:
            best_score = s
            best_P = P.copy()
    
    # Second polish
    if time.time() < deadline:
        x0 = P.flatten()
        res2 = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        if res2.fun < best_score:
            P = res2.x.reshape(N, D)
            s = _score_from_d2(_pairwise_dists_sq(P))
            if s < best_score:
                best_score = s
                best_P = P.copy()
    
    return best_P, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    time_budget_total = DEADLINE - 5.0
    
    # Phase 1: Differential evolution over 14 parameters
    # Bounds for each parameter
    bounds = [
        (0.1, 3.0),   # s1
        (0.1, 3.0),   # s2
        (-3.0, 3.0),  # tx
        (-3.0, 3.0),  # ty
        (-3.0, 3.0),  # tz
        (-np.pi, np.pi),  # rx
        (-np.pi, np.pi),  # ry
        (-np.pi, np.pi),  # rz
        (-3.0, 3.0),  # ex1
        (-3.0, 3.0),  # ey1
        (-3.0, 3.0),  # ez1
        (-3.0, 3.0),  # ex2
        (-3.0, 3.0),  # ey2
        (-3.0, 3.0),  # ez2
    ]
    
    best_score = np.inf
    best_params = None
    
    # Use differential evolution with limited evaluations
    from scipy.optimize import differential_evolution
    
    # Try a few different initial populations or just run DE
    def de_objective(params):
        return _objective(params)
    
    # Run DE with maxiter limited
    maxiter = 50
    popsize = 15
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < time_budget_total * 0.5:
        try:
            result = differential_evolution(
                de_objective, bounds, maxiter=maxiter, popsize=popsize,
                seed=42, tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=False  # We'll do our own polish
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # If DE didn't find anything good, try a grid search fallback
    if best_params is None or best_score > 10.0:
        # Grid search over key parameters
        s1_vals = [0.5, 1.0, 1.5, 2.0]
        s2_vals = [0.5, 1.0, 1.5, 2.0]
        t_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        r_vals = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for s1 in s1_vals:
            for s2 in s2_vals:
                for tx in t_vals:
                    for ty in t_vals:
                        for tz in t_vals:
                            for rx in r_vals:
                                for ry in r_vals:
                                    for rz in r_vals:
                                        # Place extra points at origin initially
                                        params = [s1, s2, tx, ty, tz, rx, ry, rz,
                                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                                        sc = _objective(params)
                                        if sc < best_score:
                                            best_score = sc
                                            best_params = np.array(params)
                                        if time.time() - start_time > time_budget_total * 0.6:
                                            break
                            if time.time() - start_time > time_budget_total * 0.6:
                                break
                        if time.time() - start_time > time_budget_total * 0.6:
                            break
                    if time.time() - start_time > time_budget_total * 0.6:
                        break
                if time.time() - start_time > time_budget_total * 0.6:
                    break
            if time.time() - start_time > time_budget_total * 0.6:
                break
        if time.time() - start_time > time_budget_total * 0.6:
            break
    
    if best_params is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(__import__('itertools').product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    P_best = _make_points(best_params)
    
    # Phase 2: Local search on the 14 parameters
    elapsed = time.time() - start_time
    if elapsed < time_budget_total * 0.8:
        # Use Nelder-Mead on the 14 parameters
        from scipy.optimize import minimize
        
        def param_obj(params):
            return _objective(params)
        
        res = minimize(param_obj, best_params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            P_best = _make_points(best_params)
    
    # Phase 3: Polish in full 42-dim space
    elapsed = time.time() - start_time
    polish_budget = min(20.0, time_budget_total - elapsed - 3.0)
    if polish_budget > 2.0:
        P_polished, sc_polished = _polish(P_best, polish_budget)
        if sc_polished < best_score:
            best_score = sc_polished
            P_best = P_polished
    
    # Verify distinctness
    D2 = _pairwise_dists_sq(P_best)
    mask = ~np.eye(N, dtype=bool)
    if D2[mask].min() < 1e-10:
        P_best = P_best + np.random.default_rng(42).normal(0, 1e-6, P_best.shape)
    
    return P_best
