"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import cos, sin, pi, log, exp
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the score: product of ordered-pair distances / 8^8, after scaling to diameter 2."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    diam = np.max(dists)
    if diam < 1e-12:
        return 0.0
    scale = 2.0 / diam
    log_prod = n * (n - 1) * log(scale) + np.sum(np.log(dists))
    score = exp(log_prod) / (n ** n)
    return score

def _neg_log_score(pts_flat):
    """Negative log-product of distances (to minimize), with scaling."""
    n = 8
    pts = pts_flat.reshape(n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
    scale = 2.0 / diam
    log_prod = n * (n - 1) * log(scale) + np.sum(np.log(dists))
    return -log_prod

def _two_squares_model(params):
    """Two concentric squares: outer square radius r1, inner square radius r2,
    with inner square rotated by angle theta relative to outer.
    params = [r1, r2, theta]
    Returns 8 points as (8,2) array.
    """
    r1, r2, theta = params
    pts = np.zeros((8, 2))
    for k in range(4):
        angle = 2 * pi * k / 4 + pi / 4  # outer square at 45, 135, 225, 315
        pts[2 * k, 0] = r1 * cos(angle)
        pts[2 * k, 1] = r1 * sin(angle)
        angle_in = 2 * pi * k / 4 + theta  # inner square rotated by theta
        pts[2 * k + 1, 0] = r2 * cos(angle_in)
        pts[2 * k + 1, 1] = r2 * sin(angle_in)
    return pts

def _two_squares_neglog(params):
    """Negative log-score for two-squares model."""
    pts = _two_squares_model(params)
    return _neg_log_score(pts.flatten())

def _five_param_model(params):
    """Five-parameter model: two concentric squares with independent rotations.
    params = [r1, r2, theta1, theta2, offset_x]
    Outer square: radius r1, rotated by theta1
    Inner square: radius r2, rotated by theta2
    Both shifted by offset_x along x-axis (for asymmetry)
    """
    r1, r2, theta1, theta2, ox = params
    pts = np.zeros((8, 2))
    for k in range(4):
        angle = 2 * pi * k / 4 + theta1
        pts[2 * k, 0] = r1 * cos(angle) + ox
        pts[2 * k, 1] = r1 * sin(angle)
        angle_in = 2 * pi * k / 4 + theta2
        pts[2 * k + 1, 0] = r2 * cos(angle_in) + ox
        pts[2 * k + 1, 1] = r2 * sin(angle_in)
    return pts

def _five_param_neglog(params):
    """Negative log-score for five-parameter model."""
    pts = _five_param_model(params)
    return _neg_log_score(pts.flatten())

def _optimize_from_start(start_pts, deadline):
    """Multi-stage optimization from a starting point. Returns best (score, pts)."""
    n = 8
    best_score = _score(start_pts)
    best_pts = start_pts.copy()
    
    # Stage 1: L-BFGS-B
    try:
        if time.time() > deadline - 15.0:
            return best_score, best_pts
        result = minimize(
            _neg_log_score,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        opt_pts = result.x.reshape(n, 2)
        opt_score = _score(opt_pts)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
    except Exception:
        pass
    
    # Stage 2: Powell method for refinement
    try:
        if time.time() > deadline - 10.0:
            return best_score, best_pts
        result = minimize(
            _neg_log_score,
            best_pts.flatten(),
            method='Powell',
            options={'maxiter': 300, 'ftol': 1e-14, 'xtol': 1e-14}
        )
        opt_pts = result.x.reshape(n, 2)
        opt_score = _score(opt_pts)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
    except Exception:
        pass
    
    # Stage 3: Nelder-Mead for final polish
    try:
        if time.time() > deadline - 5.0:
            return best_score, best_pts
        result = minimize(
            _neg_log_score,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'fatol': 1e-15, 'xatol': 1e-15, 'adaptive': True}
        )
        opt_pts = result.x.reshape(n, 2)
        opt_score = _score(opt_pts)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
    except Exception:
        pass
    
    return best_score, best_pts

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Start from regular octagon
    regular = np.array([[cos(2 * pi * k / n), sin(2 * pi * k / n)] for k in range(n)])
    best_pts = regular.copy()
    best_score = _score(regular)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Optimize the two-squares parameterized model
    try:
        if time.time() < deadline - 30.0:
            # Try several initial conditions for two-squares model
            for r1_init in [1.0, 1.2, 0.8, 1.5]:
                for r2_init in [0.5, 0.7, 0.9, 1.0, 0.3]:
                    for theta_init in [0.0, pi/4, pi/8, -pi/8]:
                        if time.time() > deadline - 20.0:
                            break
                        params0 = np.array([r1_init, r2_init, theta_init])
                        try:
                            res = minimize(
                                _two_squares_neglog,
                                params0,
                                method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-14}
                            )
                            opt_pts = _two_squares_model(res.x)
                            opt_score = _score(opt_pts)
                            if opt_score > best_score:
                                best_score = opt_score
                                best_pts = opt_pts.copy()
                        except Exception:
                            continue
    except Exception:
        pass
    
    # Strategy 2: Optimize the five-parameter model
    try:
        if time.time() < deadline - 20.0:
            for r1_init in [1.0, 1.2, 0.8]:
                for r2_init in [0.5, 0.7, 0.9]:
                    for t1_init in [pi/4, 0.0]:
                        for t2_init in [0.0, pi/4, pi/8]:
                            if time.time() > deadline - 15.0:
                                break
                            params0 = np.array([r1_init, r2_init, t1_init, t2_init, 0.0])
                            try:
                                res = minimize(
                                    _five_param_neglog,
                                    params0,
                                    method='L-BFGS-B',
                                    options={'maxiter': 200, 'ftol': 1e-14}
                                )
                                opt_pts = _five_param_model(res.x)
                                opt_score = _score(opt_pts)
                                if opt_score > best_score:
                                    best_score = opt_score
                                    best_pts = opt_pts.copy()
                            except Exception:
                                continue
    except Exception:
        pass
    
    # Strategy 3: Full 16D optimization from multiple starting points
    starting_points = [
        best_pts.copy(),
        regular.copy(),
        regular + rng.normal(0, 0.1, size=(n, 2)),
        regular + rng.normal(0, 0.2, size=(n, 2)),
        regular + rng.normal(0, 0.3, size=(n, 2)),
        regular + rng.normal(0, 0.5, size=(n, 2)),
        # Slightly asymmetric starts
        np.array([[cos(2*pi*k/8 + 0.1*sin(k)), sin(2*pi*k/8 + 0.1*cos(k))] for k in range(n)]),
        np.array([[cos(2*pi*k/8 + 0.2*sin(2*k)), sin(2*pi*k/8 + 0.2*cos(2*k))] for k in range(n)]),
    ]
    
    for start in starting_points:
        if time.time() > deadline - 10.0:
            break
        s, p = _optimize_from_start(start, deadline)
        if s > best_score:
            best_score = s
            best_pts = p.copy()
    
    # Strategy 4: Basin hopping from best found
    try:
        if time.time() < deadline - 8.0:
            from scipy.optimize import basinhopping
            def _wrapper(x):
                return _neg_log_score(x)
            
            bh_result = basinhopping(
                _wrapper,
                best_pts.flatten(),
                niter=10,
                T=0.1,
                stepsize=0.1,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'ftol': 1e-14}},
                accept_test=lambda new, old: new < old
            )
            opt_pts = bh_result.x.reshape(n, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    # Final high-precision polish
    try:
        if time.time() < deadline - 2.0:
            result = minimize(
                _neg_log_score,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            opt_pts = result.x.reshape(n, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    except Exception:
        pass
    
    # Return as list of [x, y]
    return [[float(p[0]), float(p[1])] for p in best_pts]
# EVOLVE-BLOCK-END
