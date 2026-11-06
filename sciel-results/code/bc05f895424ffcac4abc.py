"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from math import cos, sin, pi, log

def _log_score(points):
    """Compute log(Delta/n^n) for 8 points.
    
    Delta is product of ordered pair distances after scaling to diameter 2.
    Score = Delta / n^n.
    We return log(Score) = log(Delta) - n*log(n).
    """
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(sq_dists, 1e-30))
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return -1e30
    
    # Scale to diameter 2
    scale = 2.0 / diam
    scaled_dists = dists * scale
    
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # log_delta = sum of log(scaled_dist) over all ordered pairs i != j
    # Note: np.sum over the masked array includes both (i,j) and (j,i)
    log_delta = np.sum(np.log(scaled_dists[mask]))
    
    # log_score = log_delta - n * log(n)
    log_score = log_delta - n * np.log(n)
    
    return log_score

def _score(points):
    """Compute the actual score value."""
    return np.exp(_log_score(points))

def _neg_log_score(points):
    """Negative log score for minimization."""
    return -_log_score(points)

def _generate_ellipse_points(theta, eccentricity):
    """
    Generate 8 points on an ellipse.
    The points are at regular angular intervals (regular octagon angles)
    but mapped to an ellipse with semi-major axis 1 and semi-minor axis (1-eccentricity).
    Or more generally, we can modulate the radius.
    
    Let's use a more flexible model:
    Points are at angles 2*pi*k/8.
    Radius r_k = 1 + A * cos(2 * (2*pi*k/8 - phi))
    This allows for a "squashed" octagon.
    
    Parameters:
    theta: phase shift of the modulation (0 to pi/2 due to symmetry)
    A: amplitude of radial modulation (0 to 0.5)
    """
    k = np.arange(8)
    angles = 2 * pi * k / 8
    # Modulate radius
    # Using cos(2*angle) creates a 4-fold symmetry (squash in x vs y)
    # Using cos(4*angle) creates an 8-fold symmetry (petal shape)
    # Let's try a mix or just the 4-fold which is most likely to break the circle symmetry
    # Actually, the regular octagon has 8-fold symmetry. Breaking it to 4-fold (ellipse) is a strong candidate.
    r = 1.0 + A * np.cos(2 * (angles - theta))
    
    # Ensure radii are positive
    r = np.maximum(r, 1e-6)
    
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _neg_log_score_param(params):
    """
    Objective function for parameter optimization.
    params: [theta, A]
    """
    theta, A = params
    pts = _generate_ellipse_points(theta, A)
    return _neg_log_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # 1. Regular Octagon (Baseline)
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    octagon = np.array(octagon)
    score_oct = _score(octagon)
    if score_oct > best_score:
        best_score = score_oct
        best_points = octagon
    
    # 2. Search the "Elliptical Octagon" family
    # Parameters: theta (0 to pi/2), A (0 to 0.4)
    # We use a grid search first to find a good region, then refine.
    
    def _is_valid(candidate):
        """Check if candidate points are distinct and finite."""
        if not np.all(np.isfinite(candidate)):
            return False
        diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists) > 1e-8

    # Grid search for theta and A
    theta_grid = np.linspace(0, pi/2, 10)
    A_grid = np.linspace(0, 0.4, 15)
    
    best_param = [0, 0]
    best_param_score = -np.inf
    
    for t in theta_grid:
        for a in A_grid:
            if time.time() > deadline - 10:
                break
            pts = _generate_ellipse_points(t, a)
            if _is_valid(pts):
                s = _score(pts)
                if s > best_param_score:
                    best_param_score = s
                    best_param = [t, a]
                    if s > best_score:
                        best_score = s
                        best_points = pts
    
    # Refine using L-BFGS-B on the parameter space
    if best_param_score > -np.inf and time.time() < deadline - 5:
        try:
            res = minimize(_neg_log_score_param, best_param, method='L-BFGS-B',
                           bounds=[(0, pi/2), (0, 0.5)],
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                t_opt, a_opt = res.x
                pts_opt = _generate_ellipse_points(t_opt, a_opt)
                if _is_valid(pts_opt):
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt
        except Exception:
            pass

    # 3. Try a different family: "Pinched" squares (two concentric squares rotated by pi/4)
    # This is a known competitor.
    def _generate_pinched(theta, r2):
        angles1 = np.array([0, pi/2, pi, 3*pi/2])
        angles2 = angles1 + theta
        pts1 = np.column_stack([cos(angles1), sin(angles1)])
        pts2 = r2 * np.column_stack([cos(angles2), sin(angles2)])
        return np.vstack([pts1, pts2])

    def _neg_log_score_pinched(params):
        theta, r2 = params
        if r2 < 0.1: return 1e10
        pts = _generate_pinched(theta, r2)
        return _neg_log_score(pts)

    # Grid search for pinched
    theta_grid2 = np.linspace(0, pi/2, 5)
    r2_grid = np.linspace(0.2, 1.5, 15)
    best_pinched_score = -np.inf
    best_pinched_param = [0, 1]
    
    for t in theta_grid2:
        for r in r2_grid:
            if time.time() > deadline - 5:
                break
            pts = _generate_pinched(t, r)
            if _is_valid(pts):
                s = _score(pts)
                if s > best_pinched_score:
                    best_pinched_score = s
                    best_pinched_param = [t, r]
                    if s > best_score:
                        best_score = s
                        best_points = pts
    
    if best_pinched_score > -np.inf and time.time() < deadline - 5:
        try:
            res = minimize(_neg_log_score_pinched, best_pinched_param, method='L-BFGS-B',
                           bounds=[(0, pi/2), (0.1, 2.0)],
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                t_opt, r_opt = res.x
                pts_opt = _generate_pinched(t_opt, r_opt)
                if _is_valid(pts_opt):
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt
        except Exception:
            pass

    # 4. Final Polish: Full 16D optimization from the best point found
    if best_points is not None and time.time() < deadline - 10:
        try:
            x0 = best_points.flatten()
            
            def _full_neg_log_score(x):
                pts = x.reshape(8, 2)
                return _neg_log_score(pts)
            
            # Use Powell method which doesn't need gradients and is robust
            res = minimize(_full_neg_log_score, x0, method='Powell',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if res.success:
                pts_opt = res.x.reshape(8, 2)
                if _is_valid(pts_opt):
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt
        except Exception:
            pass

    # 5. One more attempt: Nelder-Mead on the best point for local refinement
    if best_points is not None and time.time() < deadline - 5:
        try:
            x0 = best_points.flatten()
            
            def _full_neg_log_score_nm(x):
                pts = x.reshape(8, 2)
                return _neg_log_score(pts)
            
            res = minimize(_full_neg_log_score_nm, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
            
            if res.success:
                pts_opt = res.x.reshape(8, 2)
                if _is_valid(pts_opt):
                    s_opt = _score(pts_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_points = pts_opt
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular octagon
        return [[float(p[0]), float(p[1])] for p in octagon]
    
    return [[float(p[0]), float(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
