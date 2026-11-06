"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def _log_score_and_grad(points):
    """Compute log(Delta/n^n) and its gradient analytically.
    
    The score is Delta/n^n where Delta is the product of all ordered pair distances
    after scaling to diameter 2. We work in log space for numerical stability.
    
    Returns (log_score, gradient) where gradient is d(log_score)/d(points).
    """
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(sq_dists, 1e-30))
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return -1e30, np.zeros_like(pts)
    
    # Scale to diameter 2: scaled_dists = dists * 2 / diam
    scale = 2.0 / diam
    scaled_dists = dists * scale
    
    # Mask out diagonal (self-distances)
    mask = ~np.eye(n, dtype=bool)
    
    # log_delta = sum over ordered pairs of log(scaled_dist)
    # = sum_{i != j} log(scaled_dists[i,j])
    log_delta = np.sum(np.log(scaled_dists[mask]))
    
    # log_score = log_delta - n * log(n)
    log_score = log_delta - n * np.log(n)
    
    # Now compute gradient of log_score w.r.t. pts
    # log_score = sum_{i!=j} log(|p_i - p_j| * 2/diam) - n*log(n)
    #           = sum_{i!=j} [log|p_i - p_j| + log(2) - log(diam)] - n*log(n)
    #           = sum_{i!=j} log|p_i - p_j| - n*(n-1)*log(diam) + n*(n-1)*log(2) - n*log(n)
    
    # Let S = sum_{i!=j} log|p_i - p_j|, D = diam = max_{i!=j} |p_i - p_j|
    # log_score = S - n*(n-1)*log(D) + const
    
    # dS/dp_k = sum_{j!=k} (p_k - p_j)/|p_k - p_j|^2  [for each coordinate]
    # d(log D)/dp_k: D is max of |p_i - p_j|, so only the pair(s) achieving the max contribute
    # If unique max pair (a,b): d(log D)/dp_k = delta_{k,a}*(p_a-p_b)/D^2 + delta_{k,b}*(p_b-p_a)/D^2
    
    # Compute dS/dp
    dS = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-15:
                continue
            # d/dp_i of log|p_i - p_j| = (p_i - p_j) / |p_i - p_j|^2
            dS[i] += diff[i, j] / (d * d)
    
    # Find which pair(s) achieve the diameter
    max_val = diam
    # Find all pairs with distance close to max
    eps = 1e-10
    max_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(dists[i, j] - max_val) < eps:
                max_pairs.append((i, j))
    
    # d(log D)/dp_k for the max pair(s)
    dlogD = np.zeros_like(pts)
    for (a, b) in max_pairs:
        if len(max_pairs) > 1:
            # Multiple max pairs - use subgradient (average)
            weight = 1.0 / len(max_pairs)
        else:
            weight = 1.0
        d_ab = diff[a, b]  # p_a - p_b
        D_sq = max_val * max_val
        dlogD[a] += weight * d_ab / D_sq
        dlogD[b] += weight * (-d_ab) / D_sq
    
    # Gradient of log_score = dS/dp - n*(n-1) * dlogD/dp
    n_pairs = n * (n - 1)
    grad = dS - n_pairs * dlogD
    
    return log_score, grad

def _score(points):
    """Compute Delta / n^n for 8 points."""
    log_s, _ = _log_score_and_grad(points)
    return np.exp(log_s)

def _neg_log_score_and_grad(points):
    """Negative log-score and its gradient for minimization."""
    pts = np.array(points, dtype=float).flatten()
    pts_2d = pts.reshape(8, 2)
    log_s, grad = _log_score_and_grad(pts_2d)
    return -log_s, grad.flatten()

def _neg_log_score(points):
    """Negative log-score."""
    pts = np.array(points, dtype=float)
    log_s, _ = _log_score_and_grad(pts)
    return -log_s

def _generate_pinched_points(theta, r2):
    """
    Generate 8 points forming two concentric squares.
    Square 1: vertices at angles 0, pi/2, pi, 3pi/2 with radius 1.
    Square 2: vertices at angles theta + 0, theta + pi/2, ... with radius r2.
    This configuration is known to potentially beat the regular octagon.
    """
    angles1 = np.array([0, pi/2, pi, 3*pi/2])
    angles2 = angles1 + theta
    
    pts1 = np.column_stack([cos(angles1), sin(angles1)])
    pts2 = r2 * np.column_stack([cos(angles2), sin(angles2)])
    
    return np.vstack([pts1, pts2])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with regular octagon (theta=pi/4, r2=1)
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    best_points = np.array(octagon, dtype=float)
    best_log_score = _log_score_and_grad(best_points)[0]
    
    rng = np.random.RandomState(42)
    
    def _is_valid(candidate):
        """Check if candidate points are distinct."""
        diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists) > 1e-8
    
    # Strategy: Search the "pinched" family explicitly.
    # The regular octagon is a specific case of this family (theta=pi/4, r2=1).
    # The "pinched" configuration (e.g., theta=0, r2 != 1 or theta=pi/8, r2 != 1)
    # often yields higher scores.
    
    # Define objective for the parameterized family
    def _neg_log_score_param(params):
        theta = params[0]
        r2 = params[1]
        # Ensure r2 is positive
        if r2 < 1e-6:
            return 1e10, np.zeros(2)
        
        pts = _generate_pinched_points(theta, r2)
        log_s, _ = _log_score_and_grad(pts)
        return -log_s, np.zeros(2) # Gradient w.r.t params is complex, use finite diff or L-BFGS-B without jac
    
    def _neg_log_score_param_jac(params):
        theta = params[0]
        r2 = params[1]
        if r2 < 1e-6:
            return 1e10, np.zeros(2)
        
        pts = _generate_pinched_points(theta, r2)
        log_s, grad_pts = _log_score_and_grad(pts)
        
        # Chain rule: d(log_s)/d(theta) = sum_{i in sq2} grad_pts[i] . d(pts[i])/d(theta)
        # d(pts[i])/d(theta) = r2 * [-sin(angle_i), cos(angle_i)]
        # d(log_s)/d(r2) = sum_{i in sq2} grad_pts[i] . d(pts[i])/d(r2)
        # d(pts[i])/d(r2) = [cos(angle_i), sin(angle_i)]
        
        angles2 = np.array([0, pi/2, pi, 3*pi/2]) + theta
        
        dtheta = 0.0
        dr2 = 0.0
        for i in range(4, 8):
            idx = i - 4
            a = angles2[idx]
            # d(pts_i)/d(theta)
            dpt_dtheta = r2 * np.array([-sin(a), cos(a)])
            dtheta += np.dot(grad_pts[i], dpt_dtheta)
            
            # d(pts_i)/d(r2)
            dpt_dr2 = np.array([cos(a), sin(a)])
            dr2 += np.dot(grad_pts[i], dpt_dr2)
            
        return -log_s, np.array([-dtheta, -dr2])

    # 1. Optimize the pinched family
    # Try several starting points in (theta, r2) space
    starts_param = []
    # Regular octagon
    starts_param.append([pi/4, 1.0])
    # Pinched square (theta=0)
    starts_param.append([0.0, 0.5])
    starts_param.append([0.0, 0.8])
    starts_param.append([0.0, 1.2])
    starts_param.append([0.0, 1.5])
    # Intermediate
    starts_param.append([pi/8, 0.8])
    starts_param.append([pi/8, 1.2])
    starts_param.append([3*pi/8, 0.8])
    starts_param.append([3*pi/8, 1.2])
    
    # Random starts
    for _ in range(20):
        t = rng.uniform(0, pi/2)
        r = rng.uniform(0.2, 2.0)
        starts_param.append([t, r])

    for x0 in starts_param:
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(_neg_log_score_param, x0, jac=_neg_log_score_param_jac,
                            method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success and result.fun < -best_log_score:
                theta_opt, r2_opt = result.x
                # Ensure r2 is positive
                if r2_opt < 1e-6:
                    continue
                
                candidate = _generate_pinched_points(theta_opt, r2_opt)
                
                if _is_valid(candidate):
                    cand_log_score = _log_score_and_grad(candidate)[0]
                    if cand_log_score > best_log_score:
                        best_log_score = cand_log_score
                        best_points = candidate
        except Exception:
            pass

    # 2. Refine the best point found in full 16D space
    # This allows breaking the symmetry if beneficial
    if time.time() < deadline - 10.0:
        try:
            x0_flat = best_points.flatten()
            result = minimize(_neg_log_score, x0_flat, jac=_neg_log_score_and_grad,
                            method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxfun': 10000})
            
            candidate = result.x.reshape(8, 2)
            if _is_valid(candidate):
                cand_log_score = _log_score_and_grad(candidate)[0]
                if cand_log_score > best_log_score:
                    best_log_score = cand_log_score
                    best_points = candidate
        except Exception:
            pass

    # 3. Final high-precision polish using Nelder-Mead on the best point
    if time.time() < deadline - 5.0:
        try:
            x0_flat = best_points.flatten()
            result = minimize(_neg_log_score, x0_flat,
                            method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-12,
                                    'fatol': 1e-14})
            
            candidate = result.x.reshape(8, 2)
            if _is_valid(candidate):
                cand_log_score = _log_score_and_grad(candidate)[0]
                if cand_log_score > best_log_score:
                    best_log_score = cand_log_score
                    best_points = candidate
        except Exception:
            pass

    # Return the best points found
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
