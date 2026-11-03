"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius(pts):
    """Compute the feasible radius for a set of points in [-0.5,0.5]^2."""
    if len(pts) < 2:
        return np.inf
    # Pairwise distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    # Distance to walls
    wall = np.minimum(0.5 - np.abs(pts), axis=1).min()
    return min(min_pair, wall)


def _soft_min_radius(pts, beta=50.0):
    """Differentiable soft-min approximation of the radius objective."""
    # Pairwise
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e6)
    # Soft min over pairs: use log-sum-exp trick for stability
    # We want min_{i<j} d_ij / 2
    # Use: -log(sum exp(-beta * d_ij)) / beta + log(K)/beta  approximates min
    # Actually simpler: use a smooth minimum
    # For pairs only
    triu_idx = np.triu_indices(N, k=1)
    pair_d = d[triu_idx] / 2.0
    # Soft min: m - log(sum(exp(-beta*(x-m))))/beta where m is current min approx
    # Simpler: use the fact that min(x) ≈ -1/beta * log(sum(exp(-beta*x)))
    # But this can be numerically unstable. Let's use a different approach.
    # Use: min(x) ≈ (sum x^p)^(1/p) for large p, but that's not differentiable at 0.
    # Let's just use the direct soft-min:
    m = pair_d.min()
    soft_pair = m - np.log(np.sum(np.exp(-beta * (pair_d - m)))) / beta
    
    # Wall distances
    wall_d = np.minimum(0.5 - np.abs(pts), axis=1)
    m_w = wall_d.min()
    soft_wall = m_w - np.log(np.sum(np.exp(-beta * (wall_d - m_w)))) / beta
    
    # Overall soft min
    m_all = min(soft_pair, soft_wall)
    soft_all = m_all - np.log(np.exp(-beta * (soft_pair - m_all)) + np.exp(-beta * (soft_wall - m_all))) / beta
    
    return soft_all


def _soft_min_radius_neg(pts, beta=50.0):
    """Negative of soft-min radius (for minimization)."""
    return -_soft_min_radius(pts, beta)


def _soft_min_radius_neg_grad(pts, beta=50.0):
    """Gradient of negative soft-min radius."""
    # This is complex; let's use finite differences or just let scipy estimate.
    # Actually, let's implement the gradient properly.
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e6)
    
    triu_idx = np.triu_indices(N, k=1)
    pair_d = d[triu_idx] / 2.0
    
    # Soft min weights
    m = pair_d.min()
    exp_vals = np.exp(-beta * (pair_d - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_pair w.r.t. pts
    # d(soft_pair)/d(pts_i) = sum_j w_ij * d(pair_d_ij)/d(pts_i)
    # where w_ij are the weights
    grad = np.zeros_like(pts)
    for idx in range(len(triu_idx[0])):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        w = weights[idx]
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            grad[i] += w * diff / dist
            grad[j] -= w * diff / dist
    
    # Wall distances
    wall_d = np.minimum(0.5 - np.abs(pts), axis=1)
    m_w = wall_d.min()
    exp_vals_w = np.exp(-beta * (wall_d - m_w))
    weights_w = exp_vals_w / np.sum(exp_vals_w)
    
    for i in range(N):
        w = weights_w[i]
        # Gradient of wall distance w.r.t. pts[i]
        # wall_d_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # d(0.5 - |x|)/dx = -sign(x)
        # d(0.5 - |y|)/dy = -sign(y)
        gx = -np.sign(pts[i, 0])
        gy = -np.sign(pts[i, 1])
        # But we need the min, so only the active constraint matters
        # Actually, the soft min already handles this
        # The gradient of min(a,b) is a weighted combination
        # Let me reconsider: wall_d_i = min(0.5-|x_i|, 0.5-|y_i|)
        # This is a min of two values, so we need to track which is active
        # For simplicity, let's just use the gradient of the individual terms
        # weighted by which is smaller
        dx = 0.5 - np.abs(pts[i, 0])
        dy = 0.5 - np.abs(pts[i, 1])
        if dx <= dy:
            grad[i, 0] += w * (-np.sign(pts[i, 0]))
        else:
            grad[i, 1] += w * (-np.sign(pts[i, 1]))
    
    # Combine: soft_all = soft_min(soft_pair, soft_wall)
    # This is getting complex. Let's just return the gradient of soft_pair and soft_wall separately
    # and let the optimizer handle it. Actually, let's simplify and just use the gradient
    # of the sum (which approximates the min when beta is large).
    
    # For now, return just the pair gradient as an approximation
    return -grad  # Negative because we're minimizing -radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    best_pts = pts.copy()
    best_r = _radius(best_pts)
    
    # Try SLSQP optimization with multiple random restarts
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Perturb the initial solution
        if restart == 0:
            pts = best_pts.copy()
        else:
            pts = best_pts + np.random.randn(N, 2) * 0.02
            # Clamp to valid region
            pts = np.clip(pts, -0.49, 0.49)
        
        # Optimize using SLSQP
        try:
            result = minimize(
                _soft_min_radius_neg,
                pts.flatten(),
                args=(50.0,),
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < -best_r:
                new_pts = result.x.reshape(N, 2)
                # Clamp to ensure validity
                new_pts = np.clip(new_pts, -0.5 + 1e-10, 0.5 - 1e-10)
                new_r = _radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # Final micro-optimization with higher beta
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _soft_min_radius_neg,
                best_pts.flatten(),
                args=(100.0,),
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            if result.fun < -best_r:
                new_pts = result.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5 + 1e-10, 0.5 - 1e-10)
                new_r = _radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
