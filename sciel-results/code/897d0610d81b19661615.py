import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    If invalid (points outside), radius is -1.
    """
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    if np.any(bound_dists < -1e-9):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    
    np.fill_diagonal(dists_sq, np.inf)
    
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def soft_min_radius(x, tau=0.01):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    x: flat array of shape (2*N,)
    Returns negative soft-min radius (to minimize).
    """
    pos = x.reshape(N, 2)
    
    # Boundary distances
    max_abs = np.maximum(pos[:, 0], pos[:, 1])
    max_abs = np.maximum(np.abs(max_abs), np.abs(pos[:, 0]))
    max_abs = np.maximum(max_abs, np.abs(pos[:, 1]))
    bound_dists = 0.5 - max_abs
    
    # Pairwise distances (only upper triangle to save computation)
    # Use a vectorized approach
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Only consider upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq)
    
    # Soft-min of all constraints (boundary and pairwise/2)
    all_vals = np.concatenate([bound_dists, pair_dists / 2.0])
    
    # Log-sum-exp for soft-min: -soft_min = logsumexp(-x / tau) * tau
    # soft_min(x) ≈ -tau * log(sum(exp(-x_i / tau)))
    neg_vals = -all_vals / tau
    max_neg = np.max(neg_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
    soft_min_val = -tau * log_sum
    
    return -soft_min_val

def soft_min_radius_grad(x, tau=0.01):
    """
    Gradient of the soft-min radius objective.
    """
    pos = x.reshape(N, 2)
    
    # Boundary distances
    abs_pos = np.abs(pos)
    max_abs = np.max(abs_pos, axis=1)
    bound_dists = 0.5 - max_abs
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq)
    
    all_vals = np.concatenate([bound_dists, pair_dists / 2.0])
    
    # Weights for soft-min
    neg_vals = -all_vals / tau
    max_neg = np.max(neg_vals)
    exp_vals = np.exp(neg_vals - max_neg)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Boundary gradient
    # d(bound_dists[i])/d(pos[i,k]) = -sign(pos[i,k]) if |pos[i,k]| == max_abs
    for i in range(N):
        w = weights[i]
        if w < 1e-10:
            continue
        # Which coordinate is max?
        if abs_pos[i, 0] >= abs_pos[i, 1]:
            grad[2*i] = -w * np.sign(pos[i, 0]) if pos[i, 0] != 0 else 0
        else:
            grad[2*i+1] = -w * np.sign(pos[i, 1]) if pos[i, 1] != 0 else 0
    
    # Pairwise distance gradient
    # d(pair_dists[pair_j])/d(pos[i,k]) = (pos[i,k] - pos[j,k]) / dist
    n_pairs = len(pair_dists)
    for p in range(n_pairs):
        w = weights[N + p]
        if w < 1e-10:
            continue
        i, j = iu[0][p], iu[1][p]
        d = pair_dists[p]
        if d < 1e-12:
            continue
        diff_vec = pos[i] - pos[j]
        # d(pair_dists)/d(pos[i,k]) = diff_vec[k] / d
        # d(pair_dists/2)/d(pos[i,k]) = diff_vec[k] / (2*d)
        grad[2*i:2*i+2] += w * diff_vec / (2.0 * d)
        grad[2*j:2*j+2] += w * (-diff_vec) / (2.0 * d)
    
    return grad

def hex_init(scale_factor=1.0):
    """Generate hexagonal grid initialization."""
    points = []
    s = scale_factor
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.45 / max_abs
        positions = positions * scale
    
    return positions

def random_init():
    """Generate random initialization inside the square."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def lbfgs_optimize(positions, max_iter=500, tau=0.01):
    """
    Optimize using L-BFGS-B with soft-min objective.
    Returns (best_positions, best_radius).
    """
    x0 = positions.flatten()
    
    # Try different tau values
    best_pos = positions.copy()
    best_r, _ = compute_radius(positions)
    
    for t in [tau]:
        result = minimize(
            soft_min_radius,
            x0,
            jac=soft_min_radius_grad,
            args=(t,),
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 20
            }
        )
        
        pos = result.x.reshape(N, 2)
        r, valid = compute_radius(pos)
        
        if valid and r > best_r:
            best_r = r
            best_pos = pos.copy()
    
    return best_pos, best_r

def nelder_mead_polish(positions, max_iter=3000):
    """
    High-precision Nelder-Mead polish.
    """
    def objective(x):
        pos = x.reshape(N, 2)
        r, valid = compute_radius(pos)
        if not valid:
            return 10.0
        return -r
    
    x0 = positions.flatten()
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': max_iter,
            'xatol': 1e-12,
            'fatol': 1e-14,
            'adaptive': True
        }
    )
    
    pos = result.x.reshape(N, 2)
    r, valid = compute_radius(pos)
    
    if valid:
        return pos, r
    return positions, compute_radius(positions)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    # 1. Generate multiple initializations
    inits = []
    
    # Hexagonal grid
    inits.append(hex_init(1.0))
    
    # Scaled hex with noise
    pos_hex = hex_init(1.0)
    pos_noisy = pos_hex + np.random.normal(0, 0.01, size=(N, 2))
    pos_noisy = np.clip(pos_noisy, -0.499, 0.499)
    inits.append(pos_noisy)
    
    # Random initializations
    for _ in range(3):
        inits.append(random_init())
    
    # 2. Optimize each initialization with L-BFGS-B
    best_positions = None
    best_radius = -1.0
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        # L-BFGS-B optimization
        opt_pos, opt_r = lbfgs_optimize(init_pos, max_iter=300, tau=0.01)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_positions = opt_pos.copy()
    
    # 3. Nelder-Mead polish on the best solution
    if time.time() < deadline - 10.0:
        polished_pos, polished_r = nelder_mead_polish(best_positions, max_iter=5000)
        if polished_r > best_radius:
            best_radius = polished_r
            best_positions = polished_pos.copy()
    
    # 4. One more round: perturb and re-optimize if time allows
    if time.time() < deadline - 15.0 and best_positions is not None:
        # Try a few perturbations
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            perturbed = best_positions + np.random.normal(0, 0.005, size=(N, 2))
            perturbed = np.clip(perturbed, -0.499, 0.499)
            
            opt_pos, opt_r = lbfgs_optimize(perturbed, max_iter=200, tau=0.01)
            if opt_r > best_radius:
                best_radius = opt_r
                best_positions = opt_pos.copy()
        
        # Final polish
        if time.time() < deadline - 5.0:
            polished_pos, polished_r = nelder_mead_polish(best_positions, max_iter=3000)
            if polished_r > best_radius:
                best_radius = polished_r
                best_positions = polished_pos.copy()
    
    # Final validation
    if best_positions is None:
        best_positions = hex_init(1.0)
    
    final_radius, final_valid = compute_radius(best_positions)
    if not final_valid or final_radius < 0:
        # Fallback
        fallback = hex_init(1.0)
        return fallback
    
    return best_positions
