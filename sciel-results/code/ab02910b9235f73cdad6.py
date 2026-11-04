<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall constraints: distance from center to nearest wall
    # d_i = min(0.5 - x, x - (-0.5), 0.5 - y, ...) = 0.5 - max(|x|, |y|, |z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # Calculate all pairwise distances efficiently
    # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    # This creates a large array (100*100*3 = 30000 floats), which is fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Get upper triangle indices (excluding diagonal)
    i, j = np.triu_indices(N, 1)
    min_pair_dist = np.min(dists[i, j])
    
    # The radius is limited by half the minimum pair distance
    # and the minimum wall distance
    return min(min_pair_dist / 2.0, min_wall_dist)

def _energy_and_gradient(centers, r_target, k=50.0):
    """
    Computes a smooth energy function and its gradient.
    Energy is the sum of soft violations of the constraints.
    We want to maximize r, so we minimize energy for a fixed r_target.
    
    Constraints:
    1. Pairwise distance >= 2*r_target
    2. Wall distance >= r_target
    
    Soft violation for pair (i,j): max(0, 2*r - d_ij)^2
    Soft violation for wall (i): max(0, r - w_i)^2
    
    To make it differentiable, we use a smooth approximation:
    soft_max(0, x) approx x + log(1 + exp(-x/k)) * k ... wait, standard softplus is log(1+exp(x)).
    Let's use: f(x) = x if x>0 else 0. Smooth version: (x + sqrt(x^2 + eps))/2? No.
    Let's use the standard softplus for the penalty term itself.
    Penalty = (softplus(2*r - d) / k)^2 ?
    
    Actually, simpler:
    We minimize: sum_{i<j} (max(0, 2*r - d_ij))^2 + sum_i (max(0, r - w_i))^2
    
    Gradient of max(0, x)^2 is 2*max(0, x) * sign(x) ? No, 2*max(0, x) if x>0 else 0.
    This is not differentiable at 0.
    
    Let's use a smooth approximation:
    h(x) = (x + sqrt(x^2 + eps))/2 is not quite right for max(0,x).
    max(0, x) approx (x + sqrt(x^2 + eps))/2 is actually approx |x|/2 + x/2.
    If x>0, approx x. If x<0, approx 0.
    So max(0, x)^2 approx ((x + sqrt(x^2 + eps))/2)^2.
    
    Let's stick to the non-smooth gradient but use a small epsilon in the sqrt to avoid division by zero if we used analytic gradient of the smooth version.
    However, L-BFGS-B handles non-smoothness okay-ish, but smooth is better.
    
    Let's define:
    E = sum_{i<j} (max(0, 2r - d_ij))^2 + sum_i (max(0, r - w_i))^2
    
    Gradient wrt center k:
    For pair (k, j):
    term = max(0, 2r - d_kj)
    grad_term = -2 * term * (c_k - c_j) / d_kj   (if term > 0)
    
    For wall k:
    term = max(0, r - w_k)
    w_k = 0.5 - max(|x_k|, |y_k|, |z_k|)
    This is tricky because of the max.
    w_k is the distance to the closest face.
    If x_k is the dominant coordinate (|x_k| > |y_k|, |z_k|), then w_k = 0.5 - |x_k|.
    grad w_k / grad x_k = -sign(x_k).
    So grad term = -2 * term * (-sign(x_k)) = 2 * term * sign(x_k).
    
    Let's implement this carefully.
    """
    centers = np.array(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Wall constraints
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    violations = np.maximum(0, r_target - wall_dists)
    
    # Sum of squared violations
    energy += np.sum(violations**2)
    
    # Gradient of wall violations
    # We need to know which coordinate was the max.
    # If there are ties, gradient is undefined, but we can pick one or average.
    # For now, let's just use the gradient of the active constraint.
    # If |x| is max, grad_x = -sign(x), grad_y=0, grad_z=0.
    for i in range(N):
        if violations[i] > 0:
            # Find which coordinate had the max absolute value
            abs_c = np.abs(centers[i])
            max_idx = np.argmax(abs_c)
            sign = np.sign(centers[i, max_idx])
            if sign == 0: sign = 1.0 # Edge case, center at 0
            
            # d(wall_dist)/d(centers[i, max_idx]) = -sign
            # d(violation)/d(centers[i, max_idx]) = -d(wall_dist)/d(centers[i, max_idx]) = sign
            # d(energy)/d(centers[i, max_idx]) = 2 * violation * sign
            grad[i, max_idx] += 2.0 * violations[i] * sign
            
    # Pairwise constraints
    # To be efficient, we can loop over pairs, but N=100 means 4950 pairs.
    # 4950 * 100 iterations is 500k operations, which is fast in Python if vectorized per pair? No, loop is slow.
    # Let's vectorize the distance calculation, then loop for gradient accumulation?
    # Or just loop over all pairs. 5000 iterations is acceptable.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask for upper triangle
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_dists = dists[i_idx, j_idx]
    pair_violations = np.maximum(0, 2.0 * r_target - pair_dists)
    
    energy += np.sum(pair_violations**2)
    
    # Gradient accumulation
    # For each violating pair (i, j), add gradient to i and j.
    # grad_i += -2 * v_ij * (c_i - c_j) / d_ij
    # grad_j +=  2 * v_ij * (c_i - c_j) / d_ij
    
    # We need to accumulate into the grad array.
    # Vectorized accumulation is tricky with variable indices.
    # Let's loop over the violating pairs only.
    viol_mask = pair_violations > 0
    if np.any(viol_mask):
        viol_i = i_idx[viol_mask]
        viol_j = j_idx[viol_mask]
        viol_v = pair_violations[viol_mask]
        viol_d = pair_dists[viol_mask]
        viol_diff = diff[viol_i, viol_j] # Shape (num_viol, 3)
        
        # Factor: 2 * v / d
        factors = 2.0 * viol_v / viol_d
        
        # Update grad for i: -factor * diff
        # Update grad for j: +factor * diff
        
        # Use np.add.at for unbuffered addition to handle duplicate indices (though here i!=j always)
        np.add.at(grad, viol_i, -factors * viol_diff)
        np.add.at(grad, viol_j, factors * viol_diff)
        
    return energy, grad

def _optimize_for_radius(centers, r_target, max_iter=200, time_limit=10.0):
    """
    Try to find a configuration where radius >= r_target.
    Returns (success, new_centers, final_energy)
    """
    start_time = time.time()
    
    # Flatten centers for scipy
    x0 = centers.flatten()
    
    def fun(x):
        c = x.reshape(N, 3)
        e, _ = _energy_and_gradient(c, r_target)
        return e
    
    def jac(x):
        c = x.reshape(N, 3)
        _, g = _energy_and_gradient(c, r_target)
        return g.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    # But actually, if they go outside, wall violation increases.
    # It's safer to constrain them to [-0.5+r_target, 0.5-r_target]?
    # No, we want to allow them to move anywhere, the penalty will push them in.
    # However, L-BFGS-B works best with bounds.
    # Let's use bounds [-0.5, 0.5].
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            fun, x0, jac=jac, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        # Check time
        if time.time() - start_time > time_limit:
            return False, centers, fun(x0)
            
        final_centers = res.x.reshape(N, 3)
        final_energy = res.fun
        
        # Check if energy is small enough
        # If energy < 1e-6, we consider it a success
        if final_energy < 1e-6:
            return True, final_centers, final_energy
        else:
            return False, final_centers, final_energy
            
    except Exception:
        return False, centers, 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial positions: Random uniform
    # A better init might be a grid, but random is fine if we optimize.
    np.random.seed(42) # For reproducibility in testing, remove for final? No, keep it for consistent behavior.
    pos = np.random.uniform(-0.4, 0.4, (N, 3))
    
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Binary search for the maximum radius
    # Lower bound: 0
    # Upper bound: 0.5 (max possible)
    # But 0.5 is impossible for N=100.
    # Let's start with a reasonable guess.
    # Specht's radius is ~0.111.
    # Let's search between 0.05 and 0.15.
    
    lo = 0.05
    hi = 0.15
    
    # First, try to improve the initial radius if it's below lo
    # If best_r < lo, set lo = best_r
    
    # Iterative refinement
    # We will do several passes of binary search.
    # Each pass tries to find a feasible solution for a given r.
    
    # To speed up, we can use the result of a previous success as the start for the next.
    
    # Let's do a coarse binary search first.
    # 10 iterations of binary search: range 0.1 / 2^10 ~ 1e-4 precision.
    
    for iteration in range(12):
        if time.time() > deadline - 5.0:
            break
            
        mid = (lo + hi) / 2.0
        
        # Try to optimize for radius mid
        success, new_pos, energy = _optimize_for_radius(best_pos, mid, max_iter=100, time_limit=5.0)
        
        if success:
            # Update best
            best_pos = new_pos
            # Recalculate actual radius to be safe
            actual_r = _compute_radius(best_pos)
            if actual_r > best_r:
                best_r = actual_r
            lo = mid
        else:
            hi = mid
            
    # After binary search, do a local polish with a slightly lower radius to ensure stability?
    # Or just return best_pos.
    
    # One final check
    final_r = _compute_radius(best_pos)
    if final_r > best_r:
        best_r = final_r
        
    return best_pos
=======
import numpy as np
import time
from scipy.optimize import minimize

N = 100
LATENT_DIM = 50  # Low-dimensional latent space

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i, j = np.triu_indices(N, 1)
    min_pair_dist = np.min(dists[i, j])
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _network_forward(latent, W1, b1, W2, b2):
    """
    Forward pass through a 2-layer MLP with tanh output.
    latent: (LATENT_DIM,)
    Returns: (N, 3) centers in [-0.5, 0.5]^3
    """
    # Layer 1: ReLU activation
    h = np.maximum(0, latent @ W1.T + b1)  # (N, hidden)
    # Layer 2: tanh activation, scaled to [-0.5, 0.5]
    out = np.tanh(h @ W2.T + b2) * 0.5  # (N, 3)
    return out

def _network_forward_grad(latent, W1, b1, W2, b2, d_centers):
    """
    Backward pass: compute gradient of loss w.r.t. latent given d_centers (grad w.r.t. centers).
    Returns: d_latent (LATENT_DIM,)
    """
    # Recompute forward pass
    h_pre = latent @ W1.T + b1  # (N, hidden)
    h = np.maximum(0, h_pre)
    out_pre = h @ W2.T + b2  # (N, 3)
    centers = np.tanh(out_pre) * 0.5
    
    # Backward through output layer
    # d_centers is (N, 3)
    # d/d(out_pre) = d_centers / 0.5 * (1 - tanh^2(out_pre)) = 2 * d_centers * (1 - tanh^2(out_pre))
    tanh_out = np.tanh(out_pre)
    dout_pre = d_centers * 2.0 * (1.0 - tanh_out**2)  # (N, 3)
    
    # d/d(h) = dout_pre @ W2  (N, hidden)
    dh = dout_pre @ W2  # (N, hidden)
    
    # Backward through ReLU
    dh_pre = dh * (h_pre > 0)  # (N, hidden)
    
    # d/d(latent) = dh_pre @ W1  (LATENT_DIM,)
    d_latent = dh_pre @ W1  # (LATENT_DIM,)
    
    return d_latent

def _soft_min_radius_obj_grad(centers, beta=20.0):
    """
    Smooth surrogate for minimizing the negative of the minimum radius.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Equivalently, minimize -r.
    
    Use soft-min: -softmin(dists) where dists are the constraint distances.
    
    For each sphere i, wall distance: w_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    For each pair (i,j), half-distance: d_ij/2
    
    softmin of a set {a_k} = -1/beta * log(sum(exp(-beta * a_k)))
    
    We want to maximize softmin, so minimize -softmin.
    """
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs  # (N,)
    
    # Pairwise half-distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_half_dists = dists[i_idx, j_idx] / 2.0  # (4950,)
    
    # Combine all constraint distances
    all_dists = np.concatenate([wall_dists, pair_half_dists])  # (N + 4950,)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * dists)))
    # For numerical stability, subtract max
    max_dist = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_dist))
    log_sum = np.log(np.sum(exp_terms))
    soft_min = max_dist - log_sum / beta
    
    # Objective: -soft_min (we minimize this)
    obj = -soft_min
    
    # Gradient: d(soft_min)/d(centers) = -sum_k (exp(-beta*(a_k - max)) / sum_j(exp(-beta*(a_j - max)))) * d(a_k)/d(centers)
    # Let p_k = exp(-beta*(a_k - max)) / sum(exp(...))
    # d(soft_min)/d(centers) = -sum_k p_k * d(a_k)/d(centers)
    # d(obj)/d(centers) = sum_k p_k * d(a_k)/d(centers)
    
    p = exp_terms / np.sum(exp_terms)  # (N + 4950,)
    
    grad_centers = np.zeros_like(centers)
    
    # Wall gradient contributions
    # a_k for wall k: w_k = 0.5 - max(|x_k|, |y_k|, |z_k|)
    # d(w_k)/d(centers[k, coord]) = -sign(centers[k, coord]) if that coord is the max abs
    for k in range(N):
        if p[k] > 0:
            abs_c = np.abs(centers[k])
            max_idx = np.argmax(abs_c)
            sign = np.sign(centers[k, max_idx])
            if sign == 0:
                sign = 1.0
            # d(w_k)/d(centers[k, max_idx]) = -sign
            # d(obj)/d(centers[k, max_idx]) += p[k] * (-sign)
            grad_centers[k, max_idx] += p[k] * (-sign)
    
    # Pairwise gradient contributions
    # a_m for pair (i,j): d_ij / 2
    # d(d_ij/2)/d(c_i) = -(c_i - c_j) / d_ij
    # d(d_ij/2)/d(c_j) = +(c_i - c_j) / d_ij
    # d(obj)/d(c_i) += p_m * (-(c_i - c_j) / d_ij)
    # d(obj)/d(c_j) += p_m * (+(c_i - c_j) / d_ij)
    
    # p for pairs starts at index N
    pair_p = p[N:]  # (4950,)
    
    # For each pair, compute the gradient contribution
    # Vectorized:
    pair_diff = centers[i_idx] - centers[j_idx]  # (4950, 3)
    pair_d = dists[i_idx, j_idx]  # (4950,)
    
    # Avoid division by zero
    safe_d = np.maximum(pair_d, 1e-10)
    
    # Gradient factor for each pair: p_m / d_m
    factors = pair_p / safe_d  # (4950,)
    
    # d(obj)/d(c_i) += -factors * pair_diff
    # d(obj)/d(c_j) += +factors * pair_diff
    np.add.at(grad_centers, i_idx, -factors[:, np.newaxis] * pair_diff)
    np.add.at(grad_centers, j_idx, factors[:, np.newaxis] * pair_diff)
    
    return obj, grad_centers

def _init_network_params(seed=42):
    """Initialize network weights with small random values."""
    rng = np.random.RandomState(seed)
    hidden_dim = 64
    
    W1 = rng.randn(LATENT_DIM, hidden_dim) * 0.1  # (LATENT_DIM, hidden)
    b1 = rng.randn(hidden_dim) * 0.1
    W2 = rng.randn(hidden_dim, 3) * 0.1  # (hidden, 3)
    b2 = rng.randn(3) * 0.1
    
    return W1, b1, W2, b2

def _optimize_latent(latent0, W1, b1, W2, b2, beta=20.0, max_iter=500, time_limit=30.0):
    """
    Optimize the latent vector to maximize the soft-min radius.
    Returns: optimized latent, best centers found
    """
    start_time = time.time()
    best_latent = latent0.copy()
    best_obj = np.inf
    best_centers = _network_forward(latent0, W1, b1, W2, b2)
    best_r = _compute_radius(best_centers)
    
    def fun(latent):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        centers = _network_forward(latent, W1, b1, W2, b2)
        obj, _ = _soft_min_radius_obj_grad(centers, beta)
        return obj
    
    def jac(latent):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        centers = _network_forward(latent, W1, b1, W2, b2)
        _, grad_centers = _soft_min_radius_obj_grad(centers, beta)
        grad_latent = _network_forward_grad(latent, W1, b1, W2, b2, grad_centers)
        return grad_latent
    
    try:
        res = minimize(
            fun, latent0, jac=jac, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_obj:
            new_latent = res.x
            new_centers = _network_forward(new_latent, W1, b1, W2, b2)
            new_r = _compute_radius(new_centers)
            
            if new_r > best_r:
                best_latent = new_latent
                best_centers = new_centers
                best_r = new_r
                best_obj = res.fun
        
    except TimeoutError:
        pass
    except Exception:
        pass
    
    return best_latent, best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize network
    W1, b1, W2, b2 = _init_network_params(seed=42)
    
    # Multiple random restarts in latent space
    best_centers = None
    best_r = 0.0
    
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial latent
        np.random.seed(42 + restart)
        latent0 = np.random.randn(LATENT_DIM) * 0.5
        
        # Optimize
        latent_opt, centers_opt, r_opt = _optimize_latent(
            latent0, W1, b1, W2, b2, 
            beta=20.0, max_iter=300, 
            time_limit=time_per_restart - 2.0
        )
        
        if r_opt > best_r:
            best_r = r_opt
            best_centers = centers_opt
    
    # If no valid centers found, use a grid initialization as fallback
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        # Simple grid-like initialization
        best_centers = np.zeros((N, 3))
        for i in range(N):
            # Place spheres in a rough grid pattern
            ix = i % 5
            iy = (i // 5) % 5
            iz = i // 25
            best_centers[i, 0] = -0.4 + 0.2 * ix
            best_centers[i, 1] = -0.4 + 0.2 * iy
            best_centers[i, 2] = -0.4 + 0.2 * iz
        best_r = _compute_radius(best_centers)
    
    # Final verification
    if not np.all(np.isfinite(best_centers)):
        # Last resort: uniform random
        np.random.seed(42)
        best_centers = np.random.uniform(-0.4, 0.4, (N, 3))
        best_r = _compute_radius(best_centers)
    
    return best_centers
>>>>>>> REPLACE
