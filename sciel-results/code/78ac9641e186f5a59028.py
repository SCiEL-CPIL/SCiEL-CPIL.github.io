import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50
DEADLINE = 135.0  # Leave 15s margin before 150s kill

def _compute_radius(centers):
    """
    Compute the maximum common radius for given centers.
    Returns 0.0 if invalid.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall constraint: min_i min_k (0.5 - |c_ik|)
    # centers are in [-0.5, 0.5]
    # Distance to wall is 0.5 - max(abs(x), abs(y), abs(z)) for each sphere? 
    # No, the constraint is per coordinate.
    # The sphere is inside if for all k, |c_ik| + r <= 0.5.
    # So r <= 0.5 - |c_ik| for all i,k.
    # r <= min_{i,k} (0.5 - |c_ik|)
    
    abs_centers = np.abs(centers)
    wall_dist = np.min(0.5 - abs_centers)
    if wall_dist <= 0:
        return 0.0

    # Pairwise constraint: min_{i<j} |c_i - c_j| / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[i_idx, j_idx])
    
    r = min(wall_dist, min_pair_dist / 2.0)
    return float(r)

def _soft_min_obj_and_grad(x, beta):
    """
    Minimize the negative soft-min of distances.
    x is flat array of shape (N*3,).
    Returns (fun, grad).
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Wall distances
    # dist_wall_i = 0.5 - max(abs(c_ix), abs(c_iy), abs(c_iz))
    # Actually, the constraint is separable per coordinate?
    # No, the radius is limited by the closest wall in ANY direction.
    # r_i <= min(0.5 - |x_i|, 0.5 - |y_i|, 0.5 - |z_i|)
    # The global radius r is min( min_i r_i, min_{i<j} d_ij/2 )
    # So we want to maximize min( min_i min_k (0.5 - |c_ik|), min_{i<j} d_ij/2 )
    
    # Let's define the set of "distances" D:
    # 1. For each pair i<j, d_ij
    # 2. For each center i and coord k, 2 * (0.5 - |c_ik|) ? 
    # Wait, the radius is r.
    # Constraint: d_ij >= 2r  => r <= d_ij/2
    # Constraint: 2(0.5 - |c_ik|) >= 2r => r <= 0.5 - |c_ik|
    # To combine these into a single soft-min, we can scale the wall distances by 2?
    # If we define "effective distance" for walls as D_wall = 2 * (0.5 - |c_ik|), then r <= D_wall / 2.
    # Then we can take the soft-min over all d_ij and D_wall, and divide the result by 2 to get r.
    # This unifies the scaling.
    
    # Let's use this unified approach.
    # D_matrix is (N, N) for pairs, but we only need i<j.
    # D_walls is (N, 3).
    
    # To avoid recomputing diff and dist every time in a way that creates large temporary arrays if possible,
    # but N=50 is small.
    
    # Pairwise
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs = dist[i_idx, j_idx]
    
    # Walls
    # d_wall_i_k = 2.0 * (0.5 - abs(c_ik))
    d_walls = 2.0 * (0.5 - np.abs(centers)) # shape (N, 3)
    
    # Combine all distances into a single vector D
    # Size: N*(N-1)/2 + N*3 = 1225 + 150 = 1375
    D = np.concatenate([d_pairs, d_walls.ravel()])
    
    # Soft min: -1/beta * log( sum(exp(-beta * D)) )
    # To avoid overflow/underflow, use log-sum-exp trick.
    # max_D = np.max(D)
    # log_sum = max_D + log(sum(exp(-beta * (D - max_D))))
    # soft_min = -1/beta * log_sum
    
    # However, we want to MAXIMIZE the soft min.
    # So we MINIMIZE -soft_min = 1/beta * log( sum(exp(-beta * D)) )
    
    # Gradient of f = 1/beta * log( sum_j exp(-beta * D_j) )
    # f' = 1/beta * ( 1 / sum(exp(-beta D_j)) ) * sum( exp(-beta D_j) * (-beta) * D_j' )
    # f' = - ( sum_j exp(-beta D_j) * D_j' ) / sum_j exp(-beta D_j)
    # f' = - sum_j p_j * D_j'  where p_j = exp(-beta D_j) / sum(exp(-beta D))
    
    # Note: D_j depends on x.
    # D_pair_ij = |c_i - c_j|
    # D_wall_ik = 2 * (0.5 - |c_ik|)
    
    # Let's compute probabilities p
    # Shift for stability
    max_D = np.max(D)
    # If max_D is very large, exp(-beta * D) underflows.
    # exp(-beta * (D - max_D)) is safe.
    exponents = -beta * (D - max_D)
    exp_vals = np.exp(exponents)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        # This shouldn't happen if D is finite
        return 0.0, np.zeros_like(x)
        
    probs = exp_vals / sum_exp
    
    # Objective value
    obj = (1.0 / beta) * (max_D + np.log(sum_exp))
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pairwise part
    # d_pair_ij = sqrt(sum((c_i - c_j)^2))
    # d/d(c_i) d_pair = (c_i - c_j) / d_pair
    # d/d(c_j) d_pair = (c_j - c_i) / d_pair
    
    # We need to accumulate -prob * d(D)/dx
    # grad += - sum_{i<j} p_ij * ( (c_i - c_j)/d_ij for i, (c_j - c_i)/d_ij for j )
    
    # Indices
    n_pairs = len(i_idx)
    
    if n_pairs > 0:
        p_pairs = probs[:n_pairs]
        d_pairs_masked = d_pairs
        
        # Vector (c_i - c_j)
        vecs = centers[i_idx] - centers[j_idx]
        
        # Coefficients for c_i: -p * (1/d) * vec
        # Coefficients for c_j: -p * (1/d) * (-vec) = p * (1/d) * vec
        
        # Handle division by zero (if d=0, gradient is undefined, but prob might be high)
        # If d=0, we can set gradient to 0 or small random? 
        # In practice, d=0 is rare.
        inv_d = np.where(d_pairs_masked > 1e-12, 1.0 / d_pairs_masked, 0.0)
        
        coeffs = -p_pairs * inv_d
        
        # Accumulate for i
        # grad_centers[i] += coeffs[:, None] * vecs
        # grad_centers[j] -= coeffs[:, None] * vecs
        
        # Using np.add.at for scatter
        grad_centers = np.zeros_like(centers)
        np.add.at(grad_centers, i_idx, coeffs[:, np.newaxis] * vecs)
        np.add.at(grad_centers, j_idx, -coeffs[:, np.newaxis] * vecs)
        
        grad = grad_centers.ravel()
        
        # Wall part
        # D_wall_ik = 2 * (0.5 - |c_ik|)
        # d(D_wall_ik)/d(c_ik) = -2 * sign(c_ik)
        # Contribution to grad: -p_wall * (-2 * sign(c_ik)) = 2 * p_wall * sign(c_ik)
        
        p_walls = probs[n_pairs:]
        # p_walls is (N*3,)
        # centers is (N, 3)
        # grad_wall = 2 * p_walls.reshape(N, 3) * sign(centers)
        
        # Add to grad
        grad_wall = 2.0 * p_walls.reshape(N, 3) * np.sign(centers)
        grad += grad_wall.ravel()
        
    else:
        # Only walls
        p_walls = probs
        grad_wall = 2.0 * p_walls.reshape(N, 3) * np.sign(centers)
        grad = grad_wall.ravel()
        
    return obj, grad

def _generate_initial_guess(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    
    strategy = seed % 5
    
    if strategy == 0:
        # Simple cubic grid + jitter
        # 4x4x4 = 64 points, take 50
        xs = np.linspace(-0.4, 0.4, 4)
        ys = np.linspace(-0.4, 0.4, 4)
        zs = np.linspace(-0.4, 0.4, 4)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        # Shuffle
        idx = rng.permutation(len(grid))
        centers = grid[idx[:N]]
        
    elif strategy == 1:
        # BCC-like lattice
        # 3x3x3 BCC
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.4
                    y = (j - 1) * 0.4
                    z = (k - 1) * 0.4
                    pts.append([x, y, z])
                    if i < 2 and j < 2 and k < 2:
                         pts.append([x + 0.2, y + 0.2, z + 0.2])
        # 35 points
        centers[:35] = np.array(pts)
        centers[35:] = rng.uniform(-0.35, 0.35, (N - 35, 3))
        
    elif strategy == 2:
        # Random uniform
        centers = rng.uniform(-0.4, 0.4, (N, 3))
        
    elif strategy == 3:
        # Helical
        t = np.linspace(0, 10 * np.pi, N)
        r = 0.3 * t / (10 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = (t / (10 * np.pi) - 0.5) * 0.8
        centers = np.column_stack((x, y, z))
        centers += rng.normal(0, 0.05, (N, 3))
        
    else:
        # Face centered cubic projection? Or just random with larger jitter
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _optimize_soft_min(centers, start_time, time_limit):
    """
    Run soft-min optimization with beta annealing.
    """
    x0 = centers.ravel()
    best_x = x0.copy()
    best_r = _compute_radius(centers)
    
    # Beta schedule
    # Start low, increase.
    # High beta -> sharper min -> harder to optimize but more accurate.
    # Low beta -> smoother -> easier to optimize but less accurate.
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    # We will loop through betas, and for each, run a few L-BFGS-B iterations.
    # Or run L-BFGS-B with a changing beta? No, beta is fixed in the function.
    # So we run separate optimizations for each beta, using the result of the previous as start.
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Define objective for this beta
        def obj(x):
            f, g = _soft_min_obj_and_grad(x, beta)
            return f, g
            
        try:
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < np.inf:
                x0 = res.x
                # Compute radius for this configuration
                r_curr = _compute_radius(x0.reshape(N, 3))
                if r_curr > best_r:
                    best_r = r_curr
                    best_x = x0.copy()
                    
        except Exception:
            # If optimization fails, keep previous x0
            pass
            
    return best_x.reshape(N, 3), best_r

def construct():
    start_time = time.time()
    time_limit = DEADLINE
    
    best_radius = 0.0
    best_centers = None
    
    # Number of restarts
    # Each restart takes ~1-5 seconds depending on convergence.
    # 150s / 3s = 50 restarts? Maybe too many.
    # Let's do 20 restarts, allowing more time for each.
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        centers = _generate_initial_guess(seed=i)
        
        # Run optimization
        opt_centers, opt_r = _optimize_soft_min(centers, start_time, time_limit)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_centers
            # Update global best
            # Optional: Polish the best solution found so far with a very high beta
            # if time allows.
            
    # Final Polish
    # If we have a best_centers, try to improve it slightly with a very high beta
    # or a direct energy minimization for the specific radius.
    
    if best_centers is not None and time.time() - start_time < time_limit - 5.0:
        # Try a few more steps with high beta
        x0 = best_centers.ravel()
        beta_final = 2000.0
        
        def obj_final(x):
            f, g = _soft_min_obj_and_grad(x, beta_final)
            return f, g
            
        try:
            res = minimize(
                obj_final,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < np.inf:
                final_r = _compute_radius(res.x.reshape(N, 3))
                if final_r > best_radius:
                    best_radius = final_r
                    best_centers = res.x.reshape(N, 3)
        except Exception:
            pass
            
    if best_centers is None:
        # Fallback
        best_centers = _generate_initial_guess(seed=99)
        
    return best_centers

if __name__ == "__main__":
    # For local testing
    start = time.time()
    centers = construct()
    r = _compute_radius(centers)
    print(f"Radius: {r}")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Centers shape: {centers.shape}")
