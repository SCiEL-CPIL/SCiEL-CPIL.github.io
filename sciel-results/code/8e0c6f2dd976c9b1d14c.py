import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _compute_score(centers):
    """
    Compute the exact radius for a given set of centers.
    This is the ground truth score used for final evaluation and tracking.
    """
    if len(centers) == 0 or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Distance to walls
    # min(0.5 - x, x + 0.5) for each coordinate, then min over all points and coords
    wall_dist = np.min(np.minimum(0.5 - centers, centers + 0.5))
    
    if wall_dist <= 0:
        return 0.0

    # Distance to neighbors
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _objective_and_grad(x):
    """
    Objective function: Minimize the negative soft-minimum distance.
    We want to maximize the minimum distance between any two circles and between circles and walls.
    
    We use a log-sum-exp trick to approximate the minimum distance smoothly.
    min(d_i) ≈ - (1/β) * log(Σ exp(-β * d_i))
    
    To maximize min(d_i), we minimize -min(d_i) ≈ (1/β) * log(Σ exp(-β * d_i)).
    
    We combine pair distances and wall distances into a single vector of 'distances' D.
    The effective radius r is min(D).
    We minimize f(x) = -soft_min(D).
    
    For stability, we use a moderate beta. Too high is non-differentiable (hard min), too low is insensitive.
    """
    beta = 50.0  # Temperature parameter for the soft min
    
    centers = x.reshape(-1, 2)
    
    # 1. Compute all pair distances (squared for efficiency in calculation, but we need distances for the exp)
    # Actually, for the gradient of exp(-beta * d), we need d.
    # diff[i, j] = c_i - c_j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero, will be masked
    
    # Distances
    dists = np.sqrt(dist_sq)
    
    # 2. Compute wall distances
    # d_wall_i = min(0.5 - x_i, x_i + 0.5, 0.5 - y_i, y_i + 0.5)
    # We can just compute the 4 values for each point and flatten them
    walls = np.stack([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=1) # shape (N, 4)
    
    # Flatten all distances into one vector
    # Pairs: N*(N-1)/2 unique pairs. We can use the upper triangle of dists.
    # To be efficient and vectorized for gradient, it's often easier to work with the full matrix 
    # or a flattened version of the unique pairs.
    # Let's flatten the upper triangle of dists (excluding diagonal).
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall dists are already computed.
    wall_dists = walls.flatten()
    
    # Combine
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min calculation
    # To avoid overflow in exp(-beta * d), shift by max
    max_d = np.max(all_dists)
    shifted = -beta * (all_dists - max_d)
    # exp(shifted) is <= 1
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + (-beta * max_d)
    
    # soft_min = max_d + (1/beta) * log_sum_exp ? 
    # Wait. 
    # min(d) ≈ (1/β) log Σ exp(β d_i) ... no, that's max.
    # min(d) ≈ - (1/β) log Σ exp(-β d_i)
    # So -min(d) ≈ (1/β) log Σ exp(-β d_i)
    
    # Our objective is to MINIMIZE -min(d).
    # So we minimize: (1/β) log Σ exp(-β d_i)
    
    obj_val = (1.0 / beta) * log_sum_exp
    
    # Gradient calculation
    # d/dx_j [ (1/β) log Σ exp(-β d_i) ]
    # = (1/β) * [ 1 / Σ exp(-β d_i) ] * Σ [ exp(-β d_i) * (-β) * (d d_i / dx_j) ]
    # = - [ Σ exp(-β d_i) * (d d_i / dx_j) ] / Σ exp(-β d_i)
    # Let p_i = exp(-β d_i) / Σ exp(-β d_k)  (softmax probabilities)
    # grad = - Σ p_i * (d d_i / dx_j)
    
    # We need to compute the weighted sum of gradients of distances.
    # p_i are the probabilities.
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    probs = exp_terms / sum_exp
    
    # Now we need to accumulate gradients.
    # grad_x = - Σ p_pair * grad(d_pair) - Σ p_wall * grad(d_wall)
    
    grad_centers = np.zeros_like(centers)
    
    # 1. Wall gradients
    # wall_dists are stacked. 
    # For each point i, we have 4 wall distances.
    # d(0.5 - x)/dx = -1
    # d(x + 0.5)/dx = 1
    # d(0.5 - y)/dy = -1
    # d(y + 0.5)/dy = 1
    
    wall_probs = probs[N*(N-1)//2:] # The last 4N probabilities correspond to walls
    wall_probs = wall_probs.reshape(N, 4)
    
    # grad_x from walls
    # term 0: 0.5-x, prob p0. d/dx = -1. contrib: -p0
    # term 1: x+0.5, prob p1. d/dx = 1. contrib: p1
    # term 2: 0.5-y, prob p2. d/dy = -1. contrib: -p2
    # term 3: y+0.5, prob p3. d/dy = 1. contrib: p3
    
    grad_centers[:, 0] += -wall_probs[:, 0] + wall_probs[:, 1]
    grad_centers[:, 1] += -wall_probs[:, 2] + wall_probs[:, 3]
    
    # 2. Pair gradients
    pair_probs = probs[:N*(N-1)//2]
    
    # We need to map pair_probs back to the (i, j) indices and add gradients.
    # d/dx_i (dist_ij) = (x_i - x_j) / dist_ij
    # d/dx_j (dist_ij) = (x_j - x_i) / dist_ij
    
    # Instead of looping, we can use matrix operations or sparse accumulation.
    # Since N=200, looping over ~20k pairs in Python is slow. 
    # Let's use a vectorized approach with scatter-add logic or just a loop if it's fast enough.
    # 20k iterations is fine for one gradient evaluation if the inner loop is tight, but 
    # L-BFGS calls gradient many times. 
    # Let's try to vectorize the accumulation.
    
    # We have indices iu (i, j) and probs.
    # diff_ij = centers[i] - centers[j]
    # grad_i += prob * diff_ij / dist
    # grad_j -= prob * diff_ij / dist
    
    # Let's create a sparse-like accumulation.
    # Actually, for N=200, a simple loop in numpy might be okay if we pre-compute diffs.
    # But let's try a fully vectorized scatter.
    
    # Create a matrix P of size NxN where P[i,j] = prob if (i,j) is a pair, 0 otherwise.
    # This is memory heavy (200x200 is small though).
    P = np.zeros((N, N))
    P[iu] = pair_probs
    
    # diff is already computed (NxNx2)
    # dist is already computed (NxN)
    # We need to handle division by zero. We set diagonal to 1.0 in dist_sq earlier, so dist diagonal is 1.0.
    # But P diagonal is 0.0, so it won't contribute.
    
    # Weighted diff: W[i,j] = P[i,j] * diff[i,j] / dist[i,j]
    # Note: diff[i,j] = c_i - c_j.
    # grad_c_i gets + sum_j W[i,j]
    # grad_c_j gets - sum_i W[i,j] ... wait.
    # d/dc_i dist_ij = (c_i - c_j)/dist_ij
    # d/dc_j dist_ij = (c_j - c_i)/dist_ij = - (c_i - c_j)/dist_ij
    
    # So grad_c = - (sum_j P[i,j] * (c_i - c_j)/dist_ij)
    
    # Let's compute the sum of weighted differences for each i.
    # S_i = sum_j P[i,j] * (c_i - c_j) / dist_ij
    
    # We can compute this by:
    # M = P / dist  (elementwise)
    # S = M @ (centers - centers)? No.
    # S_i = sum_j M[i,j] * (c_i - c_j)
    # S_i = c_i * sum_j M[i,j] - sum_j M[i,j] * c_j
    
    M = P / dist
    # Handle any remaining NaNs from 0/0, though P is 0 on diag and dist is 1 on diag.
    M = np.nan_to_num(M)
    
    row_sums = M.sum(axis=1)
    S = centers * row_sums[:, np.newaxis] - (M @ centers)
    
    # The gradient contribution to the objective (which is -softmin) is:
    # grad_obj = - sum p_i grad(d_i)
    # We already calculated grad_centers for walls as -sum p_wall grad(d_wall).
    # For pairs, the term in the sum is p_ij * grad(d_ij).
    # grad_obj_pair = - S
    
    grad_centers += -S
    
    # Flatten gradient
    grad = grad_centers.flatten()
    
    return obj_val, grad

def _generate_hex_grid(N):
    """Generate a hexagonal grid of points."""
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    # Initializations
    inits = []
    
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    square_grid = _generate_square_grid(N)
    inits.append(square_grid)
    
    # Jittered Hex
    if hex_grid is not None:
        np.random.seed(42)
        jittered_hex = hex_grid + np.random.normal(0, 0.01, hex_grid.shape)
        jittered_hex = np.clip(jittered_hex, -0.49, 0.49)
        inits.append(jittered_hex)
        
    # Random
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)

    # Sort by initial score
    scored_inits = []
    for init in inits:
        score = _compute_score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Simulated Annealing Loop with L-BFGS-B
    # We start with a high "temperature" (perturbation size) and cool down.
    # Each step:
    # 1. Pick a starting point (best found so far or a new random init).
    # 2. Run L-BFGS-B to local optimum.
    # 3. Evaluate score.
    # 4. If better, update best.
    # 5. Perturb the best solution and use as next start.
    
    current_best_centers = scored_inits[0][1].copy()
    current_best_score = scored_inits[0][0]
    
    # We will iterate through a mix of inits and perturbations.
    # To save time, we limit L-BFGS-B iterations and function evaluations.
    
    # Pre-allocate result
    best_centers = current_best_centers
    best_score = current_best_score
    
    # List of starting points to try
    # 1. The sorted inits
    # 2. Perturbations of the best found so far
    
    starts_to_try = [c for _, c in scored_inits]
    
    # We'll use a counter to decide when to generate new perturbations
    perturbation_count = 0
    
    for idx, start_c in enumerate(starts_to_try):
        if time.time() - start_time > time_limit:
            break
        
        # If we have finished the initial inits, generate perturbations from the best
        if idx >= len(starts_to_try):
             # This logic is tricky in a simple for loop. 
             # Let's restructure: 
             # Loop while time < limit:
             #   if have more inits, take next init.
             #   else, perturb best.
             pass
             
    # Restructured Loop
    init_idx = 0
    while time.time() - start_time < time_limit - 5:
        
        # Select starting point
        if init_idx < len(scored_inits):
            start_c = scored_inits[init_idx][1].copy()
            init_idx += 1
        else:
            # Perturb the best solution found so far
            # Temperature decays as we do more iterations
            temp = 0.05 * (0.9 ** perturbation_count)
            if temp < 0.005:
                # If temp is too low, we might be stuck. 
                # Try a random restart from a different init?
                # Or just break if we've done enough.
                if perturbation_count > 20:
                    break
                temp = 0.02 # Reset to a moderate temp for a "kick"
            
            np.random.seed(int(time.time() * 1000) % 100000)
            start_c = best_centers + np.random.normal(0, temp, (N, 2))
            start_c = np.clip(start_c, -0.49, 0.49)
            perturbation_count += 1
            
        # Run L-BFGS-B
        x0 = start_c.flatten()
        
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            # maxiter and maxfun to prevent hanging
            # L-BFGS-B is fast, but we want to ensure we don't spend too long on one run.
            # 1000 iterations is usually enough for convergence.
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if res.success or res.status == 0:
            opt_centers = res.x.reshape(-1, 2)
            opt_score = _compute_score(opt_centers)
            
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers
        else:
            # Even if not "success", the result might be better than start
            # But usually if it fails, it's due to maxiter, so the point is likely good.
            opt_centers = res.x.reshape(-1, 2)
            opt_score = _compute_score(opt_centers)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers

    # Final check
    if best_centers is None:
        best_centers = scored_inits[0][1]
        
    return best_centers

# Test locally
if __name__ == "__main__":
    c = construct()
    print("Score:", _compute_score(c))
