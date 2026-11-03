"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

N = 200

def _compute_radius(centers):
    """
    Compute the exact maximum radius for given centers.
    """
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Distance between pairs
    # Using cdist is efficient for N=200
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    
    r = min(min_wall, min_pair / 2.0)
    return max(0.0, r)

def _soft_min_log_sum_exp(values, beta):
    """
    Compute soft-min using log-sum-exp trick for numerical stability.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    # Shift by min to prevent overflow in exp
    m = np.min(values)
    if not np.isfinite(m):
        return 0.0
    shifted = values - m
    # exp(-beta * (x - m)) = exp(-beta*x + beta*m)
    # We want -1/beta * log(sum(exp(-beta * x)))
    # = -1/beta * log(sum(exp(-beta * (x-m)) * exp(-beta * m)))
    # = -1/beta * (log(sum(exp(-beta * (x-m)))) - beta * m)
    # = m - 1/beta * log(sum(exp(-beta * (x-m))))
    
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    if sum_exp == 0:
        return m
    return m - (1.0 / beta) * np.log(sum_exp)

def _objective_and_grad(centers, beta):
    """
    Compute the soft-min clearance and its gradient.
    Objective: Maximize soft_min(clearances)
    Clearances:
      1. Wall clearances: 0.5 - |c_i| for each coordinate
      2. Pair clearances: |c_i - c_j| / 2 for each pair
    
    We return the negative of the soft-min (to minimize) and the gradient.
    """
    n = len(centers)
    
    # 1. Wall clearances
    # x-coords: 0.5 - x, 0.5 + x
    # y-coords: 0.5 - y, 0.5 + y
    # Let's gather all wall distances
    # D_x_pos = 0.5 - centers[:, 0]
    # D_x_neg = 0.5 + centers[:, 0]
    # D_y_pos = 0.5 - centers[:, 1]
    # D_y_neg = 0.5 + centers[:, 1]
    
    wx_pos = 0.5 - centers[:, 0]
    wx_neg = 0.5 + centers[:, 0]
    wy_pos = 0.5 - centers[:, 1]
    wy_neg = 0.5 + centers[:, 1]
    
    wall_dists = np.concatenate([wx_pos, wx_neg, wy_pos, wy_neg])
    
    # 2. Pair clearances
    # Vectorized distance calculation
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=-1))
    # This creates a (N, N, 2) array which is 200*200*2 = 80k floats. Fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Pair clearance is dist / 2
    pair_dists = dists / 2.0
    # Flatten to get unique pairs? Or just use all? 
    # Using all i<j is standard. The log-sum-exp over all i,j (i!=j) works too, 
    # but weights each pair twice. Since it's a min, duplication doesn't change the 
    # location of the minimum, just the scale of the gradient slightly. 
    # To be precise, let's take upper triangle.
    iu = np.triu_indices(n, k=1)
    pair_dists_unique = pair_dists[iu]
    
    # Combine all clearances
    all_dists = np.concatenate([wall_dists, pair_dists_unique])
    
    # Compute soft-min
    # soft_min = -1/beta * log(sum(exp(-beta * d)))
    m = np.min(all_dists)
    if not np.isfinite(m):
         return 0.0, np.zeros_like(centers)
    
    shifted = all_dists - m
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    
    if sum_exp == 0:
        return m, np.zeros_like(centers)
    
    soft_min_val = m - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of soft_min w.r.t. centers
    # d/dx soft_min = sum_j ( exp(-beta * d_j) / sum_exp ) * (d/dx d_j)
    # Note: d/dx of -1/beta log(sum) = -1/beta * (1/sum) * sum(exp * d/dx(-beta d_j))
    # = -1/beta * (1/sum) * sum(exp * (-beta) * d_j')
    # = sum ( exp / sum ) * d_j'
    # So gradient = weighted average of gradients of individual distances.
    
    weights = exp_term / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Wall gradients
    # d(wx_pos)/dx = -1, d(wx_neg)/dx = 1
    # d(wy_pos)/dy = -1, d(wy_neg)/dy = 1
    
    # Indices:
    # 0:N: wx_pos (grad x = -1)
    # N:2N: wx_neg (grad x = 1)
    # 2N:3N: wy_pos (grad y = -1)
    # 3N:4N: wy_neg (grad y = 1)
    
    w_wp = weights[:N]
    w_wn = weights[N:2*N]
    w_yp = weights[2*N:3*N]
    w_yn = weights[3*N:4*N]
    
    grad[:, 0] += -w_wp + w_wn
    grad[:, 1] += -w_yp + w_yn
    
    # Pair gradients
    # d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2 * dist)
    # d(|c_i - c_j|/2)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|) = -(c_i - c_j) / (2 * dist)
    
    # We have unique pairs in iu.
    # weights for pairs are from index 4N onwards
    w_pairs = weights[4*N:]
    
    # diff[i, j] = c_i - c_j
    # dists[i, j]
    # For each pair (i, j) in iu:
    # grad[i] += w * (c_i - c_j) / (2 * dist)
    # grad[j] += w * (c_j - c_i) / (2 * dist) = -w * (c_i - c_j) / (2 * dist)
    
    # Vectorized update
    # diff is (N, N, 2)
    # dists is (N, N)
    # We need to sum over j for each i, and over i for each j.
    
    # Let's construct a matrix G where G[i, j] is the vector gradient contribution to i from pair (i, j)
    # G[i, j] = w_{ij} * (c_i - c_j) / (2 * dist_{ij})
    # Note: w_{ij} is only non-zero for i<j.
    
    # Create a weight matrix (N, N)
    W_mat = np.zeros((n, n))
    W_mat[iu] = w_pairs
    
    # Compute normalized diff vectors
    # safe_dists = np.where(dists > 1e-10, dists, 1e-10)
    # unit_diff = diff / safe_dists[:,:,np.newaxis]
    # contrib = W_mat[:,:,np.newaxis] * unit_diff * 0.5
    
    # grad[i] = sum_j contrib[i, j]
    # grad[j] -= sum_i contrib[i, j]  (since contrib[j, i] is 0 for i<j, we just subtract the sum of contributions received)
    
    # Actually, simpler:
    # grad[i] = sum_{j != i} W_mat[i, j] * (c_i - c_j) / (2 * dist_{ij})
    
    # Let's just loop over the unique pairs for clarity and correctness, N=200 is small enough 
    # for a loop over ~20k pairs if optimized, but vectorized is better.
    
    # Vectorized approach:
    # We want grad = sum_j (W_mat[i,j] * (c_i - c_j) / (2*dist))
    
    # Let's use the full diff matrix
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_diff = diff / safe_dists[:, :, np.newaxis]
    
    # contrib[i, j] is the vector added to c_i due to interaction with c_j
    # It should be W_mat[i, j] * unit_diff[i, j] * 0.5
    # But W_mat is only upper triangular.
    # The force on i from j is symmetric in magnitude but opposite in direction.
    # F_ij on i = W_ij * u_ij
    # F_ji on j = W_ji * u_ji = 0 (since W is upper)
    # But physically, the pair contributes to both.
    # Our weight calculation `weights` came from `pair_dists_unique`.
    # So `w_pairs[k]` corresponds to pair (iu[0][k], iu[1][k]).
    # This weight applies to the distance term.
    # The gradient of the distance term w.r.t c_i is (c_i-c_j)/dist.
    # The gradient w.r.t c_j is (c_j-c_i)/dist.
    
    # So:
    # grad[i] += w * (c_i - c_j) / dist
    # grad[j] += w * (c_j - c_i) / dist
    
    # Let's build a matrix of vectors `Force` where Force[i, j] is the vector added to i from j.
    # Force[i, j] = W_mat[i, j] * unit_diff[i, j]
    # Note: W_mat[i, j] is non-zero only for i < j.
    # So Force[i, j] is non-zero for i < j.
    # Force[j, i] is zero in this matrix.
    
    # Total grad for i is sum_j Force[i, j].
    # Total grad for j is sum_i Force[j, i].
    # But Force[j, i] is 0 for i < j.
    # We need to account for the symmetric part.
    # The term in the objective is d_ij.
    # grad d_ij = u_ij (for i) and -u_ij (for j).
    # Weight is w_ij.
    # So contribution to grad_i is w_ij * u_ij.
    # Contribution to grad_j is w_ij * (-u_ij).
    
    # My W_mat has w_ij at [i, j] for i<j.
    # Let's compute Sum_i = np.sum(Force, axis=1). This gives sum_j Force[i, j].
    # This is correct for the "i" part of pairs where i is the smaller index.
    # For the "j" part (where j is the larger index), we need to subtract the sum of forces coming from smaller indices.
    
    # Let's just do it explicitly with the upper triangle indices to avoid confusion.
    # N=200, ~20k pairs. A loop might be slow in Python, but we can use numpy advanced indexing.
    
    i_idx, j_idx = iu
    # w_pairs corresponds to these pairs
    # u_ij = unit_diff[i_idx, j_idx]
    u_ij = unit_diff[i_idx, j_idx]
    
    # Vector to add to c_i
    vec_i = w_pairs[:, np.newaxis] * u_ij
    # Vector to add to c_j
    vec_j = -vec_i
    
    # Use np.add.at for accumulation
    np.add.at(grad, i_idx, vec_i)
    np.add.at(grad, j_idx, vec_j)
    
    # The objective is to MAXIMIZE soft_min.
    # So we return -soft_min_val and -grad.
    return -soft_min_val, -grad

def _initial_hex_grid(rows):
    """
    Generate a hexagonal packing layout.
    """
    cols = math.ceil(N / rows)
    
    # Estimate spacing
    d_w = 1.0 / (cols - 0.5)
    d_h = 1.0 / ((rows - 1) * math.sqrt(3)/2 + 1.0)
    d = min(d_w, d_h)
    
    pts = []
    row_spacing = d * math.sqrt(3) / 2
    
    for i in range(rows):
        y = -0.5 + d/2 + i * row_spacing
        if i % 2 == 0:
            n_in_row = cols
            x_start = -0.5 + d/2
        else:
            n_in_row = cols - 1
            x_start = -0.5 + d
            
        for j in range(n_in_row):
            x = x_start + j * d
            pts.append([x, y])
            
    pts = np.array(pts)
    
    if len(pts) < N:
        return None
    if len(pts) > N:
        pts = pts[:N]
        
    return pts

def _gradient_optimize(centers, time_limit):
    """
    Perform gradient-based optimization using L-BFGS-B from scipy or a simple gradient descent.
    Since we have bounds, L-BFGS-B is ideal.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_val = float('inf')
    best_centers = centers.copy()
    
    # We will run minimize with a callback to check time.
    # But minimize doesn't have a built-in time limit easily.
    # We can wrap the objective to check time and raise an exception or return best.
    # However, L-BFGS-B might be slow if it takes many iterations.
    # Let's use a simple gradient descent with adaptive step size or just L-BFGS-B with maxiter.
    
    # Given N=200, 400 variables. L-BFGS-B should be fast per iteration.
    # Let's try L-BFGS-B.
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj_fun(x):
        nonlocal start_time
        if time.time() > start_time + time_limit:
            raise TimeoutError()
        c = x.reshape(-1, 2)
        val, grad = _objective_and_grad(c, beta=100.0) # Start with moderate beta
        return val, grad
    
    try:
        res = minimize(obj_fun, centers.flatten(), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.x is not None:
            c_res = res.x.reshape(-1, 2)
            # Check validity
            if np.all(np.isfinite(c_res)):
                r = _compute_radius(c_res)
                if r > 0:
                    best_centers = c_res
    except TimeoutError:
        pass
    except Exception:
        pass
        
    return best_centers

def _local_search_polish(centers, time_limit):
    """
    A final polish using coordinate ascent or small perturbations to escape any numerical traps.
    """
    start_time = time.time()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Simple random jitter + local gradient step
    while time.time() < start_time + time_limit:
        # Try a small random perturbation
        noise_scale = 0.001
        perturbed = best_centers + np.random.randn(N, 2) * noise_scale
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.5, 0.5)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.5, 0.5)
        
        # Run a short gradient step
        # We can just run the gradient optimizer for a very short time
        # Or do a few manual gradient steps
        c = perturbed.copy()
        for _ in range(5):
            if time.time() > start_time + time_limit:
                break
            val, grad = _objective_and_grad(c, beta=200.0)
            # Simple gradient ascent step
            step_size = 0.001
            c -= grad * step_size
            c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
            c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
            
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_centers = c.copy()
            
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Generate multiple hex initializations
    for rows in range(12, 18):
        if time.time() > deadline - 60:
            break
        pts = _initial_hex_grid(rows)
        if pts is not None:
            r = _compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()
                
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_radius = _compute_radius(best_centers)
        
    # 2. Gradient-based optimization
    # Run on the best initial
    time_left = deadline - time.time() - 10
    if time_left > 5:
        optimized = _gradient_optimize(best_centers, time_limit=min(time_left, 80))
        r_opt = _compute_radius(optimized)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()
            
    # 3. Inflation phase / Polish
    # Increase beta and refine
    time_left = deadline - time.time() - 5
    if time_left > 5:
        # Run gradient optimization again with higher beta for more precision
        # We need to modify _gradient_optimize to accept beta or just use a wrapper
        # For now, let's do a manual high-beta polish
        
        c = best_centers.copy()
        beta = 500.0
        step_size = 0.0005
        last_r = _compute_radius(c)
        
        for _ in range(100):
            if time.time() > start_time + time_left:
                break
            val, grad = _objective_and_grad(c, beta=beta)
            c -= grad * step_size
            c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
            c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
            
            r = _compute_radius(c)
            if r > last_r:
                last_r = r
                best_centers = c.copy()
                best_radius = r
            else:
                # If no improvement, break or reduce step
                pass
                
    # 4. Final local search polish
    time_left = deadline - time.time() - 2
    if time_left > 2:
        polished = _local_search_polish(best_centers, time_limit=min(time_left, 10))
        r_pol = _compute_radius(polished)
        if r_pol > best_radius:
            best_radius = r_pol
            best_centers = polished.copy()
            
    return best_centers
# EVOLVE-BLOCK-END
