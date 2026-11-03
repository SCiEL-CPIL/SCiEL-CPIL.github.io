"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 150.0  # Kill limit

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances using pdist for exact min
    min_pair = np.min(pdist(centers))
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    # Try different row counts to find the best hex grid fit
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        # Estimate spacing to fit in the square
        # Width constraint: n_cols * s <= 1 => s <= 1/n_cols
        # Height constraint: (n_rows - 1) * s * sqrt(3)/2 + s <= 1 => s * ((n_rows-1)*sqrt(3)/2 + 1) <= 1
        # So s <= 1 / ((n_rows-1)*sqrt(3)/2 + 1)
        
        s_x = 1.0 / n_cols
        s_y_factor = (n_rows - 1) * np.sqrt(3) / 2 + 1
        s_y = 1.0 / s_y_factor
        s = min(s_x, s_y)
        
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + s/2 + j * s + (i % 2) * s / 2
                y = -0.5 + s/2 + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        # Ensure centers are within bounds, centering if necessary
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _soft_min(x, beta=50.0):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    x_min = np.min(x)
    shifted = -(x - x_min) * beta
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return x_min - log_sum / beta

def _objective_and_grad(x_flat, beta=50.0):
    """
    Objective: -soft_min(constraints)
    Constraints: 
      - Wall distances: 0.5 - |x|, 0.5 - |y|
      - Pair distances / 2
    We want to maximize the min constraint, so minimize -soft_min(constraints).
    """
    x = x_flat.reshape(-1, 2)
    
    # Wall constraints
    wall_dists = np.min(np.minimum(0.5 - x, x + 0.5), axis=1)
    
    # Pair constraints
    # Using cKDTree is faster for large N, but for N=200 pdist is fine and simple.
    # However, for gradient we need the specific pairs that are close.
    # The soft-min gradient is a weighted average of the gradients of all constraints.
    # Weights are proportional to exp(-beta * (d_i - d_min)).
    
    d_pairs = pdist(x)
    
    # Combine all constraints into a single array for soft-min calculation
    # This is memory efficient enough for N=200 (200 walls + 19900 pairs)
    all_dists = np.concatenate([wall_dists, d_pairs / 2.0])
    
    # Soft min
    d_min = np.min(all_dists)
    shifted = -(all_dists - d_min) * beta
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_min_val = d_min - np.log(sum_exp) / beta
    
    # Objective is -soft_min
    obj = -soft_min_val
    
    # Gradient calculation
    # grad(soft_min) = sum_i w_i * grad(d_i)
    # where w_i = exp(-beta*(d_i - d_min)) / sum_exp = exp_shifted[i] / sum_exp
    
    weights = exp_shifted / sum_exp
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # d_wall_i = min(0.5 - x_i, x_i + 0.5)
    # grad wrt x: -1 if x_i > 0, 1 if x_i < 0 (roughly, depends on which wall is active)
    # More precisely:
    # d_right = 0.5 - x, d_left = x + 0.5
    # d_wall = min(d_right, d_left)
    # If d_right < d_left (x > 0), grad_x = -1, else grad_x = 1.
    # If d_top < d_bottom (y > 0), grad_y = -1, else grad_y = 1.
    
    # Actually, d_wall_i = min(0.5 - x_i, 0.5 - y_i, -0.5 - x_i, -0.5 - y_i) ? No.
    # d_wall_i = min(0.5 - x_i, x_i + 0.5, 0.5 - y_i, y_i + 0.5)
    # Let's compute gradients for each component.
    
    # Wall 0: Right (0.5 - x). Active if x > 0. Grad: [-1, 0]
    # Wall 1: Left (x + 0.5). Active if x < 0. Grad: [1, 0]
    # Wall 2: Top (0.5 - y). Active if y > 0. Grad: [0, -1]
    # Wall 3: Bottom (y + 0.5). Active if y < 0. Grad: [0, 1]
    
    # The wall_dists array computed above is the min of these 4 for each point.
    # So the gradient for point i is the gradient of the active wall, weighted by w_i.
    
    # We need to know which wall is active for each point.
    # d_right = 0.5 - x[:, 0]
    # d_left = 0.5 + x[:, 0]
    # d_top = 0.5 - x[:, 1]
    # d_bottom = 0.5 + x[:, 1]
    
    d_right = 0.5 - x[:, 0]
    d_left = 0.5 + x[:, 0]
    d_top = 0.5 - x[:, 1]
    d_bottom = 0.5 + x[:, 1]
    
    # For each point, find the min of these 4.
    # We can do this efficiently.
    # But we already have wall_dists = min(d_right, d_left, d_top, d_bottom).
    # Let's just check which one is the min.
    
    # To save time, we can approximate or just use the full check.
    # N=200, so 200 checks is trivial.
    
    for i in range(N):
        w = weights[i]
        if w < 1e-10: continue
        
        dr = d_right[i]
        dl = d_left[i]
        dt = d_top[i]
        db = d_bottom[i]
        
        min_val = np.min([dr, dl, dt, db])
        
        # It's possible multiple are equal (e.g., corner). We sum gradients.
        if dr == min_val:
            grad[i, 0] -= w
        if dl == min_val:
            grad[i, 0] += w
        if dt == min_val:
            grad[i, 1] -= w
        if db == min_val:
            grad[i, 1] += w
    
    # Pair gradients
    # d_pair_ij = ||x_i - x_j|| / 2
    # grad_i = (x_i - x_j) / (2 * ||x_i - x_j||)
    # grad_j = (x_j - x_i) / (2 * ||x_i - x_j||)
    
    # We need to sum over all pairs (i, j) with weight w_pair.
    # d_pairs is from pdist, which returns condensed form.
    # Index mapping for pdist (N points):
    # (0,1), (0,2), ..., (0,N-1), (1,2), ..., (1,N-1), ...
    
    # It is faster to compute gradients using the full distance matrix or by iterating.
    # Given N=200, iterating over 19900 pairs in Python might be slow (0.02s per iteration?).
    # 500 iterations * 0.02s = 10s. Acceptable.
    # But let's try to vectorize or use a smarter approach.
    
    # Alternative: Use the fact that only close pairs have significant weight.
    # But for robustness, let's do the full calculation.
    
    # Let's reconstruct the distance matrix to get indices? No, pdist doesn't return indices.
    # We can use cKDTree to find close pairs and only compute gradients for those?
    # But soft-min weights all pairs, just very small for far ones.
    # Threshold: if weight < 1e-10, ignore.
    
    # Let's use a loop but optimized.
    # Actually, we can compute the gradient of the sum of exp(-beta * d/2) terms.
    
    # Let's stick to the loop for correctness first.
    # To speed it up, we can precompute the weights for pairs.
    # weights_pairs = weights[N:]
    
    # We need to map the condensed index to (i, j).
    # This is tedious.
    
    # Better approach: Compute the gradient of the pair term directly using vectorized ops on the full matrix.
    # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :] # (N, N, 2)
    # dist = np.linalg.norm(diff, axis=2) # (N, N)
    # np.fill_diagonal(dist, np.inf)
    # pair_dists = dist / 2.0
    
    # We already have d_pairs from pdist.
    # Let's just use the full matrix approach for gradient, it's O(N^2) memory which is fine for N=200.
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    pair_dists_mat = dist_mat / 2.0
    
    # We need weights corresponding to this matrix.
    # The soft_min was computed on a concatenated array.
    # The order in all_dists was: wall_dists (N), then d_pairs (pdist order).
    # The matrix pair_dists_mat has all pairs twice (i,j) and (j,i).
    # pdist has them once.
    # So weights for matrix entries (i, j) where i < j is weights[N + idx].
    # For (j, i) it's the same weight.
    
    # Let's create a weight matrix.
    weight_mat = np.zeros((N, N))
    # Fill upper triangle with weights from pdist
    # We need the index in the condensed vector for (i, j).
    # It's easier to just compute the soft min and gradient on the full symmetric matrix?
    # No, the objective function must be consistent.
    
    # Let's re-calculate soft_min and gradient using the full symmetric matrix for pairs.
    # This doubles the number of terms but simplifies indexing.
    # The value of soft_min will be slightly different (log term changes), but the gradient direction is the same.
    # Actually, the objective value will change, which might confuse the optimizer if not careful.
    # But since we are minimizing -soft_min, and the soft_min is just a proxy, it's fine as long as it's consistent within the optimization.
    
    # Let's restart the objective calculation with the full matrix to simplify gradient.
    
    # Wall dists
    wall_dists = np.min(np.minimum(0.5 - x, x + 0.5), axis=1)
    
    # Pair dists (full matrix)
    pair_dists = dist_mat / 2.0
    
    # Combine
    # We can treat walls and pairs separately or combine.
    # Combining: all_dists = np.concatenate([wall_dists, pair_dists[np.triu_indices(N, 1)]])
    # This is what we did before.
    
    # Let's stick to the previous method but optimize the pair gradient loop.
    
    # Actually, let's just use the full matrix for the soft min calculation too.
    # It's cleaner.
    # all_dists = np.concatenate([wall_dists, pair_dists[np.triu_indices(N, 1)]])
    # This is the same as before.
    
    # The gradient of the pair term:
    # sum_{i<j} w_{ij} * grad_{ij}
    # grad_{ij} wrt x_i: (x_i - x_j) / (2 * dist_ij)
    # grad_{ij} wrt x_j: (x_j - x_i) / (2 * dist_ij)
    
    # We can vectorize this.
    # Let W be the weight matrix (N, N), symmetric, zero diagonal.
    # W[i, j] = w_{ij} for i < j, and W[j, i] = w_{ij}.
    
    # grad_pairs = sum_j W[i, j] * (x_i - x_j) / (2 * dist_ij)
    
    # Let's construct W.
    # We have weights_pairs = weights[N:] which are in pdist order.
    # We need to map them to the matrix.
    
    # This mapping is the bottleneck.
    # Let's just use the full matrix for the soft min as well.
    # all_dists_full = np.concatenate([wall_dists, pair_dists[np.triu_indices(N, 1)]])
    # This is what we have.
    
    # Okay, let's just do the loop. N=200 is small.
    
    # Pre-compute unit vectors for all pairs?
    # unit_vecs = diff / dist_mat[:, :, np.newaxis]
    # grad_pairs[i] = sum_j W[i, j] * unit_vecs[i, j] / 2
    
    # Let's build W from the condensed weights.
    # This is still O(N^2) to fill.
    
    # Let's try a different tactic:
    # Compute the gradient using the full symmetric matrix directly in the soft-min.
    # This means the objective function changes slightly (factor of 2 in the log term for pairs).
    # But it allows full vectorization.
    
    # Let's redefine the objective to use the full symmetric matrix for pairs.
    # This is a valid smooth approximation of the min distance.
    
    # New Objective:
    # d_pairs_full = dist_mat / 2.0 (with inf on diagonal)
    # all_dists = np.concatenate([wall_dists, d_pairs_full[np.triu_indices(N, 1)]])
    # This is exactly what we had.
    
    # Let's just accept the O(N^2) loop for now.
    
    # Optimization: Only compute for pairs where weight > threshold.
    # But we need the weights first.
    
    # Let's just do it.
    
    # To speed up the loop, we can use numba if available, but let's stick to standard Python/NumPy.
    
    # Actually, we can compute the pair gradient vectorized if we can get the weight matrix.
    # Let's create the weight matrix W.
    
    # The condensed index for (i, j) with i < j is:
    # idx = i * (N - 1) - i * (i - 1) / 2 + (j - i - 1) ? No.
    # Standard pdist index:
    # (0,1)=0, (0,2)=1, ..., (0,N-1)=N-2
    # (1,2)=N-1, (1,3)=N, ..., (1,N-1)=2N-4
    # ...
    
    # It's easier to just use a helper or loop.
    
    # Let's try to vectorize the gradient accumulation.
    # grad_pairs = np.zeros_like(x)
    # for i in range(N):
    #   for j in range(i+1, N):
    #     w = weight_mat[i, j] # if we had it
    #     ...
    
    # Let's just use the full matrix approach for the soft-min calculation to avoid the index mapping.
    # We will compute soft_min on [wall_dists, d_pairs_full_upper_triangle].
    # And for the gradient, we will use the full matrix logic but only sum over i < j.
    
    # Wait, if I change the objective, I need to change the gradient to match.
    
    # Let's stick to the current objective (pdist based) and just do the loop.
    # It's the safest.
    
    # Loop optimization:
    # Pre-compute the weights for pairs.
    w_pairs = weights[N:]
    
    # We need to iterate over pairs.
    # Let's use a list of (i, j, w) for active pairs?
    # Or just nested loops.
    
    # To make it faster, we can skip pairs with small weights.
    # But we don't know the weights without the loop? No, we have w_pairs.
    # We just need to map w_pairs[k] to (i, j).
    
    # Let's pre-compute the index mapping.
    # This is O(N^2) memory, which is fine.
    
    idx_i = np.zeros(len(w_pairs), dtype=int)
    idx_j = np.zeros(len(w_pairs), dtype=int)
    
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_i[k] = i
            idx_j[k] = j
            k += 1
    
    # Now we can vectorize the gradient update.
    # diff_vec = x[idx_i] - x[idx_j] # (M, 2)
    # dist_vec = d_pairs # (M,)
    # unit_vec = diff_vec / dist_vec[:, np.newaxis] # (M, 2)
    # grad_vec = unit_vec * (w_pairs / 2)[:, np.newaxis] # (M, 2)
    
    # Then scatter add to grad.
    # np.add.at(grad, idx_i, grad_vec)
    # np.add.at(grad, idx_j, -grad_vec)
    
    diff_vec = x[idx_i] - x[idx_j]
    dist_vec = d_pairs
    # Avoid division by zero
    mask = dist_vec > 1e-10
    if np.any(mask):
        unit_vec = diff_vec[mask] / dist_vec[mask, np.newaxis]
        grad_vec = unit_vec * (w_pairs[mask] / 2.0)[:, np.newaxis]
        
        np.add.at(grad, idx_i[mask], grad_vec)
        np.add.at(grad, idx_j[mask], -grad_vec)
    
    # The objective is -soft_min
    return obj, -grad.flatten() # Negative gradient for minimization of -soft_min

def _gradient_refine(centers, max_iter=1000, lr=0.01):
    """
    Gradient-based refinement using L-BFGS-B or simple gradient descent.
    """
    centers = centers.copy()
    best_t = _compute_radius(centers)
    best_centers = centers.copy()
    
    x_flat = centers.flatten()
    
    # Use L-BFGS-B for better convergence
    # But we need to handle the time limit.
    
    # Simple gradient descent with line search might be more robust to time limits.
    # Let's use a simple adaptive gradient descent.
    
    current_lr = lr
    no_improve_count = 0
    
    for it in range(max_iter):
        if _time_left() < 15.0:
            break
        
        obj, grad = _objective_and_grad(x_flat)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        new_x = x_flat + current_lr * grad
        new_centers = new_x.reshape(-1, 2)
        
        # Clip to bounds
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(new_centers)
        
        if new_r > best_t + 1e-12:
            best_t = new_r
            best_centers = new_centers.copy()
            x_flat = new_x
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If no improvement, reduce learning rate
            if no_improve_count > 10:
                current_lr *= 0.9
                no_improve_count = 0
                if current_lr < 1e-6:
                    break
        
        # Optional: Line search
        # If the step was too big, reduce it.
    
    return best_centers, best_t

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Generate initial configuration
    base_centers = _generate_hex_grid()
    
    # Create variants
    centers_list = [base_centers.copy()]
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = base_centers.copy()
        noise = rng.normal(0, 0.01, size=perturbed.shape)
        perturbed += noise
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.45, 0.45)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.45, 0.45)
        centers_list.append(perturbed)
    
    best_centers = None
    best_r = 0.0
    
    # Optimize each start
    for centers in centers_list:
        if _time_left() < 30.0:
            break
        
        centers_opt, r = _gradient_refine(centers, max_iter=500, lr=0.01)
        
        if r > best_r:
            best_r = r
            best_centers = centers_opt.copy()
    
    if best_centers is None:
        best_centers = base_centers
        best_r = _compute_radius(base_centers)
    
    # Final polish
    if _time_left() > 20.0:
        # Try a few more iterations on the best solution
        centers_polished, r_polished = _gradient_refine(best_centers, max_iter=500, lr=0.005)
        if r_polished > best_r:
            best_centers = centers_polished
            best_r = r_polished
    
    return best_centers

# EVOLVE-BLOCK-END
