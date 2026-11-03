"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(c):
    """Compute the packing radius for centres c of shape (N, 3)."""
    # Wall distances
    wall_dists = 0.5 - np.abs(c)  # (N, 3)
    wall_min = wall_dists.min()
    
    # Pairwise distances
    # Using broadcasting for 100x100 is fine
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_min = dists.min()
    
    return min(pair_min / 2.0, wall_min)

def _objective(c):
    """Return negative score to maximize."""
    c = c.reshape(N, 3)
    return -_score(c)

def _energy_and_grad(c):
    """
    Compute an energy function and its gradient for optimization.
    We want to maximize the minimum distance.
    This is a non-smooth problem. We can use a soft-min approximation.
    E = -softmin(dists, wall_dists)
    Using log-sum-exp for softmin is tricky for minimization.
    Instead, let's try to minimize the sum of penalties for being closer than target r.
    Or better: maximize r.
    Let's use the auxiliary variable t.
    Maximize t subject to d_ij >= 2t and wall_i >= t.
    This is a linear programming problem if we fix the graph, but here the graph changes.
    
    Alternative: Gradient ascent on the soft-min.
    softmin(x) = -1/k * log(sum(exp(-k*x)))
    grad softmin(x) = exp(-k*x) / sum(exp(-k*x))
    
    Let's use this.
    """
    c = c.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)  # (N, 3)
    flat_walls = wall_dists.flatten()
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = (diff ** 2).sum(axis=2)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # We care about the minimum distance.
    # Let's define a set of "critical" distances.
    # To make it smooth, we use a soft-min over all pair distances and wall distances.
    # The value we want to maximize is r.
    # Let's define energy E = -softmin(all_distances)
    # all_distances = [d_ij/2, wall_ik]
    
    # To save memory and time, we can approximate the softmin by just taking the min
    # and using subgradients, but that's non-smooth.
    # Let's stick to the log-sum-exp softmin.
    
    # The number of terms is N*(N-1)/2 + N*3 = 4950 + 300 = 5250.
    # This is manageable.
    
    pair_dists = dists / 2.0
    # Remove diagonal (infs)
    triu_indices = np.triu_indices(N, k=1)
    pair_flat = pair_dists[triu_indices]
    
    all_dists = np.concatenate([pair_flat, flat_walls])
    
    # Softmin with parameter k
    # k too large -> sharp min (discontinuous grad)
    # k too small -> smooth but biased
    k = 50.0
    
    # To avoid overflow in exp(-k*x), we shift by min(all_dists)
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    weights = np.exp(-k * shifted)
    Z = np.sum(weights)
    softmin_val = min_d - np.log(Z) / k
    
    # Gradient of softmin wrt all_dists
    grad_softmin_all = weights / Z
    
    # Now map back to c
    grad_c = np.zeros_like(c)
    
    # Wall part
    grad_walls = grad_softmin_all[-N*3:]
    grad_walls = grad_walls.reshape(N, 3)
    # wall_dist = 0.5 - |c|
    # d(wall)/d(c) = -sign(c)
    # So grad_c += grad_walls * (-sign(c))
    signs = np.sign(c)
    # Handle c=0 (gradient undefined, set to 0)
    signs[signs == 0] = 1.0 
    grad_c += grad_walls * (-signs)
    
    # Pair part
    grad_pairs = grad_softmin_all[:-N*3]
    
    # Reconstruct gradient for pairs
    # d_ij = |c_i - c_j|
    # d(d_ij)/d(c_i) = (c_i - c_j) / |c_i - c_j|
    # d(d_ij)/d(c_j) = (c_j - c_i) / |c_i - c_j|
    
    # This loop is slow in Python. Let's vectorize or use sparse.
    # With 4950 pairs, a loop might be okay if not done too often, but gradient descent needs many steps.
    # Let's try to vectorize.
    
    # Create matrix G of shape (N, N) for pair weights? No, only upper triangle.
    # Let's use the diff and dists arrays.
    # We have grad_pairs corresponding to triu_indices.
    
    # Initialize pair grad matrix
    grad_pair_mat = np.zeros((N, N))
    # Map flat indices to 2D indices
    # triu_indices gives (i, j) for i < j
    i_idx, j_idx = triu_indices
    # grad_pairs is already in the order of triu_indices
    # We need to add to grad_c[i] and grad_c[j]
    
    # Vectorized update is tricky because of the division by dist.
    # Let's use a loop for clarity first, then optimize if needed.
    # 4950 iterations per gradient call. If we do 1000 gradient calls, that's ~5M ops.
    # Python loops are slow. ~5M simple ops might take 0.5s - 1s.
    # With 120s limit, we can afford ~100-200 gradient steps if each is 1s.
    # Let's try to speed it up.
    
    # Vectorized approach:
    # We have vectors v_ij = c_i - c_j
    # dist_ij = |v_ij|
    # grad_c_i += w_ij * (v_ij / dist_ij)
    # grad_c_j += w_ij * (-v_ij / dist_ij)
    
    # v_ij for all i<j is stored in diff[i_idx, j_idx] ? No, diff is (N, N, 3).
    # diff[i, j] = c_i - c_j
    # So v = diff[i_idx, j_idx] has shape (4950, 3)
    v = diff[i_idx, j_idx]
    d = dists[i_idx, j_idx]  # shape (4950,)
    
    # unit vectors
    # d is > 0 due to max(1e-12)
    unit_v = v / d[:, np.newaxis]
    
    # contributions
    contrib = grad_pairs[:, np.newaxis] * unit_v  # shape (4950, 3)
    
    # Accumulate to grad_c
    # grad_c[i] += contrib[k]
    # grad_c[j] -= contrib[k]
    
    # np.add.at is slow but correct.
    # np.bincount is faster for 1D, but we have 3D vectors.
    # We can do 3 bincounts, one per coordinate.
    
    grad_c += np.bincount(i_idx, weights=contrib[:, 0], minlength=N)
    grad_c += np.bincount(j_idx, weights=-contrib[:, 0], minlength=N)
    
    grad_c[:, 1] += np.bincount(i_idx, weights=contrib[:, 1], minlength=N)
    grad_c[:, 1] += np.bincount(j_idx, weights=-contrib[:, 1], minlength=N)
    
    grad_c[:, 2] += np.bincount(i_idx, weights=contrib[:, 2], minlength=N)
    grad_c[:, 2] += np.bincount(j_idx, weights=-contrib[:, 2], minlength=N)
    
    # Energy is -softmin
    energy = -softmin_val
    # Gradient of energy is -gradient of softmin
    # Wait, we want to MINIMIZE energy.
    # objective = -score.
    # score = softmin.
    # energy = -softmin.
    # grad energy = -grad softmin.
    # We computed grad_softmin_all (grad of softmin wrt dists).
    # So grad_c computed above is grad(softmin) wrt c.
    # So grad(energy) = -grad_c.
    
    return energy, -grad_c.reshape(-1)

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Initialize with cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    # Shuffle to break symmetry? Maybe.
    # np.random.shuffle(c0)
    
    best_c = c0.copy()
    best_score = _score(best_c)
    
    # Simple Gradient Descent / Ascent
    # Since we are maximizing softmin, we do ascent.
    # Or minimize -softmin.
    
    # Let's try L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use multiple restarts or just one long run?
    # Given time limit, one long run with adaptive step or L-BFGS is safer.
    
    # L-BFGS-B requires smooth objective. Our softmin is smooth.
    
    # However, L-BFGS-B might struggle with the non-convexity.
    # Let's try a few iterations.
    
    # To improve, we can add a "jitter" and run multiple optimizations.
    
    num_restarts = 5
    time_per_restart = deadline / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            c_start = c0.copy()
        else:
            # Add random jitter to best known or random
            if r == 1:
                c_start = best_c + np.random.normal(0, 0.05, (N, 3))
            else:
                # Random restart
                c_start = np.random.uniform(-0.5, 0.5, (N, 3))
                # Snap to grid? No.
                
        c_start = np.clip(c_start, -0.5, 0.5)
        
        # Optimize
        res = minimize(
            _objective, 
            c_start.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        c_opt = res.x.reshape(N, 3)
        score_opt = _score(c_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_c = c_opt.copy()
            
        # Check time
        if time.time() > deadline:
            break
            
        # Refine with local polish?
        # Try a few more steps with smaller step size or higher precision?
        
    return best_c

# EVOLVE-BLOCK-START
# (The code above is the implementation. The original code had this block structure.)
# I will place the construct function and helpers inside or outside?
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program had:
# N = 100
# def construct(): ...
#
# I have added helper functions. This should be fine as long as construct is there.

# EVOLVE-BLOCK-END
