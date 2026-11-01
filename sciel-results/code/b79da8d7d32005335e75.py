import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """
    Compute the exact maximum radius for the given centers.
    """
    centers = np.asarray(centers)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return -np.inf
    
    # Check if centers are inside the cube
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    # Distance to each face
    # Face x=0.5: 0.5 - c_x
    # Face x=-0.5: 0.5 + c_x
    # Min distance to any face for point i is 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    # Wait, the constraint is that the sphere must be inside.
    # r <= 0.5 - |c_ik| for all k.
    # So r <= min_k (0.5 - |c_ik|).
    abs_c = np.abs(centers)
    boundary_dists = 0.5 - abs_c
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _smooth_objective_and_grad(centers, beta):
    """
    Compute the smooth surrogate objective (negative soft-min) and its gradient.
    We want to maximize the min distance, so we minimize the negative soft-min.
    """
    n = len(centers)
    # Get all pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # 1. Pairwise distances
    diff = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diff, axis=1)
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-10, dists, 1e-10)
    
    # Values v_ij = dists / 2
    v_pairs = dists / 2.0
    
    # 2. Boundary distances
    # For each point i and dim k, v_ik = 0.5 - abs(c_ik)
    abs_c = np.abs(centers)
    v_bonds = 0.5 - abs_c # Shape (N, 3)
    
    # Combine all values into a single array for the soft-min
    # v_pairs shape: (M_pairs,)
    # v_bonds shape: (N*3,)
    all_v = np.concatenate([v_pairs, v_bonds.flatten()])
    
    # Soft-min calculation
    # S = -1/beta * ln(sum(exp(-beta * v)))
    # To avoid overflow, use log-sum-exp trick
    max_v = np.max(all_v)
    # Note: exp(-beta * (v - max_v)) is stable
    exp_terms = np.exp(-beta * (all_v - max_v))
    log_sum = np.log(np.sum(exp_terms))
    obj = -1.0 / beta * (log_sum - beta * max_v) # This is the soft-min value
    # We want to MINIMIZE -obj. So objective to return is -obj?
    # Wait. Maximize R. R approx softmin(v).
    # Minimize -softmin(v).
    # So objective function for optimizer is -obj.
    
    # Gradient calculation
    # p_m = exp(-beta * (v_m - max_v)) / sum(exp(...))
    probs = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(centers)
    
    # Pair gradients
    # v_ij = dist/2
    # d v_ij / d c_i = 0.5 * (c_i - c_j) / dist
    # d v_ij / d c_j = 0.5 * (c_j - c_i) / dist
    # Contribution to grad of -S:
    # d(-S)/d c = - sum p_m * d v_m / d c
    # So for pair (i,j) with prob p:
    # grad[i] -= p * 0.5 * (c_i - c_j) / dist
    # grad[j] -= p * 0.5 * (c_j - c_i) / dist
    
    unit_vecs = diff / dists_safe[:, np.newaxis]
    weights = probs[:len(v_pairs)] * 0.5
    
    # Accumulate for i
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = weights[k]
        u = unit_vecs[k]
        grad[i] -= w * u
        grad[j] -= w * (-u) # which is + w * u
    
    # Boundary gradients
    # v_ik = 0.5 - |c_ik|
    # If c_ik > 0, d v_ik / d c_ik = -1
    # If c_ik < 0, d v_ik / d c_ik = 1
    # So d v_ik / d c_ik = -sign(c_ik)
    # Contribution to grad:
    # grad[i] -= p_ik * (-sign(c_ik)) e_k = p_ik * sign(c_ik) e_k
    
    probs_bonds = probs[len(v_pairs):].reshape(N, 3)
    signs = np.sign(centers)
    # grad += probs_bonds * signs
    grad += probs_bonds * signs
    
    return -obj, grad

def _local_refine(centers, beta=50.0, max_iter=200):
    """
    Refine centers using L-BFGS-B on the smooth objective.
    """
    centers = centers.copy()
    centers = np.clip(centers, -0.5, 0.5)
    
    def fun(x):
        c = np.clip(x, -0.5, 0.5)
        val, g = _smooth_objective_and_grad(c, beta)
        return val, g
    
    # Initial call
    val, g = fun(centers)
    
    res = minimize(fun, centers, method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Clip result to ensure validity
    res.x = np.clip(res.x, -0.5, 0.5)
    
    return res.x

def _generate_fcc_init():
    """
    Generate an initial configuration based on FCC lattice.
    """
    # FCC basis in a unit cell
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # Generate points in a 4x4x4 grid of FCC cells
    # This gives 4*4*4*4 = 256 points. We need 50.
    # We will pick the 50 closest to the center of the cube [-0.5, 0.5]^3?
    # Or just fill from center outwards.
    
    points = []
    # Center of cube is (0,0,0)
    # FCC cell size? 
    # If we use cell size 0.5, the basis points are within [0, 0.5].
    # Let's generate a supercell and shift to center.
    
    # Let's try a simpler approach: Random points with a minimum distance constraint (Poisson Disk Sampling-like)
    # Or just use the grid jittered.
    
    # Let's stick to a jittered grid for simplicity and robustness, 
    # but add an FCC-like structure if possible.
    
    # For now, return a jittered 4x4x4 grid subset
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_centers = None
    best_r = 0.0
    
    # Initial configs
    # 1. Jittered Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N].copy()
    
    # 2. Random
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))
    
    # 3. FCC-like (try to construct)
    # Let's just use multiple random restarts with different seeds and the grid.
    
    candidates = [
        init_grid + np.random.normal(0, 0.01, init_grid.shape),
        init_grid + np.random.normal(0, 0.05, init_grid.shape),
        init_random,
        np.random.uniform(-0.45, 0.45, (N, 3))
    ]
    
    # Clip all to valid range
    for i in range(len(candidates)):
        candidates[i] = np.clip(candidates[i], -0.5, 0.5)
        
    for i, init_c in enumerate(candidates):
        if time.time() > deadline - 20: # Need at least 20s left
            break
            
        # Try different betas (annealing)
        # Start with low beta (smooth) then high beta (sharp)
        betas = [10.0, 50.0, 200.0]
        
        current_c = init_c.copy()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            # Refine
            refined_c = _local_refine(current_c, beta=beta, max_iter=100)
            
            # Evaluate exact score
            r = _score(refined_c)
            
            if r > best_r:
                best_r = r
                best_centers = refined_c
            
            # Use the refined center as the start for the next (sharper) beta
            current_c = refined_c
            
    # Final polish on the best center found
    if best_centers is not None and time.time() < deadline - 2:
        # One more refinement with high beta
        final_c = _local_refine(best_centers, beta=500.0, max_iter=50)
        final_r = _score(final_c)
        if final_r > best_r:
            best_centers = final_c
            best_r = final_r

    if best_centers is None:
        return init_grid
        
    return best_centers
