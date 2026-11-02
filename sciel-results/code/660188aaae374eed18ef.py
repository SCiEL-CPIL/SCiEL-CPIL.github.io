"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import math

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances (vectorized)
    # Calculate pairwise distance matrix
    # diff[i,j] = centres[i] - centres[j]
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # We only care about i < j, but min over all i!=j is same as min over i<j
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)


def _soft_min_objective_and_grad(flat, beta=50.0):
    """
    Compute soft-min objective and its gradient.
    Objective: minimize -softmin(values)
    Gradient: -d(softmin)/d(flat)
    """
    centres = flat.reshape(N, 3)
    
    # 1. Wall distances
    # v_wall[i, d] = 0.5 - |c[i, d]|
    wall_dists = 0.5 - np.abs(centres) # shape (N, 3)
    
    # 2. Pairwise distances / 2
    # We need gradients for these too.
    # diff[i,j] = c_i - c_j
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :] # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2) # (N, N)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0 # (N, N)
    
    # Combine all values into a single list for softmin calculation
    # To be efficient, we calculate the weighted sum directly.
    # weights w_k = exp(-beta * v_k) / sum(exp(-beta * v_k))
    # grad_obj = (1/beta) * sum( w_k * grad_v_k )
    
    # Flatten values for log-sum-exp
    wall_flat = wall_dists.ravel() # size 3N
    # For pairs, we only need i < j to avoid double counting in the sum?
    # Softmin is over the set of all constraints.
    # The set of constraints is:
    # 1. Each wall distance (3N constraints)
    # 2. Each pairwise distance (N(N-1)/2 constraints)
    
    # Let's construct the full vector of values V
    # And we need to map gradients back.
    
    # It is more efficient to compute the weights on the fly without storing the huge V vector explicitly if possible,
    # but here N=100, so 300 wall + 4950 pair = 5250 values. This is small.
    
    # Create indices for upper triangle pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_vals_flat = pair_vals[i_idx, j_idx]
    
    all_vals = np.concatenate([wall_flat, pair_vals_flat])
    
    # Calculate weights
    neg_beta_vals = -beta * all_vals
    max_neg_beta = np.max(neg_beta_vals)
    exp_vals = np.exp(neg_beta_vals - max_neg_beta)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp # w_k
    
    # Objective value: -softmin = (1/beta) * log(sum_exp) + max_neg_beta/beta
    # softmin = -1/beta * log(sum(exp(-beta v)))
    # -softmin = 1/beta * log(sum(exp(-beta v)))
    # Using stability: log(sum(exp(x))) = max(x) + log(sum(exp(x-max(x))))
    obj_val = (max_neg_beta + np.log(sum_exp)) / beta
    
    # Calculate Gradient
    # grad_obj = (1/beta) * sum_k w_k * grad_v_k
    # Note: obj_val is -softmin. We want to minimize obj_val.
    # So we return the gradient of obj_val w.r.t flat.
    
    grad_flat = np.zeros_like(flat)
    
    # Part 1: Wall contributions
    # v_wall[i, d] = 0.5 - |c[i, d]|
    # grad_v_wall[i, d] w.r.t c[i, d] is -sign(c[i, d])
    # The weights for walls are weights[:3*N]
    wall_weights = weights[:3*N].reshape(N, 3)
    
    # grad_v_wall is -sign(centres)
    grad_wall_contrib = -np.sign(centres) * wall_weights
    grad_flat += grad_wall_contrib.ravel()
    
    # Part 2: Pair contributions
    # v_pair[i, j] = 0.5 * ||c_i - c_j||
    # grad_v_pair w.r.t c_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # grad_v_pair w.r.t c_j = -0.5 * (c_i - c_j) / ||c_i - c_j||
    
    pair_weights = weights[3*N:]
    
    # Get the actual vectors and distances for the pairs
    ci = centres[i_idx]
    cj = centres[j_idx]
    vecs = ci - cj # (P, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) # (P, 1)
    
    # Avoid division by zero (should not happen if distinct, but safe)
    norms = np.where(norms == 0, 1.0, norms)
    
    unit_vecs = vecs / norms # (P, 3)
    
    # Gradient contribution for each pair (i, j) with weight w
    # d(v)/dc_i = 0.5 * unit_vec
    # d(v)/dc_j = -0.5 * unit_vec
    
    grad_pairs = np.zeros_like(centres)
    np.add.at(grad_pairs, i_idx, 0.5 * unit_vecs * pair_weights[:, np.newaxis])
    np.add.at(grad_pairs, j_idx, -0.5 * unit_vecs * pair_weights[:, np.newaxis])
    
    grad_flat += grad_pairs.ravel()
    
    # Multiply by 1/beta
    grad_flat /= beta
    
    return obj_val, grad_flat


def _make_bcc_init():
    """
    Create a body-centered cubic (BCC) inspired initialization.
    """
    centres = []
    m = 5  # 5x5x4 = 100
    spacing = 1.0 / m
    offset = spacing / 2.0
    
    for iz in range(4):
        for iy in range(5):
            for ix in range(5):
                x = -0.5 + offset + ix * spacing
                y = -0.5 + offset + iy * spacing
                z = -0.5 + (iz + 0.5) * (1.0/4)
                centres.append([x, y, z])
    
    centres = np.array(centres[:N])
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centres += rng.uniform(-0.01, 0.01, size=centres.shape)
    
    # Clip to stay inside cube
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _local_optimize(centres, beta=50.0, max_iter=100, time_limit=10.0):
    """
    Local optimization using L-BFGS-B on the soft-min objective with analytical gradient.
    """
    flat = centres.flatten()
    start_time = time.time()
    
    def objective(flat_x):
        val, _ = _soft_min_objective_and_grad(flat_x, beta)
        return val
    
    def gradient(flat_x):
        _, grad = _soft_min_objective_and_grad(flat_x, beta)
        return grad
    
    result = minimize(
        objective,
        flat,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-7}
    )
    
    return result.x.reshape(N, 3)


def _simulated_annealing(centres, n_iterations=2000, T_start=0.05, T_end=0.001, time_limit=30.0):
    """
    Simulated annealing to escape local minima.
    """
    rng = np.random.default_rng(123)
    current = centres.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    start_time = time.time()
    
    for i in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
            
        T = T_start * (T_end / T_start) ** (i / n_iterations)
        
        # Random perturbation
        idx = rng.integers(0, N)
        perturbation = rng.normal(0, T, size=3)
        new_centre = current[idx].copy()
        new_centre += perturbation
        
        # Keep inside cube
        new_centre = np.clip(new_centre, -0.499, 0.499)
        
        new_centres = current.copy()
        new_centres[idx] = new_centre
        
        new_r = _compute_radius(new_centres)
        
        # Accept if better or with probability exp((new_r - current_r)/T)
        if new_r >= current_r or rng.random() < math.exp((new_r - current_r) / max(T, 1e-10)):
            current = new_centres
            current_r = new_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with BCC-inspired grid
    centres = _make_bcc_init()
    current_r = _compute_radius(centres)
    best_centres = centres.copy()
    best_r = current_r
    
    # Phase 1: Simulated annealing to find a good basin
    if time.time() < deadline - 40.0:
        sa_centres = _simulated_annealing(centres, n_iterations=3000, T_start=0.03, T_end=0.0005, time_limit=30.0)
        sa_r = _compute_radius(sa_centres)
        if sa_r > best_r:
            best_centres = sa_centres
            best_r = sa_r
    
    # Phase 2: Local optimization with L-BFGS-B
    if time.time() < deadline - 20.0:
        opt_centres = _local_optimize(best_centres, beta=50.0, max_iter=100, time_limit=15.0)
        opt_r = _compute_radius(opt_centres)
        if opt_r > best_r:
            best_centres = opt_centres
            best_r = opt_r
    
    # Phase 3: Another round of SA + polish if time permits
    if time.time() < deadline - 10.0:
        sa2_centres = _simulated_annealing(best_centres, n_iterations=2000, T_start=0.01, T_end=0.0001, time_limit=10.0)
        sa2_r = _compute_radius(sa2_centres)
        if sa2_r > best_r:
            best_centres = sa2_centres
            best_r = sa2_r
    
    # Phase 4: Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        opt2_centres = _local_optimize(best_centres, beta=100.0, max_iter=50, time_limit=5.0)
        opt2_r = _compute_radius(opt2_centres)
        if opt2_r > best_r:
            best_centres = opt2_centres
            best_r = opt2_r
    
    # Ensure all centres are strictly inside the cube
    best_centres = np.clip(best_centres, -0.499999, 0.499999)
    
    return best_centres
# EVOLVE-BLOCK-END
