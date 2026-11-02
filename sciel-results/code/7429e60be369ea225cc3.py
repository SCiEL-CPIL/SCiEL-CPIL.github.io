import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes -soft_min.
    """
    centers = centers_flat.reshape(N, 3)
    
    # 1. Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    pair_vals = pair_dists[iu]
    
    all_vals = np.concatenate([wall_dists.flatten(), pair_vals])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # soft_min = -1/beta * ln(sum(exp(-beta * x)))
    # We want to maximize soft_min, so minimize -soft_min
    # -soft_min = 1/beta * ln(sum(exp(-beta * x)))
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    log_sum_exp = max_val + np.log(sum_exp)
    obj = log_sum_exp / beta
    
    return obj

def _soft_min_grad(centers_flat, beta, iu):
    """
    Gradient of the soft-min objective (negative soft min).
    """
    centers = centers_flat.reshape(N, 3)
    
    # 1. Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    pair_vals = pair_dists[iu]
    
    all_vals = np.concatenate([wall_dists.flatten(), pair_vals])
    
    # Calculate weights
    # We need d(-soft_min)/dx = - d(soft_min)/dx
    # soft_min = -1/beta * ln(sum(exp(-beta * x)))
    # d(soft_min)/dx_k = exp(-beta * x_k) / sum(exp(-beta * x_j))
    # So d(-soft_min)/dx_k = - exp(-beta * x_k) / sum(exp(-beta * x_j))
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_shifted = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_shifted)
    
    # weights_k = - exp(-beta * x_k) / sum_exp
    # Note: exp(-beta * x_k) = exp(neg_beta_x_k) = exp_shifted_k * exp(max_val)
    # So weights_k = - exp_shifted_k / sum_exp
    # The exp(max_val) cancels out in the ratio.
    
    weights = -exp_shifted / sum_exp
    
    grad_centers = np.zeros_like(centers)
    
    # Wall gradients
    # x_k = 0.5 - |c_ik|
    # d(x_k)/d(c_ik) = -sign(c_ik)
    wall_weights = weights[:N*3]
    # Vectorized wall gradient accumulation
    # grad_c[i,k] += wall_weights[i*3+k] * (-sign(c[i,k]))
    signs = np.sign(centers)
    grad_centers += wall_weights.reshape(N, 3) * (-signs)
    
    # Pair gradients
    # x_k = dist(i,j)/2
    # d(x_k)/d(c_i) = (c_i - c_j) / (2 * dist(i,j))
    # d(x_k)/d(c_j) = (c_j - c_i) / (2 * dist(i,j)) = -(c_i - c_j) / (2 * dist(i,j))
    
    pair_weights = weights[N*3:]
    i_idx, j_idx = iu
    
    # Vectorized pair gradient accumulation
    # For each pair (i, j) in iu:
    # diff = c_i - c_j
    # dist = ||diff||
    # factor = pair_weight / (2 * dist)
    # grad_c[i] += factor * diff
    # grad_c[j] -= factor * diff
    
    # This is tricky to fully vectorize without loops or fancy indexing that might be slow.
    # Let's use a loop for clarity and correctness, N=100 -> 4950 pairs.
    # 4950 iterations is very fast in Python if simple ops.
    
    for k in range(len(pair_weights)):
        i = i_idx[k]
        j = j_idx[k]
        d = centers[i] - centers[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            continue
        # factor = w * 1/(2*dist)
        factor = pair_weights[k] / (2.0 * dist)
        grad_centers[i] += factor * d
        grad_centers[j] -= factor * d
        
    return grad_centers.flatten()

def _build_grid_init():
    """
    Build a grid initialization.
    5x5x4 grid.
    """
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return centers

def _build_halton_init():
    """
    Build a Halton sequence initialization.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        points.append([x - 0.5, y - 0.5, z - 0.5])
    
    return np.array(points)

def _build_random_init(seed=42):
    """
    Build a random initialization.
    """
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _optimize_from_initial(initial_centers, deadline, iu):
    """
    Run the optimization loop starting from given initial centers.
    Returns best centers and best radius.
    """
    best_centers = initial_centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Continuation method for beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                jac=lambda x, b, iu: _soft_min_grad(x, b, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            # Check if result is better
            if res.fun < _soft_min_objective(x0, beta, iu):
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass
            
    return best_centers, best_radius

def _polish(centers, deadline, iu):
    """
    Final high-precision polish pass.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try a few very high beta values
    betas = [5000.0, 10000.0, 20000.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                jac=lambda x, b, iu: _soft_min_grad(x, b, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < _soft_min_objective(x0, beta, iu):
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Safety margin: 140s out of 150s limit (10s buffer)
    deadline = start_time + 140.0
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Try multiple initializations
    inits = []
    inits.append(_build_grid_init())
    inits.append(_build_halton_init())
    inits.append(_build_random_init(42))
    inits.append(_build_random_init(123))
    
    best_overall_centers = None
    best_overall_radius = -1.0
    
    # Time allocation
    # Total time available for optimization: 130s (leaving 10s for polish + safety)
    # 4 inits -> ~32s each
    
    num_inits = len(inits)
    opt_time_total = 130.0
    time_per_init = opt_time_total / num_inits
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0: # Leave 10s for polish
            break
            
        # Sub-deadline for this init
        # Ensure we don't exceed the global deadline - 10s
        sub_deadline = min(start_time + (i + 1) * time_per_init, deadline - 10.0)
        
        centers, radius = _optimize_from_initial(init, sub_deadline, iu)
        
        if radius > best_overall_radius:
            best_overall_radius = radius
            best_overall_centers = centers
    
    # Final polish
    if best_overall_centers is not None:
        best_overall_centers, best_overall_radius = _polish(best_overall_centers, deadline, iu)
    else:
        # Fallback if something went wrong
        best_overall_centers = _build_grid_init()
        best_overall_radius = _compute_radius(best_overall_centers)
        
    # Ensure centers are strictly inside to avoid boundary issues in evaluator
    best_overall_centers = np.clip(best_overall_centers, -0.499999, 0.499999)
        
    return best_overall_centers

# EVOLVE-BLOCK-END
