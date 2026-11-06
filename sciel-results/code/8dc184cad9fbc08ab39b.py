"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE_SECONDS = 140.0  # safety margin before the 150s kill

def _exact_radius(x):
    """
    x: (N, 3) array of centres.
    Returns the minimum of:
      - half the minimum pairwise distance
      - the minimum distance from any centre to the cube boundary
    """
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    # Distance to nearest wall for each sphere
    # min(0.5 - x, 0.5 + x) for each coordinate, then min over coords
    wall_dists = np.minimum(0.5 - x, 0.5 + x)
    min_wall = np.min(wall_dists)

    return min(min_pair, min_wall)

def _soft_min_radius_and_grad(x, lam):
    """
    Surrogate objective: -soft_min_radius.
    soft_min uses log-sum-exp for the minimum of pairwise distances and wall distances.
    We maximize the radius, so we minimize the negative.
    
    x: (N, 3)
    lam: sharpness parameter (higher = closer to true min)
    """
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - x, 0.5 + x)
    flat_wall = wall_dists.flatten()
    
    # Combine all distances into one array
    # We take the min over all pairwise (i<j) and all wall distances (N*3 of them)
    # To avoid double counting, we take upper triangle for pairwise
    iu = np.triu_indices(N, k=1)
    pair_flat = dists[iu]
    
    all_dists = np.concatenate([pair_flat, flat_wall])
    
    # soft min = -1/lam * log(sum(exp(-lam * d)))
    # To avoid overflow, subtract min
    min_d = np.min(all_dists)
    scaled = -lam * (all_dists - min_d)
    # Clip for numerical stability
    scaled = np.clip(scaled, -700, 700)
    logsumexp = np.log(np.sum(np.exp(scaled)))
    soft_min = min_d - logsumexp / lam
    
    # Gradient w.r.t. x
    # d(soft_min)/d(d_k) = exp(-lam * (d_k - min_d)) / sum(exp(-lam * (d_j - min_d)))
    # which is exp(scaled_k) / sum(exp(scaled))
    weights = np.exp(scaled)
    weights /= np.sum(weights)
    
    # Gradient of soft_min w.r.t. x = sum_k weights_k * d(d_k)/dx
    grad = np.zeros((N, 3))
    
    # Pairwise contributions
    # For each pair (i, j), d_ij = ||x_i - x_j||
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # d(d_ij)/dx_j = (x_j - x_i) / d_ij
    pair_grad = np.zeros((N, 3))
    # Reuse diff
    # diff[i, j] = x_i - x_j
    # dists[i, j] = ||x_i - x_j||
    # We need to accumulate weights
    # Let's build a weighted sum
    # pair_flat is indexed by iu[0], iu[1]
    # We need to map back
    # For each pair (i, j) in iu, weight w_ij
    # grad_x_i += w_ij * (x_i - x_j) / d_ij
    # grad_x_j += w_ij * (x_j - x_i) / d_ij
    
    # Vectorized approach:
    # Create a (N, N) weight matrix W where W[i, j] = weight of pair (i, j)
    W = np.zeros((N, N))
    W[iu] = weights[:len(iu[0])]
    W = W + W.T  # Symmetric
    np.fill_diagonal(W, 0.0)
    
    # grad = W * diff / dists  (element-wise, but careful with zeros)
    # diff[i, j] = x_i - x_j
    # We want: sum_j W[i, j] * (x_i - x_j) / dists[i, j]
    # Let's compute: (W / dists) * diff, then sum over j
    # Avoid division by zero on diagonal (dists is inf, W is 0)
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    factor = W / safe_dists
    # factor[i, j] = W[i, j] / d_ij
    # grad_i = sum_j factor[i, j] * (x_i - x_j)
    # = sum_j factor[i, j] * x_i - sum_j factor[i, j] * x_j
    # = x_i * sum_j factor[i, j] - sum_j factor[i, j] * x_j
    sum_factor = np.sum(factor, axis=1)
    weighted_x = factor @ x  # (N, 3)
    pair_grad = x * sum_factor[:, np.newaxis] - weighted_x
    
    # Wall contributions
    # wall_dists[i, k] = min(0.5 - x_ik, 0.5 + x_ik)
    # This is non-smooth at x_ik = 0.
    # For the gradient, we use the subgradient or just pick one side.
    # Since we're using a soft min over these values, the weight is applied.
    # The derivative of min(0.5 - x, 0.5 + x) w.r.t x is -1 if x < 0, +1 if x > 0, 0 if x == 0 (subgrad).
    # For x < 0: wall_dist = 0.5 + x, d/dx = 1
    # For x > 0: wall_dist = 0.5 - x, d/dx = -1
    # Let's use the sign: d/dx = -sign(x) (approximately, ignoring x=0)
    # Actually:
    # if x > 0, min is 0.5 - x, deriv is -1
    # if x < 0, min is 0.5 + x, deriv is 1
    # So deriv = -sign(x)
    # But we have weights from the soft min.
    # wall_weights is the portion of weights corresponding to wall distances
    wall_weights = weights[len(iu[0]):].reshape(N, 3)
    
    # grad_wall_i = sum_k wall_weights[i, k] * d(wall_dist_ik)/dx_ik
    # d(wall_dist_ik)/dx_ik = -sign(x_ik)
    # So grad_wall = wall_weights * (-sign(x))
    sign_x = np.sign(x)
    wall_grad = -wall_weights * sign_x
    
    grad = pair_grad + wall_grad
    
    # We are minimizing -soft_min, so objective = -soft_min, grad_obj = -grad
    return -soft_min, -grad

def _init_bcc():
    """
    Initialize with a BCC (Body Centered Cubic) like structure or similar dense packing.
    For N=50, we can use a 3x3x3 grid (27 points) + body centers (27) = 54, take first 50.
    Or a 4x4x3 grid (48) + 2 more.
    Let's try a jittered grid first, it's often effective.
    """
    # BCC lattice: points at (i, j, k) and (i+0.5, j+0.5, k+0.5) for i,j,k in grid
    # Scale to fit in [-0.5, 0.5]
    # Let's use a 4x4x4 grid of cells.
    m = 4
    cell_size = 1.0 / m
    centers = []
    for k in range(m):
        for j in range(m):
            for i in range(m):
                # Corner
                centers.append((i * cell_size - 0.5 + 0.5 * cell_size, 
                                j * cell_size - 0.5 + 0.5 * cell_size, 
                                k * cell_size - 0.5 + 0.5 * cell_size))
                # Body center
                centers.append((i * cell_size - 0.5 + 0.5 * cell_size, 
                                j * cell_size - 0.5 + 0.5 * cell_size, 
                                k * cell_size - 0.5 + 0.5 * cell_size))
                # This gives 2 * 4^3 = 128 points, too many.
                # Let's just use the grid points.
    
    # Let's stick to a simpler, denser grid for initialization.
    # 50 points in a cube.
    # 4x4x4 = 64 points.
    # We can take 50 of them.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take first 50? Or just take first 50.
    # Taking first 50 from a lexicographic grid might leave empty corners.
    # Let's jitter them.
    np.random.seed(42)
    grid = grid[:N] + np.random.randn(N, 3) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def _init_random():
    """Random initialization."""
    np.random.seed(123)
    x = np.random.uniform(-0.49, 0.49, (N, 3))
    return x

def construct():
    start_time = time.time()
    best_x = None
    best_r = -1.0
    
    # List of initializers
    initializers = [
        _init_bcc,
        _init_random
    ]
    
    # We will iterate through initializers and try to optimize.
    # We also have a time limit.
    
    for init_fn in initializers:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        x0 = init_fn()
        
        # Annealing-like schedule for lambda
        # Start with small lambda (smooth), increase to sharp (exact)
        lambdas = [5, 20, 100, 500]
        
        x_curr = x0.copy()
        
        for lam in lambdas:
            if time.time() - start_time > DEADLINE_SECONDS:
                break
                
            # Optimize with this lambda
            try:
                res = minimize(
                    _soft_min_radius_and_grad,
                    x_curr,
                    method='L-BFGS-B',
                    args=(lam,),
                    jac=True,
                    bounds=[(-0.499, 0.499)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 20}
                )
                x_curr = res.x
            except Exception:
                pass
            
            # Clip to valid region
            x_curr = np.clip(x_curr, -0.499, 0.499)
        
        # Evaluate exact radius
        r = _exact_radius(x_curr)
        
        if r > best_r:
            best_r = r
            best_x = x_curr.copy()
            
    # If we didn't find anything (shouldn't happen), return random
    if best_x is None:
        best_x = _init_random()
        
    # Final clip
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
