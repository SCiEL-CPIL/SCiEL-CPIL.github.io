"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 118.0  # safety margin before 120s kill


def _hex_lattice_init(n):
    """Generate hexagonal lattice points and select n closest to center."""
    # Estimate spacing for ~n points in unit disk
    # Area of hex lattice per point ~ s^2 * sqrt(3)/2
    # Area of unit disk = pi
    # n * s^2 * sqrt(3)/2 ~ pi  =>  s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    # Add margin
    s *= 0.95
    
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
    
    points = np.array(points)
    # Sort by distance from center, take closest n
    idx = np.argsort(points[:, 2])
    selected = points[idx[:n]][:, :2]
    
    if len(selected) < n:
        # Fallback: fill remaining randomly
        remaining = n - len(selected)
        extra = np.random.randn(remaining, 2) * 0.3
        selected = np.vstack([selected, extra])
    
    # Add small random noise to break symmetry
    noise = np.random.randn(n, 2) * 0.001
    selected = selected + noise
    
    return selected


def _compute_score(pts):
    """Compute the packing score (minimum radius)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts**2, axis=1))
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_dist / 2.0, min_boundary)


def _soft_min_objective_and_grad(pts, beta):
    """
    Compute negative soft-min objective and its gradient.
    Returns (value, grad).
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Boundary distances
    radii_sq = np.sum(pts**2, axis=1)
    radii = np.sqrt(radii_sq)
    boundary_dists = 1.0 - radii
    
    # Constraint values
    # Pairwise: dists / 2
    # Boundary: boundary_dists
    pair_vals = dists / 2.0
    bound_vals = boundary_dists
    
    # We only care about upper triangle for pairs to avoid double counting in the sum
    # But the soft-min is over all constraints.
    # Let's collect all constraint values into a single vector for the log-sum-exp
    # To save memory and time, we can compute the log-sum-exp in parts.
    
    # For pairs, we take the upper triangle
    ii, jj = np.triu_indices(n, k=1)
    pair_vals_upper = pair_vals[ii, jj]
    
    # Combine all values
    # Note: The soft-min is defined over the set of all constraints.
    # The set includes each pair once and each boundary constraint once.
    all_vals = np.concatenate([pair_vals_upper, bound_vals])
    
    # Clip very small values to avoid log(0) or overflow in exp
    all_vals = np.maximum(all_vals, 1e-12)
    
    # Compute log-sum-exp of -beta * all_vals
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_val + np.log(sum_exp)
    
    # Objective: -softmin = - ( -1/beta * log_sum_exp ) = log_sum_exp / beta
    obj_val = log_sum_exp / beta
    
    # Compute weights p_k = exp(-beta * v_k) / sum(exp(-beta * v_j))
    weights = exp_vals / sum_exp
    
    # Gradient computation
    grad = np.zeros_like(pts)
    
    # 1. Pairwise contributions
    # For each pair (i, j) with i < j, weight is w_ij.
    # Constraint v_ij = dist(i,j) / 2.
    # grad_{P_i} v_ij = (P_i - P_j) / (2 * dist(i,j))
    # grad_{P_j} v_ij = (P_j - P_i) / (2 * dist(i,j))
    # grad_{P_i} (-softmin) = - sum_j w_ij * grad_{P_i} v_ij
    #                      = - sum_j w_ij * (P_i - P_j) / (2 * dist(i,j))
    
    # We can vectorize this.
    # Let W be a matrix of size N x N, initialized to 0.
    # W[i, j] = w_ij / (2 * dist(i,j)) for i < j.
    # Then grad_i = - ( P_i * sum_j W[i,j] - sum_j W[i,j] P_j )
    #             = - P_i * row_sum_W[i] + W[i,:] @ P
    
    # Construct W matrix
    W = np.zeros((n, n))
    # dists_upper = dists[ii, jj]
    dists_upper = dists[ii, jj]
    # Avoid division by zero
    safe_dists = np.maximum(dists_upper, 1e-10)
    
    # w_ij is weights[:n_pairs]
    w_pairs = weights[:len(ii)]
    
    # W[i, j] and W[j, i] should both be w_ij / (2 * d_ij)
    # Because the gradient for P_i involves sum over j, and for P_j involves sum over i.
    # Wait, the formula is:
    # grad P_i = - sum_{j != i} w_{ij} * (P_i - P_j) / (2 d_{ij})
    # This can be written as - sum_{j != i} [ w_{ij} / (2 d_{ij}) ] * (P_i - P_j)
    # Let A_{ij} = w_{ij} / (2 d_{ij}) if i != j, else 0.
    # Then grad P_i = - sum_j A_{ij} P_i + sum_j A_{ij} P_j
    #               = - P_i (sum_j A_{ij}) + sum_j A_{ij} P_j
    
    # So we need matrix A where A[i,j] = w_{ij} / (2 d_{ij}).
    # Note w_{ij} is the weight associated with the constraint (i,j).
    # In our `weights` vector, the first part corresponds to upper triangle pairs.
    # We need to map these weights to the matrix A.
    
    A = np.zeros((n, n))
    A[ii, jj] = w_pairs / (2.0 * safe_dists)
    A[jj, ii] = A[ii, jj]  # Symmetric
    
    # Row sums of A
    row_sums = np.sum(A, axis=1)
    
    # grad = - P * row_sums + A @ P
    # A @ P: (N, N) @ (N, 2) -> (N, 2)
    A_P = A @ pts
    grad_pairs = - pts * row_sums[:, np.newaxis] + A_P
    
    # 2. Boundary contributions
    # Constraint v_i = 1 - |P_i|
    # grad_{P_i} v_i = - P_i / |P_i|
    # grad_{P_i} (-softmin) = - w_i * (- P_i / |P_i|) = w_i * P_i / |P_i|
    
    w_bounds = weights[len(ii):]
    # safe_radii
    safe_radii = np.maximum(radii, 1e-10)
    grad_bounds = w_bounds[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    
    return obj_val, grad


def construct():
    """Construct 300 circle centers in unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Initialize with hexagonal lattice
    pts = _hex_lattice_init(N)
    
    # Ensure valid initial points (inside unit disk)
    radii = np.sqrt(np.sum(pts**2, axis=1))
    mask = radii < 1.0
    if np.sum(mask) < N:
        # Rescale
        pts = pts * (0.95 / np.max(radii))
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Adaptive beta schedule
    # Start with lower beta for smooth landscape, increase for sharper min
    beta_schedule = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    for beta in beta_schedule:
        if time.time() > _DEADLINE - 10:
            break
        
        # Define objective and gradient closures
        def obj(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            val, _ = _soft_min_objective_and_grad(x, beta)
            return val
        
        def grad(x):
            if time.time() > _DEADLINE - 5:
                raise ValueError("timeout")
            x = x.reshape(N, 2)
            _, g = _soft_min_objective_and_grad(x, beta)
            return g.ravel()
        
        try:
            result = minimize(
                obj,
                pts.ravel(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 500
                }
            )
            
            new_pts = result.x.reshape(N, 2)
            
            # Check validity
            radii = np.sqrt(np.sum(new_pts**2, axis=1))
            if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
            else:
                pass
                
        except Exception:
            pass
    
    # Final polish with high beta
    if time.time() < _DEADLINE - 15:
        for beta in [20000, 50000]:
            if time.time() > _DEADLINE - 5:
                break
            
            def obj(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                val, _ = _soft_min_objective_and_grad(x, beta)
                return val
            
            def grad(x):
                if time.time() > _DEADLINE - 5:
                    raise ValueError("timeout")
                x = x.reshape(N, 2)
                _, g = _soft_min_objective_and_grad(x, beta)
                return g.ravel()
            
            try:
                result = minimize(
                    obj,
                    best_pts.ravel(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxfun': 300
                    }
                )
                
                new_pts = result.x.reshape(N, 2)
                radii = np.sqrt(np.sum(new_pts**2, axis=1))
                if np.all(radii < 1.0) and np.all(np.isfinite(new_pts)):
                    new_score = _compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                        
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
