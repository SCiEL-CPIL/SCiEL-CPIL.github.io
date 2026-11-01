"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _soft_min(x, eps):
    """Smooth approximation of min(x). x is 1D array."""
    # Use log-sum-exp trick for stability
    x = np.asarray(x, dtype=np.float64)
    m = x.min()
    if m < -1000:
        return m
    return m - eps * np.log(np.sum(np.exp(-(x - m) / eps)))

def _objective_and_grad(params, eps):
    """
    params: flat array of 2*N center coordinates.
    Returns (-soft_min_value, -gradient) for minimization.
    """
    c = params.reshape(N, 2)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu]
    # Boundary distances
    r_centers = np.sqrt(np.sum(c ** 2, axis=1))
    b_vals = 1.0 - r_centers
    # All constraint values
    all_vals = np.concatenate([d_pairs / 2.0, b_vals])
    
    # Soft min
    m = all_vals.min()
    if m < -1000:
        val = m
        grad = np.zeros_like(params)
        return -val, -grad
    
    # d(soft_min)/d(all_vals) = -exp(-x/eps) / sum(exp(-x/eps))
    exp_terms = np.exp(-(all_vals - m) / eps)
    denom = np.sum(exp_terms)
    weights = exp_terms / denom  # sum to 1
    
    # Gradient w.r.t. all_vals
    grad_all = -weights / eps  # d(soft_min)/dx_k = -w_k / eps ... wait
    # soft_min = m - eps * log(sum exp(-(x-m)/eps))
    # d(soft_min)/dx_k = -eps * (1/sum) * d(sum)/dx_k = -eps * (1/sum) * exp(-(x_k-m)/eps) * (-1/eps) = exp(-(x_k-m)/eps)/sum = w_k
    # So d(soft_min)/dx_k = w_k
    grad_all = weights  # this is d(soft_min)/dx_k
    
    # Now chain rule: d(soft_min)/d(params) = sum_k grad_all[k] * dx_k/d(params)
    grad_params = np.zeros_like(params)
    
    # d_pairs: d_ij = sqrt((c_i - c_j)^2)
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
    # d(d_ij/2)/d(c_i) = (c_i - c_j) / (2 * d_ij)
    
    # For each pair (i,j), contribution to grad_params
    # grad_all[0:len(d_pairs)] corresponds to d_pairs/2
    pair_weights = grad_all[:len(d_pairs)]
    
    # Build gradient from pairs
    # For pair (i,j) with weight w:
    #   grad_params[2*i:2*i+2] += w * (c_i - c_j) / (2 * d_ij)
    #   grad_params[2*j:2*j+2] += w * (c_j - c_i) / (2 * d_ij)
    
    # Vectorized: use the diff matrix
    # diff[i,j] = c_i - c_j
    # For upper triangle, we need to accumulate
    # Let's do it with a loop over pairs for clarity (N=50, ~1225 pairs, fast)
    
    # Actually, let's vectorize:
    # inv_dist = 1/d for all pairs, 0 where d=0
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = np.where(dist > 1e-10, 1.0/dist, 0.0)
    
    # grad_pair[i,j] = w_ij * (c_i - c_j) / (2 * d_ij) for i<j, 0 otherwise
    # w_ij is only defined for i<j
    # Let's build a weight matrix
    W = np.zeros((N, N))
    W[iu] = pair_weights
    
    # grad_params[i] = sum_j W[i,j] * (c_i - c_j) / (2 * d_ij) - sum_j W[j,i] * (c_j - c_i) / (2 * d_ij)
    # = sum_j W[i,j] * (c_i - c_j) / (2 * d_ij) + sum_j W[j,i] * (c_i - c_j) / (2 * d_ij)
    # Since W is symmetric (we only fill upper, but W[j,i] = 0 for j<i in upper... wait)
    # Actually W[iu] = pair_weights, so W[i,j] = w for i<j, W[j,i] = 0
    # So grad_params[i] = sum_{j>i} W[i,j] * (c_i - c_j)/(2*d_ij) + sum_{j<i} W[j,i] * (c_i - c_j)/(2*d_ij)
    # Note: for j<i, W[j,i] = 0 in our fill, but we want the contribution from pair (j,i) which has weight w_ji
    # Let me just make W symmetric
    W = W + W.T
    
    # Now grad_params[i] = sum_j W[i,j] * (c_i - c_j) / (2 * d_ij)
    # = sum_j W[i,j] * diff[i,j] * inv_dist[i,j] / 2
    
    # diff[i,j] * inv_dist[i,j] = unit vector from j to i
    # So grad_params = 0.5 * sum_j W[i,j] * unit_vec[i,j]
    
    # Vectorized:
    # unit_vec[i,j] = diff[i,j] * inv_dist[i,j]
    unit_vec = diff * inv_dist[:, :, np.newaxis]
    # grad_params[i] = 0.5 * sum_j W[i,j] * unit_vec[i,j]
    grad_pairs = 0.5 * np.sum(W[:, :, np.newaxis] * unit_vec, axis=1)
    
    # Boundary: b_i = 1 - |c_i|
    # d(b_i)/d(c_i) = -c_i / |c_i|
    b_weights = grad_all[len(d_pairs):]
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r = np.where(r_centers > 1e-10, 1.0/r_centers, 0.0)
    grad_boundary = -b_weights[:, np.newaxis] * c * inv_r[:, np.newaxis]
    
    grad_params = grad_pairs + grad_boundary
    
    val = soft_min_val = m - eps * np.log(denom)
    return -val, -grad_params

def _score(c):
    """Compute the actual score for center array c."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dist[iu].min()
    r_centers = np.sqrt(np.sum(c ** 2, axis=1))
    min_boundary = (1.0 - r_centers).min()
    return min(min_pair / 2.0, min_boundary)

def _hex_init():
    """Initialize on hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def construct():
    # Initialize
    c0 = _hex_init()
    best_c = c0.copy()
    best_score = _score(c0)
    
    # Annealing schedule for epsilon
    epsilons = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    params = c0.flatten()
    
    for eps in epsilons:
        if time.time() > _DEADLINE:
            break
        
        # Use L-BFGS-B
        res = minimize(
            _objective_and_grad,
            params,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        new_params = res.x
        new_c = new_params.reshape(N, 2)
        new_score = _score(new_c)
        
        if new_score > best_score:
            best_score = new_score
            best_c = new_c.copy()
            params = new_params
        else:
            # Keep previous if worse
            params = best_c.flatten()
        
        if time.time() > _DEADLINE:
            break
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 5.0:
        eps_final = 1e-5
        res = minimize(
            _objective_and_grad,
            best_c.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        new_c = res.x.reshape(N, 2)
        new_score = _score(new_c)
        if new_score > best_score:
            best_c = new_c
    
    return best_c
# EVOLVE-BLOCK-END
