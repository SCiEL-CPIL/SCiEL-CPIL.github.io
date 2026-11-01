"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
from scipy.spatial.distance import pdist

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def _gram_to_points(G, n, d):
    """Extract points from Gram matrix via eigendecomposition, keeping top d eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(G)
    # Take top d eigenvalues
    idx = np.argsort(eigvals)[::-1][:d]
    V = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
    return V

def _rank_penalty(G, d, penalty_weight):
    """Penalty for eigenvalues beyond rank d."""
    eigvals = np.linalg.eigvalsh(G)
    # Sort in descending order
    eigvals_sorted = np.sort(eigvals)[::-1]
    # Penalty on eigenvalues beyond the first d
    excess = eigvals_sorted[d:]
    return penalty_weight * np.sum(excess ** 2)

def _sdp_objective(x, n, d, indices, num_pairs, T, penalty_weight):
    """
    SDP-style objective: optimize Gram matrix G (flattened upper triangle).
    Minimize ratio of max to min squared distance + rank penalty.
    """
    # Reconstruct symmetric G from flattened upper triangle
    G = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G[i, j] = x[idx]
            G[j, i] = x[idx]
            idx += 1
    
    # Ensure PSD by projecting: G = (G + G.T) / 2, then clip negative eigenvalues
    G = (G + G.T) / 2
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals_clipped = np.maximum(eigvals, 0)
    G_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    G_psd = (G_psd + G_psd.T) / 2
    
    # Compute squared distances from G: ||x_i - x_j||^2 = G_ii + G_jj - 2*G_ij
    d_sq = np.zeros(num_pairs)
    for k, (i, j) in enumerate(indices):
        d_sq[k] = G_psd[i, i] + G_psd[j, j] - 2 * G_psd[i, j]
    
    # Smooth max and min
    max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
    min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
    
    if min_approx < 1e-12:
        return 1e10
    
    ratio = max_approx / min_approx
    
    # Add rank penalty
    penalty = _rank_penalty(G_psd, d, penalty_weight)
    
    return ratio + penalty

def construct():
    n = N
    d = D
    
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # Number of variables: upper triangle of n x n matrix
    num_vars = n * (n + 1) // 2
    
    np.random.seed(42)
    
    # Initializations for Gram matrix
    # 1. Random PSD Gram
    V_rand = np.random.randn(n, d)
    G_rand = V_rand @ V_rand.T
    x_rand = np.zeros(num_vars)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            x_rand[idx] = G_rand[i, j]
            idx += 1
    
    # 2. Fibonacci sphere Gram
    x0_fib = np.zeros(n * d)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        x0_fib[i*d:(i+1)*d] = np.array([x, y, z])
    V_fib = x0_fib.reshape(n, d)
    G_fib = V_fib @ V_fib.T
    x_fib = np.zeros(num_vars)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            x_fib[idx] = G_fib[i, j]
            idx += 1
    
    # 3. Scaled identity-based initialization
    x_id = np.zeros(num_vars)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                x_id[idx] = 1.0
            else:
                x_id[idx] = 0.1
            idx += 1
    
    best_x = None
    best_val = 1e10
    
    # Anneal temperature and penalty weight
    T_values = [1.0, 0.5, 0.1, 0.05, 0.01]
    penalty_weights = [0.1, 0.5, 1.0, 2.0]
    
    for T in T_values:
        for pw in penalty_weights:
            for x0 in [x_rand, x_fib, x_id]:
                try:
                    res = minimize(
                        _sdp_objective, x0.copy(),
                        args=(n, d, indices, num_pairs, T, pw),
                        method='L-BFGS-B',
                        options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
    
    if best_x is None:
        # Fallback
        return np.random.randn(n, d)
    
    # Extract points from best Gram matrix
    G_best = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G_best[i, j] = best_x[idx]
            G_best[j, i] = best_x[idx]
            idx += 1
    G_best = (G_best + G_best.T) / 2
    
    points = _gram_to_points(G_best, n, d)
    
    # Refine in coordinate space
    def coord_objective(x_coords):
        V = x_coords.reshape(n, d)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            diff = V[i] - V[j]
            d_sq[k] = np.dot(diff, diff)
        if np.min(d_sq) < 1e-12:
            return 1e10
        T_ref = 0.01
        max_approx = T_ref * np.log(np.sum(np.exp(d_sq / T_ref)))
        min_approx = -T_ref * np.log(np.sum(np.exp(-d_sq / T_ref)))
        if min_approx < 1e-12:
            return 1e10
        return max_approx / min_approx
    
    try:
        res_refine = minimize(coord_objective, points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res_refine.fun < _compute_score(points):
            points = res_refine.x.reshape(n, d)
    except Exception:
        pass
    
    return points

# EVOLVE-BLOCK-END
