# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
from scipy.spatial.distance import pdist

N = 14
D = 3

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

def _gram_to_points(G):
    """
    Project Gram matrix G to rank D PSD matrix and return coordinates.
    """
    eigvals, eigvecs = np.linalg.eigh(G)
    # Keep top D eigenvalues
    idx = np.argsort(eigvals)[::-1][:D]
    eigvals_top = eigvals[idx]
    eigvecs_top = eigvecs[:, idx]
    
    # Ensure eigenvalues are non-negative
    eigvals_top = np.maximum(eigvals_top, 0)
    
    # Reconstruct G_proj
    G_proj = (eigvecs_top * eigvals_top) @ eigvecs_top.T
    
    # Ensure symmetry
    G_proj = (G_proj + G_proj.T) / 2.0
    
    # Extract points from G_proj
    # G_proj = X X^T. We can use Cholesky or SVD.
    # SVD is more robust for PSD matrices with numerical noise.
    U, s, Vt = np.linalg.svd(G_proj)
    # X = U * sqrt(s)
    X = U * np.sqrt(s)
    
    return X

def construct():
    n = N
    d = D
    
    # Number of variables in symmetric matrix G
    num_vars = n * (n + 1) // 2
    
    def x_to_G(x):
        """Reconstruct symmetric matrix from upper triangle vector."""
        G = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                G[i, j] = x[idx]
                G[j, i] = x[idx]
                idx += 1
        return G

    def G_to_x(G):
        """Flatten upper triangle of symmetric matrix."""
        x = []
        for i in range(n):
            for j in range(i, n):
                x.append(G[i, j])
        return np.array(x)

    def objective(x):
        G = x_to_G(x)
        
        # Compute squared distances
        # d_ij^2 = G_ii + G_jj - 2 G_ij
        diag = np.diag(G)
        D_sq = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D_sq[i, j] = diag[i] + diag[j] - 2 * G[i, j]
        
        # Extract upper triangle distances
        iu = np.triu_indices(n, 1)
        d_vals = D_sq[iu]
        
        # Filter out very small distances to avoid log issues
        d_vals_safe = np.maximum(d_vals, 1e-10)
        
        # Smooth max and min using log-sum-exp
        # max(d) approx T * log(sum(exp(d/T)))
        # min(d) approx -T * log(sum(exp(-d/T)))
        T = 0.01
        max_approx = T * np.log(np.sum(np.exp(d_vals_safe / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_vals_safe / T)))
        
        if min_approx < 1e-10:
            return 1e10
            
        ratio = max_approx / min_approx
        
        # Eigenvalue penalties
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1] # Descending
        
        # PSD penalty: sum of squares of negative eigenvalues
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2)
        
        # Rank penalty: sum of squares of eigenvalues from index D onwards (0-indexed, so D, D+1, ...)
        # We want rank <= D, so eigenvalues D..n-1 should be 0
        rank_eigs = eigvals[D:]
        rank_penalty = np.sum(rank_eigs ** 2)
        
        # Weight for penalties
        w_psd = 1.0
        w_rank = 1.0
        
        return ratio + w_psd * psd_penalty + w_rank * rank_penalty

    # Initialization: Random rank-d PSD matrix
    np.random.seed(42)
    V_init = np.random.randn(n, d)
    G_init = V_init @ V_init.T
    x0 = G_to_x(G_init)
    
    # Optimize
    # Using L-BFGS-B for better convergence on smooth-ish functions
    # Note: objective is not perfectly smooth due to eigenvalue sorting, but usually works well
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-6})
        x_opt = res.x
    except Exception:
        x_opt = x0
    
    G_opt = x_to_G(x_opt)
    
    # Project to rank d
    points = _gram_to_points(G_opt)
    
    # Verify distinctness
    if np.any(pdist(points) < 1e-8):
        # Fallback to random if points collapse
        points = np.random.randn(n, d)
        
    return points

# EVOLVE-BLOCK-END
