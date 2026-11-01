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
    # Compute all pairwise distances
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # We optimize the Gram matrix G directly.
    # Variables: upper triangle of G.
    
    def _gram_value(x):
        i_idx, j_idx = np.triu_indices(n)
        G = np.zeros((n, n))
        G[i_idx, j_idx] = x
        G[j_idx, i_idx] = x
        
        # 1. PSD Penalty
        eigvals = np.linalg.eigvalsh(G)
        neg_penalty = np.sum(np.maximum(0, -eigvals)**2)
        
        # 2. Rank Penalty (keep top d eigenvalues)
        eigvals_desc = np.sort(eigvals)[::-1]
        rank_penalty = np.sum(eigvals_desc[d:]**2)
        
        # 3. Distance Ratio
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = G[i,i] + G[j,j] - 2*G[i,j]
            
        # Smooth max and min
        T = 0.1
        # Clip to avoid numerical issues
        d_sq_clipped = np.clip(d_sq, 1e-10, 1e6)
        
        max_approx = T * np.log(np.sum(np.exp(d_sq_clipped / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq_clipped / T)))
        
        if min_approx < 1e-10:
            min_approx = 1e-10
            
        ratio = max_approx / min_approx
        
        total_val = ratio + 100.0 * neg_penalty + 100.0 * rank_penalty
        return total_val, G

    def objective_and_grad(x):
        val, G = _gram_value(x)
        
        # Numerical gradient
        grad = np.zeros_like(x)
        eps = 1e-5
        for idx in range(len(x)):
            x_plus = x.copy()
            x_plus[idx] += eps
            f_plus, _ = _gram_value(x_plus)
            grad[idx] = (f_plus - val) / eps
            
        return val, grad

    # Initialization: Start with random coordinates, compute Gram matrix
    np.random.seed(42)
    best_x = None
    best_val = 1e10
    
    for restart in range(5):
        V_init = np.random.randn(n, d)
        G_init = V_init @ V_init.T
        i_idx, j_idx = np.triu_indices(n)
        x0 = G_init[i_idx, j_idx].copy()
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        # Fallback
        V_init = np.random.randn(n, d)
        return V_init
    
    # Reconstruct G
    G_best = np.zeros((n, n))
    i_idx, j_idx = np.triu_indices(n)
    G_best[i_idx, j_idx] = best_x
    G_best[j_idx, i_idx] = best_x
    
    # Ensure PSD and extract points
    eigvals, eigvecs = np.linalg.eigh(G_best)
    eigvals = np.maximum(eigvals, 0)
    
    # Keep top d eigenvalues
    idx = np.argsort(eigvals)[::-1][:d]
    V_best = eigvecs[:, idx] * np.sqrt(eigvals[idx])
    
    return V_best

# EVOLVE-BLOCK-END
