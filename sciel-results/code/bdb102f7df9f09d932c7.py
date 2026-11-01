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
    
    # SDP Relaxation Approach:
    # Optimize the Gram matrix G directly.
    # G must be symmetric and PSD.
    # We parameterize G by its upper triangular elements.
    # We add penalties for non-PSD eigenvalues and for rank > d.
    
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # Map from (i, j) with i <= j to index in flat array
    # We only store upper triangle including diagonal
    upper_indices = [(i, j) for i in range(n) for j in range(i, n)]
    num_upper = len(upper_indices)
    
    def upper_to_gram(x):
        """Convert flat upper triangle array to symmetric Gram matrix."""
        G = np.zeros((n, n))
        for k, (i, j) in enumerate(upper_indices):
            G[i, j] = x[k]
            G[j, i] = x[k]
        return G
    
    def gram_to_upper(G):
        """Extract upper triangle from symmetric Gram matrix."""
        x = np.zeros(num_upper)
        for k, (i, j) in enumerate(upper_indices):
            x[k] = G[i, j]
        return x
    
    def objective(x):
        G = upper_to_gram(x)
        
        # Compute eigenvalues of G
        try:
            eigvals = np.linalg.eigvalsh(G)
        except np.linalg.LinAlgError:
            return 1e10
        
        # Penalty for non-PSD (negative eigenvalues)
        neg_eigvals = eigvals[eigvals < 0]
        if len(neg_eigvals) > 0:
            psd_penalty = 100.0 * np.sum(neg_eigvals**2)
        else:
            psd_penalty = 0.0
            
        # Penalty for rank > d
        # We want only d non-zero eigenvalues.
        # Sort eigenvalues in descending order
        eigvals_sorted = np.sort(eigvals)[::-1]
        # Penalties for eigenvalues beyond index d-1
        if d < n:
            rank_penalty = 10.0 * np.sum(eigvals_sorted[d:]**2)
        else:
            rank_penalty = 0.0
            
        # Compute squared distances from G
        # d_ij^2 = G_ii + G_jj - 2 G_ij
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            val = G[i, i] + G[j, j] - 2 * G[i, j]
            # If val is negative, it's a sign of non-PSD or numerical error
            # We clip to 0 for distance calculation but the PSD penalty handles it
            if val < 0:
                d_sq[k] = 0.0
            else:
                d_sq[k] = val
                
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            # Avoid division by zero
            ratio_penalty = 1e10
        else:
            # Smooth approximation for max/min ratio
            # Use log-sum-exp for stability
            T = 0.1
            # Max approximation
            max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
            # Min approximation
            min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
            
            if min_approx < 1e-12:
                ratio_penalty = 1e10
            else:
                ratio = max_approx / min_approx
                ratio_penalty = ratio ** 2 # We want to minimize squared ratio
        
        return ratio_penalty + psd_penalty + rank_penalty

    # Initial configuration:
    # Start with a valid Gram matrix from random points in 3D
    np.random.seed(42)
    V_init = np.random.randn(n, d)
    G_init = V_init @ V_init.T
    x0 = gram_to_upper(G_init)
    
    # Optimization
    # Nelder-Mead is derivative-free and robust for this non-convex problem
    # Powell is also good
    best_x = x0
    best_val = objective(x0)
    
    # Try multiple methods
    for method in ['Nelder-Mead', 'Powell']:
        try:
            if method == 'Nelder-Mead':
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(objective, x0, method='Powell',
                               options={'maxiter': 20000, 'ftol': 1e-8, 'xtol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception as e:
            pass
            
    # Extract points from the optimized Gram matrix
    G_final = upper_to_gram(best_x)
    
    # Ensure PSD by clipping negative eigenvalues to 0
    eigvals, eigvecs = np.linalg.eigh(G_final)
    eigvals_clipped = np.clip(eigvals, 0, None)
    
    # Project to rank d
    # Keep top d eigenvalues/vectors
    if d < n:
        idx = np.argsort(eigvals_clipped)[::-1][:d]
        eigvals_top = eigvals_clipped[idx]
        eigvecs_top = eigvecs[:, idx]
    else:
        eigvals_top = eigvals_clipped
        eigvecs_top = eigvecs
        
    # Reconstruct V
    # V = eigvecs_top * sqrt(eigvals_top)
    V_final = eigvecs_top @ np.diag(np.sqrt(eigvals_top))
    
    # Center the points (optional, but good for numerical stability)
    V_final = V_final - np.mean(V_final, axis=0)
    
    return V_final

# EVOLVE-BLOCK-END
