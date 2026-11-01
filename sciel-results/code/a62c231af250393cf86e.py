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

def construct():
    n = N
    d = D
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # SDP relaxation: optimize Gram matrix G (n x n symmetric)
    # Variables: upper triangle of G flattened
    # Constraints: G PSD (eigenvalues >= 0), rank(G) <= d (eigenvalues beyond top d ~ 0)
    # Objective: minimize ratio of max to min squared pairwise distance
    
    # Number of variables: n*(n+1)/2 for upper triangle
    num_vars = n * (n + 1) // 2
    
    def _unpack_G(x):
        """Reconstruct symmetric matrix G from upper-triangle flat vector."""
        G = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                G[i, j] = x[idx]
                G[j, i] = x[idx]
                idx += 1
        return G
    
    def _pack_G(G):
        """Flatten upper triangle of symmetric matrix G."""
        x = np.zeros(num_vars)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                x[idx] = G[i, j]
                idx += 1
        return x
    
    def _squared_distances_from_G(G):
        """Compute squared pairwise distances from Gram matrix."""
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = G[i, i] + G[j, j] - 2 * G[i, j]
        return d_sq
    
    def objective_sdp(x, T=0.1, lambda_psd=1.0, lambda_rank=1.0):
        """
        SDP relaxation objective with penalties.
        - Smooth max/min ratio of squared distances
        - Penalty for negative eigenvalues (PSD violation)
        - Penalty for eigenvalues beyond rank d (rank violation)
        """
        G = _unpack_G(x)
        
        # Compute squared distances
        d_sq = _squared_distances_from_G(G)
        
        # Clip to avoid log of negative
        d_sq_clipped = np.maximum(d_sq, 1e-12)
        
        # Smooth max and min using log-sum-exp
        max_approx = T * np.log(np.sum(np.exp(d_sq_clipped / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq_clipped / T)))
        
        if min_approx < 1e-12:
            min_approx = 1e-12
        
        ratio = max_approx / min_approx
        
        # Eigenvalue decomposition for penalties
        try:
            eigvals = np.linalg.eigvalsh(G)
        except np.linalg.LinAlgError:
            eigvals = np.zeros(n)
        
        # PSD penalty: sum of squares of negative eigenvalues
        neg_eigs = np.minimum(eigvals, 0)
        psd_penalty = np.sum(neg_eigs ** 2)
        
        # Rank penalty: sum of squares of eigenvalues beyond the top d
        sorted_eigs = np.sort(eigvals)[::-1]  # descending
        if d < n:
            rank_penalty = np.sum(sorted_eigs[d:] ** 2)
        else:
            rank_penalty = 0.0
        
        total = ratio + lambda_psd * psd_penalty + lambda_rank * rank_penalty
        return total
    
    # Initial Gram matrices from various point configurations
    np.random.seed(42)
    
    def _make_G_from_points(points):
        """Create Gram matrix from point coordinates."""
        return points @ points.T
    
    # 1. Fibonacci sphere
    points_fib = np.zeros((n, d))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        points_fib[i] = [np.sin(phi) * np.cos(theta),
                         np.sin(phi) * np.sin(theta),
                         np.cos(phi)]
    G_fib = _make_G_from_points(points_fib)
    x0_fib = _pack_G(G_fib)
    
    # 2. Random
    points_rand = np.random.randn(n, d)
    G_rand = _make_G_from_points(points_rand)
    x0_rand = _pack_G(G_rand)
    
    # 3. Grid
    side = int(np.ceil(n ** (1.0 / d)))
    grid = list(itertools.product(range(side), repeat=d))[:n]
    points_grid = np.array(grid, dtype=float)
    G_grid = _make_G_from_points(points_grid)
    x0_grid = _pack_G(G_grid)
    
    # 4. Regular polytope-like: place points on sphere with some structure
    points_sphere = np.zeros((n, d))
    for i in range(n):
        angle = 2 * np.pi * i / n
        points_sphere[i, 0] = np.cos(angle)
        points_sphere[i, 1] = np.sin(angle)
        points_sphere[i, 2] = 0.5 * (i - n/2) / n
    G_sphere = _make_G_from_points(points_sphere)
    x0_sphere = _pack_G(G_sphere)
    
    best_x = None
    best_val = 1e10
    
    # Optimize over multiple initializations and temperature values
    for x0 in [x0_fib, x0_rand, x0_grid, x0_sphere]:
        for T in [0.05, 0.1, 0.2, 0.5]:
            for lambda_psd in [0.5, 1.0, 2.0]:
                for lambda_rank in [0.5, 1.0, 2.0]:
                    try:
                        res = minimize(
                            objective_sdp, x0,
                            method='Nelder-Mead',
                            args=(T, lambda_psd, lambda_rank),
                            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                        )
                        if res.fun < best_val:
                            best_val = res.fun
                            best_x = res.x
                    except Exception:
                        pass
    
    if best_x is None:
        # Fallback
        return points_fib
    
    G_best = _unpack_G(best_x)
    
    # Extract points from Gram matrix via eigenvalue decomposition
    # G = V @ V.T, so V = Q[:, :d] * sqrt(eigvals[:d])
    try:
        eigvals, eigvecs = np.linalg.eigh(G_best)
        # Take top d eigenvalues (should be positive if PSD)
        idx = np.argsort(eigvals)[::-1][:d]
        eigvals_top = eigvals[idx]
        eigvecs_top = eigvecs[:, idx]
        
        # Ensure non-negative eigenvalues
        eigvals_top = np.maximum(eigvals_top, 0)
        
        points = eigvecs_top @ np.diag(np.sqrt(eigvals_top))
    except Exception:
        points = points_fib
    
    # Post-refinement: optimize coordinates directly from the SDP solution
    def objective_coords(x):
        V = x.reshape(n, d)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            diff = V[i] - V[j]
            d_sq[k] = np.dot(diff, diff)
        d_sq = np.maximum(d_sq, 1e-12)
        T = 0.1
        max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
        if min_approx < 1e-12:
            return 1e10
        return max_approx / min_approx
    
    x0_coords = points.flatten()
    try:
        res_final = minimize(
            objective_coords, x0_coords,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if res_final.fun < 1e10:
            points = res_final.x.reshape(n, d)
    except Exception:
        pass
    
    return points

# EVOLVE-BLOCK-END
