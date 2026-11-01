"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    
    # Precompute index arrays for vectorized distance computation
    ii = np.array([i for i, j in indices])
    jj = np.array([j for i, j in indices])
    
    def gram_to_distances(G):
        """Compute squared distances from Gram matrix G."""
        # d^2_ij = G_ii + G_jj - 2*G_ij
        return G[ii, ii] + G[jj, jj] - 2 * G[ii, jj]
    
    def eigenvalue_penalty(G, target_rank=3):
        """Penalty for eigenvalues beyond target_rank."""
        eigvals = np.linalg.eigvalsh(G)
        eigvals_sorted = np.sort(eigvals)[::-1]
        # Penalize eigenvalues at positions >= target_rank
        excess = eigvals_sorted[target_rank:]
        return np.sum(np.maximum(excess, 0) ** 2)
    
    def objective_gram(x, T=0.05, lam=1.0):
        """
        Objective for Gram matrix optimization.
        x: flattened symmetric matrix (upper triangle)
        """
        # Reconstruct symmetric Gram matrix from upper triangle
        G = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                G[i, j] = x[idx]
                G[j, i] = x[idx]
                idx += 1
        
        # Ensure PSD by projecting: G = max(G, 0) in eigenbasis
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals_clipped = np.maximum(eigvals, 0)
        G_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
        
        # Compute squared distances
        d_sq = gram_to_distances(G_psd)
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Smooth max and min using log-sum-exp
        max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
        
        if min_approx < 1e-12:
            return 1e10
        
        ratio = max_approx / min_approx
        
        # Eigenvalue penalty to enforce rank <= d
        eigvals_psd = np.sort(eigvals_clipped)[::-1]
        excess = eigvals_psd[d:]
        penalty = lam * np.sum(np.maximum(excess, 0) ** 2)
        
        return ratio + penalty
    
    def objective_gram_grad(x, T=0.05, lam=1.0):
        """
        Objective with numerical gradient for Gram matrix.
        """
        f = objective_gram(x, T, lam)
        if f > 1e9:
            return f, np.zeros_like(x)
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            f_plus = objective_gram(x_plus, T, lam)
            x_minus = x.copy()
            x_minus[k] -= eps
            f_minus = objective_gram(x_minus, T, lam)
            grad[k] = (f_plus - f_minus) / (2 * eps)
        
        return f, grad
    
    def gram_from_coords(coords):
        """Compute Gram matrix from coordinates."""
        return coords @ coords.T
    
    def extract_coords_from_gram(G, d=3):
        """Extract d-dimensional coordinates from PSD Gram matrix."""
        eigvals, eigvecs = np.linalg.eigh(G)
        # Take top d eigenvalues
        idx = np.argsort(eigvals)[::-1][:d]
        eigvals_top = eigvals[idx]
        eigvecs_top = eigvecs[:, idx]
        # Coordinates: V = eigvecs_top * sqrt(eigvals_top)
        V = eigvecs_top * np.sqrt(np.maximum(eigvals_top, 0))
        return V
    
    # Generate initial Gram matrices
    np.random.seed(42)
    
    initial_grams = []
    
    # 1. Random PSD Gram matrix
    for trial in range(5):
        V_rand = np.random.randn(n, d + 2)  # Start with higher rank
        G_rand = V_rand @ V_rand.T
        # Flatten upper triangle
        G_flat = np.zeros(n * (n + 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                G_flat[idx] = G_rand[i, j]
                idx += 1
        initial_grams.append(G_flat)
    
    # 2. Fibonacci sphere Gram matrix
    golden_ratio = (1 + np.sqrt(5)) / 2
    V_fib = np.zeros((n, d))
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        V_fib[i, 0] = np.sin(phi) * np.cos(theta)
        V_fib[i, 1] = np.sin(phi) * np.sin(theta)
        V_fib[i, 2] = np.cos(phi)
    G_fib = V_fib @ V_fib.T
    G_fib_flat = np.zeros(n * (n + 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G_fib_flat[idx] = G_fib[i, j]
            idx += 1
    initial_grams.append(G_fib_flat)
    
    # 3. Regular simplex-like (centered)
    V_simplex = np.random.randn(n, d)
    V_simplex -= V_simplex.mean(axis=0)
    G_simplex = V_simplex @ V_simplex.T
    G_simplex_flat = np.zeros(n * (n + 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G_simplex_flat[idx] = G_simplex[i, j]
            idx += 1
    initial_grams.append(G_simplex_flat)
    
    # 4. Identity-based (points on sphere with equal pairwise distances ideal)
    G_id = np.eye(n)
    G_id_flat = np.zeros(n * (n + 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G_id_flat[idx] = G_id[i, j]
            idx += 1
    initial_grams.append(G_id_flat)
    
    best_flat = None
    best_val = 1e10
    
    # Optimize Gram matrix with multiple strategies
    for g0 in initial_grams:
        for T_val in [0.05, 0.1, 0.2]:
            for lam_val in [0.01, 0.1, 1.0]:
                try:
                    res = minimize(
                        objective_gram,
                        g0,
                        args=(T_val, lam_val),
                        method='L-BFGS-B',
                        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.fun < best_val:
                        best_val = res.fun
                        best_flat = res.x
                except Exception:
                    pass
        
        # Also try Nelder-Mead for local refinement
        try:
            res = minimize(
                objective_gram,
                g0,
                args=(0.1, 0.1),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_flat = res.x
        except Exception:
            pass
    
    if best_flat is None:
        # Fallback: use Fibonacci sphere
        V_fib = np.zeros((n, d))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = 2 * np.pi * i / golden_ratio
            phi = np.arccos(1 - 2 * (i + 0.5) / n)
            V_fib[i, 0] = np.sin(phi) * np.cos(theta)
            V_fib[i, 1] = np.sin(phi) * np.sin(theta)
            V_fib[i, 2] = np.cos(phi)
        return V_fib
    
    # Extract coordinates from best Gram matrix
    G_best = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G_best[i, j] = best_flat[idx]
            G_best[j, i] = best_flat[idx]
            idx += 1
    
    points = extract_coords_from_gram(G_best, d)
    
    # Refine in coordinate space
    def objective_coords(x, T=0.05):
        V = x.reshape(n, d)
        d_sq = np.zeros(num_pairs)
        for k in range(num_pairs):
            diff = V[ii[k]] - V[jj[k]]
            d_sq[k] = np.dot(diff, diff)
        d_sq = np.maximum(d_sq, 1e-12)
        max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
        if min_approx < 1e-12:
            return 1e10
        return max_approx / min_approx
    
    x0_coords = points.flatten()
    try:
        res = minimize(
            objective_coords,
            x0_coords,
            args=(0.05,),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_val:
            points = res.x.reshape(n, d)
    except Exception:
        pass
    
    try:
        res = minimize(
            objective_coords,
            x0_coords,
            args=(0.1,),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < best_val:
            points = res.x.reshape(n, d)
    except Exception:
        pass
    
    return points

# EVOLVE-BLOCK-END
