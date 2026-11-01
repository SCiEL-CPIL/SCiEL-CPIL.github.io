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
    
    # We optimize the Gram matrix G directly using SDP relaxation approach.
    # G is n x n symmetric PSD matrix.
    # We parameterize G as G = V @ V.T where V is n x d (d=3).
    # But to use SDP relaxation, we work with G directly and enforce rank via penalty.
    
    # Alternative: Since d=3, we can parameterize using a low-rank factorization.
    # Let's use the approach: minimize ratio of max to min squared distance,
    # with rank constraint enforced by eigenvalue penalty on the Gram matrix.
    
    # Variables: G flattened (n*n elements), but we only use upper triangle for symmetry.
    # Actually, let's use a factorization approach: X = V @ V.T where V is n x d.
    # This naturally enforces rank <= d and PSD.
    
    # Objective: minimize (max_{i<j} ||x_i - x_j||^2) / (min_{i<j} ||x_i - x_j||^2)
    # Subject to: G = X @ X.T is PSD (automatically satisfied)
    # And we add a penalty for rank > d (though with factorization, rank is at most d)
    
    # Since we're using factorization V (n x d), rank is automatically <= d.
    # We just need to minimize the distance ratio.
    
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    def objective_and_grad(x):
        """Objective and gradient for coordinate optimization."""
        V = x.reshape(n, d)
        
        # Compute squared distances
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            diff = V[i] - V[j]
            d_sq[k] = np.dot(diff, diff)
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10, np.zeros_like(x)
        
        # Ratio = max_d_sq / min_d_sq
        ratio = max_d_sq / min_d_sq
        
        # Compute gradient numerically or analytically
        # Let's use numerical gradient for reliability
        eps = 1e-7
        grad = np.zeros_like(x)
        for idx in range(len(x)):
            x_plus = x.copy()
            x_plus[idx] += eps
            f_plus = objective_and_grad(x_plus)[0]
            grad[idx] = (f_plus - ratio) / eps
        
        return ratio, grad
    
    # Better: Use a smooth surrogate objective
    def objective(x):
        V = x.reshape(n, d)
        
        # Compute squared distances
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            diff = V[i] - V[j]
            d_sq[k] = np.dot(diff, diff)
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        
        # Smooth log-sum-exp approximation for max and min
        # max_d_sq ≈ T * log(sum(exp(d_sq/T)))
        # min_d_sq ≈ -T * log(sum(exp(-d_sq/T)))
        T = 0.1
        max_approx = T * np.log(np.sum(np.exp(d_sq / T)))
        min_approx = -T * np.log(np.sum(np.exp(-d_sq / T)))
        
        if min_approx < 1e-12:
            return 1e10
        
        return max_approx / min_approx
    
    # Initial configurations
    np.random.seed(42)
    
    # 1. Random initialization
    x0_rand = np.random.randn(n * d)
    
    # 2. Fibonacci sphere
    x0_fib = np.zeros(n * d)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        x0_fib[i*d:(i+1)*d] = np.array([x, y, z])
    
    # 3. Simplex-like initialization
    x0_simplex = np.zeros(n * d)
    for i in range(n):
        x0_simplex[i*d:(i+1)*d] = np.random.randn(d)
    
    # 4. Grid
    side = int(np.ceil(n ** (1.0 / d)))
    grid = list(itertools.product(range(side), repeat=d))[:n]
    x0_grid = np.array(grid, dtype=float).flatten()
    
    best_x = None
    best_val = 1e10
    
    # Try multiple optimization methods and initializations
    for x0 in [x0_rand, x0_fib, x0_simplex, x0_grid]:
        for method in ['Nelder-Mead', 'L-BFGS-B']:
            try:
                if method == 'L-BFGS-B':
                    # L-BFGS-B requires gradient, use numerical
                    res = minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
                else:
                    res = minimize(objective, x0, method='Nelder-Mead', 
                                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass
    
    if best_x is None:
        # Fallback to grid
        return np.array(grid, dtype=float)
    
    points = best_x.reshape(n, d)
    
    # Post-processing: try to improve with local optimization on coordinates
    # Use the best point as starting point for a final refinement
    try:
        res_final = minimize(objective, best_x, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res_final.fun < best_val:
            best_x = res_final.x
            best_val = res_final.fun
    except Exception:
        pass
    
    points = best_x.reshape(n, d)
    return points

# EVOLVE-BLOCK-END
