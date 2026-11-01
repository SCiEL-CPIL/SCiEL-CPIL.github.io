"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    if n < 2:
        return 1e10
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
    
    def objective(x):
        """
        Objective function for optimization.
        x: flattened array of shape (n*d,)
        Returns: soft approximation of (max_dist / min_dist)^2
        """
        P = x.reshape(n, d)
        # Center the points to remove translation invariance
        P = P - np.mean(P, axis=0, keepdims=True)
        
        # Compute squared distances
        d_sq = pdist(P) ** 2
        
        # Avoid log(0) or division by zero
        if np.any(d_sq < 1e-12):
            return 1e10
            
        # Temperature for soft max/min
        T = 0.1
        
        # Soft max of squared distances
        # max_d_sq ~ T * log(sum(exp(d_sq/T)))
        # To avoid overflow, subtract max
        max_d_sq_raw = np.max(d_sq)
        soft_max_d_sq = T * (np.log(np.sum(np.exp((d_sq - max_d_sq_raw) / T))) + max_d_sq_raw / T)
        
        # Soft min of squared distances
        # min_d_sq ~ -T * log(sum(exp(-d_sq/T)))
        min_d_sq_raw = np.min(d_sq)
        soft_min_d_sq = -T * (np.log(np.sum(np.exp(-(d_sq - min_d_sq_raw) / T))) - min_d_sq_raw / T)
        
        if soft_min_d_sq < 1e-12:
            return 1e10
            
        ratio = soft_max_d_sq / soft_min_d_sq
        return ratio

    best_points = None
    best_score = 1e10
    
    # Multiple initializations
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # 1. Random points
    initializations.append(rng.normal(size=(n, d)))
    
    # 2. Points on a sphere (random)
    V = rng.normal(size=(n, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    initializations.append(V)
    
    # 3. Simplex-like initialization (if n <= d+1, but here n=14, d=3, so not a simplex)
    # Use a configuration close to a cube or other symmetric structure
    # For 14 points, maybe 2 cubes?
    cube1 = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ])
    cube2 = cube1 * 0.5
    initializations.append(np.vstack([cube1, cube2]))
    
    # 4. Fibonacci sphere
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / (phi ** 2)
    z = np.linspace(-1, 1, n)
    r = np.sqrt(1 - z**2)
    theta = golden_angle * np.arange(n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    initializations.append(np.column_stack([x, y, z]))
    
    for init in initializations:
        try:
            res = minimize(objective, init.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e9:
                P_opt = res.x.reshape(n, d)
                P_opt = P_opt - np.mean(P_opt, axis=0, keepdims=True)
                
                # Final score calculation
                score = _compute_score(P_opt)
                
                if score < best_score:
                    best_score = score
                    best_points = P_opt
                    
        except Exception:
            continue
            
    if best_points is None:
        return np.random.randn(n, d)
        
    return best_points
# EVOLVE-BLOCK-END
