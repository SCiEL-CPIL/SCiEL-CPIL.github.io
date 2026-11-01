"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
from scipy.spatial.distance import pdist

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
    
    # Vectorized distance calculation helper
    def get_d_sq(V):
        return pdist(V) ** 2

    def objective_smooth(x, temp):
        V = x.reshape(n, d)
        
        # Normalize to unit trace to fix scale
        norm_sq = np.sum(V * V)
        if norm_sq < 1e-12:
            return 1e10
        V_norm = V / np.sqrt(norm_sq)
        
        d_sq = get_d_sq(V_norm)
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
            
        t = temp
        
        # Soft Max: (1/t) * log(sum(exp(t * d_sq)))
        # Stable version:
        m = max_d_sq
        soft_max = m + (1.0 / t) * np.log(np.sum(np.exp(t * (d_sq - m))))
        
        # Soft Min: -(1/t) * log(sum(exp(-t * d_sq)))
        # Stable version:
        m_min = min_d_sq
        soft_min = m_min - (1.0 / t) * np.log(np.sum(np.exp(-t * (d_sq - m_min))))
        
        if soft_min < 1e-12:
            return 1e10
            
        return soft_max / soft_min

    best_x = None
    best_val = 1e10
    
    # Initializations
    inits = []
    np.random.seed(42)
    
    # 1. Random Gaussian
    inits.append(np.random.randn(n * d))
    
    # 2. Fibonacci Sphere
    x0_fib = np.zeros(n * d)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        x0_fib[i*d:(i+1)*d] = np.array([x, y, z])
    inits.append(x0_fib)
    
    # 3. Random Uniform in Cube
    inits.append(np.random.rand(n * d) * 2 - 1)
    
    # 4. Grid
    side = int(np.ceil(n ** (1.0 / d)))
    grid = list(itertools.product(range(side), repeat=d))[:n]
    inits.append(np.array(grid, dtype=float).flatten())
    
    # Optimization with Annealing
    temps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for x0 in inits:
        x_curr = x0.copy()
        # Normalize
        norm_sq = np.sum(x_curr * x_curr)
        if norm_sq > 0:
            x_curr = x_curr / np.sqrt(norm_sq)
        
        for t in temps:
            try:
                res = minimize(
                    lambda x: objective_smooth(x, t), 
                    x_curr, 
                    method='Powell', 
                    options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-8, 'maxfev': 5000}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
                x_curr = res.x
            except Exception:
                pass
        
        # Final sharp optimization
        try:
            res = minimize(
                lambda x: objective_smooth(x, 0.001), 
                x_curr, 
                method='Powell', 
                options={'maxiter': 5000, 'xtol': 1e-10, 'ftol': 1e-10, 'maxfev': 10000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        return np.array(inits[0]).reshape(n, d)
        
    points = best_x.reshape(n, d)
    norm_sq = np.sum(points * points)
    if norm_sq > 0:
        points = points / np.sqrt(norm_sq)
        
    return points

# EVOLVE-BLOCK-END
