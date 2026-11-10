# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _area_fast(x, n=64, M=20000):
    """Compute union area via quadrature at M heights."""
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
    return float(np.mean(union_len))

def _area_coarse(x, n=64, M=2000):
    """Compute union area via quadrature at M heights (faster for optimization)."""
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
    return float(np.mean(union_len))

def _aligned_positions(n=64):
    """
    Initialize such that all triangle apices converge to x=0.5 at y=1.
    Vertex at y=1 is (x_i + a_i, 1).
    We want x_i + a_i = 0.5 => x_i = 0.5 - a_i.
    """
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    return 0.5 - a

def _random_positions(n=64, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    # Random positions in a reasonable range
    # The triangles are width 1/64. To minimize union area, they should overlap.
    # A random start might help escape local minima of the aligned structure.
    return rng.uniform(-0.5, 1.0, size=n)

def _optimize_l_bfgs(x0, n=64, time_budget=300.0, M_coarse=2000):
    """
    Perform L-BFGS-B optimization on the coarse grid.
    """
    start_time = time.time()
    
    # Define objective for coarse grid
    def objective_coarse(x):
        if time.time() - start_time > time_budget:
            raise Exception("Time budget exceeded")
        return _area_coarse(x, n, M_coarse)
    
    # L-BFGS-B with finite differences
    # We need to be careful with the step size for finite differences
    # Default is usually okay, but we can specify eps if needed
    try:
        res = minimize(
            objective_coarse, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return res.x
    except Exception:
        return x0

def _polish_fine(x0, n=64, time_budget=100.0):
    """
    Polish the solution using the fine grid (M=20000) with a few L-BFGS-B iterations.
    """
    start_time = time.time()
    
    def objective_fine(x):
        if time.time() - start_time > time_budget:
            raise Exception("Time budget exceeded")
        return _area_fast(x, n, 20000)
    
    try:
        res = minimize(
            objective_fine, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
        return res.x
    except Exception:
        return x0

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 570.0 # Safety margin
    safety = 10.0
    
    best_x = None
    best_area = float('inf')
    
    # Strategy: Multiple starts with L-BFGS-B on coarse grid, then fine polish.
    # Start 1: Aligned
    # Start 2: Random
    # Start 3: Perturbed Aligned
    
    starts = []
    
    # 1. Aligned
    x_aligned = _aligned_positions(n)
    starts.append(x_aligned)
    
    # 2. Random
    rng = np.random.default_rng(123)
    x_rand = _random_positions(n, rng)
    starts.append(x_rand)
    
    # 3. Perturbed Aligned
    x_pert = x_aligned + rng.normal(0, 0.05, size=n)
    starts.append(x_pert)
    
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 100.0:
            break
            
        # Coarse optimization
        # Budget: ~150s per start for coarse
        x_coarse = _optimize_l_bfgs(x_start, n, time_budget=150.0, M_coarse=2000)
        
        # Evaluate on fine grid to compare
        area_coarse_fine = _area_fast(x_coarse, n, 20000)
        
        if area_coarse_fine < best_area:
            best_area = area_coarse_fine
            best_x = x_coarse.copy()
            
        # Fine polish if time remains
        if time.time() < deadline - safety - 30.0:
            x_fine = _polish_fine(x_coarse, n, time_budget=60.0)
            area_fine = _area_fast(x_fine, n, 20000)
            
            if area_fine < best_area:
                best_area = area_fine
                best_x = x_fine.copy()
                
    # Fallback
    if best_x is None:
        best_x = x_aligned
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
