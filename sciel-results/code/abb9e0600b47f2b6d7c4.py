# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x):
    """Compute the union area of 64 triangles given base positions x."""
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n  # shape (64,)
    # L_i(y) = x_i + a_i * y, shape (64, 20000)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n  # shape (20000,)
    # Sort L along axis 0 for each y
    L_sorted = np.sort(L, axis=0)  # shape (64, 20000)
    # Union length: w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    gaps = np.diff(L_sorted, axis=0)  # shape (63, 20000)
    union_len = w + np.sum(np.minimum(gaps, w[None, :]), axis=0)  # shape (20000,)
    area = np.mean(union_len)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 600.0
    
    # Start from the Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(x0, dtype=np.float64)
    
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Try several optimization strategies
    # Strategy 1: Gradient descent with analytic gradient approximation
    # Strategy 2: Nelder-Mead or L-BFGS-B
    
    # First, try to get a good initial guess using the "overlap" heuristic
    # We want x_i + a_i * y to be as clustered as possible
    # A good heuristic: x_i = -a_i * 0.5 + shift (midpoint compensation)
    a = (np.arange(n) + 1) / n
    x_heur = -a * 0.5
    x_heur -= np.min(x_heur)  # shift to keep values reasonable
    area_heur = _compute_area(x_heur)
    
    if area_heur < best_area:
        best_area = area_heur
        best_x = x_heur.copy()
    
    # Try several beta values for x_i = alpha - beta * a_i
    for beta in np.linspace(0.0, 1.0, 21):
        x_try = -beta * a
        x_try -= np.min(x_try)
        area_try = _compute_area(x_try)
        if area_try < best_area:
            best_area = area_try
            best_x = x_try.copy()
    
    # Now do gradient-based optimization from the best starting point
    # Use L-BFGS-B with numerical gradients (since analytic gradient is complex)
    def objective(x):
        return _compute_area(x)
    
    # Check time
    elapsed = time.time() - (deadline - 600.0)
    if elapsed < 500.0:
        # Try L-BFGS-B
        try:
            result = minimize(objective, best_x, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                     'maxfun': 20000})
            if result.success or result.fun < best_area:
                best_x = result.x
                best_area = result.fun
        except Exception:
            pass
    
    elapsed = time.time() - (deadline - 600.0)
    if elapsed < 450.0:
        # Try Nelder-Mead from multiple starting points
        starting_points = [best_x.copy(), x0.copy(), x_heur.copy()]
        # Add some perturbed versions
        rng = np.random.RandomState(42)
        for _ in range(3):
            starting_points.append(best_x + rng.randn(n) * 0.01)
        
        for sp in starting_points:
            if time.time() > deadline - 60.0:
                break
            try:
                result = minimize(objective, sp, method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                if result.fun < best_area:
                    best_area = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    # Final refinement with smaller steps
    elapsed = time.time() - (deadline - 600.0)
    if elapsed < 550.0:
        try:
            result = minimize(objective, best_x, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                     'maxfun': 50000})
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_x = np.asarray(best_x, dtype=np.float64)
    if np.any(~np.isfinite(best_x)):
        best_x = x0.copy()
    
    return best_x

# EVOLVE-BLOCK-END
