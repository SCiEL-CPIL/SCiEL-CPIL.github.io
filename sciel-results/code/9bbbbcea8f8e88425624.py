# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x):
    """Compute the exact union area of 64 triangles given base positions x."""
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    union_len = w + np.sum(np.minimum(gaps, w[None, :]), axis=0)
    area = np.mean(union_len)
    return area

def _compute_area_smooth(x, eps=0.001):
    """Smooth approximation of union area using soft-min for |gaps-w|."""
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    # soft_min(gaps, w) = (gaps + w - soft_abs(gaps - w)) / 2
    diff = gaps - w[None, :]
    soft_abs = np.sqrt(diff * diff + eps * eps)
    soft_min = (gaps + w[None, :] - soft_abs) / 2.0
    union_len = w + np.sum(soft_min, axis=0)
    area = np.mean(union_len)
    return area

def _compute_area_smooth_grad(x, eps=0.001):
    """Compute smooth area and its gradient."""
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    diff = gaps - w[None, :]
    soft_abs = np.sqrt(diff * diff + eps * eps)
    # d(soft_abs)/d(diff) = diff / soft_abs
    d_soft_abs = diff / (soft_abs + 1e-15)
    # soft_min = (gaps + w - soft_abs) / 2
    # d(soft_min)/d(gaps) = (1 - d_soft_abs) / 2
    d_soft_min = (1.0 - d_soft_abs) / 2.0  # shape (63, 20000)
    
    union_len = w + np.sum((gaps + w[None, :] - soft_abs) / 2.0, axis=0)
    area = np.mean(union_len)
    
    # Gradient: d(area)/d(x_i) = mean over y of sum_j d(soft_min_j)/d(L_j) * d(L_j)/d(x_i)
    # L_j depends on x through sorting, so this is complex.
    # For now, use numerical gradient as fallback, or use the structure.
    # Actually, let's just compute area and let scipy handle gradient numerically
    # but with a smoother objective.
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
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
    
    # Try heuristic starting points
    a = (np.arange(n) + 1) / n
    x_heur = -a * 0.5
    x_heur -= np.min(x_heur)
    area_heur = _compute_area(x_heur)
    
    if area_heur < best_area:
        best_area = area_heur
        best_x = x_heur.copy()
    
    # Try several beta values
    for beta in np.linspace(0.0, 1.0, 21):
        x_try = -beta * a
        x_try -= np.min(x_try)
        area_try = _compute_area(x_try)
        if area_try < best_area:
            best_area = area_try
            best_x = x_try.copy()
    
    # Phase 1: Optimize with smooth surrogate
    elapsed = time.time() - start_time
    if elapsed < 300.0:
        def objective_smooth(x):
            return _compute_area_smooth(x, eps=0.01)
        
        try:
            result = minimize(objective_smooth, best_x, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10,
                                     'maxfun': 50000})
            if result.fun < _compute_area(result.x):
                # Verify with exact area
                exact_area = _compute_area(result.x)
                if exact_area < best_area:
                    best_area = exact_area
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Refine with exact area using L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < 450.0:
        def objective_exact(x):
            return _compute_area(x)
        
        try:
            result = minimize(objective_exact, best_x, method='L-BFGS-B',
                            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                     'maxfun': 100000})
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead from multiple starting points
    elapsed = time.time() - start_time
    if elapsed < 520.0:
        rng = np.random.RandomState(42)
        starting_points = [best_x.copy()]
        for _ in range(5):
            starting_points.append(best_x + rng.randn(n) * 0.005)
        
        for sp in starting_points:
            if time.time() > deadline - 20.0:
                break
            try:
                result = minimize(_compute_area, sp, method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
                if result.fun < best_area:
                    best_area = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            result = minimize(_compute_area, best_x, method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14,
                                     'maxfun': 200000})
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
