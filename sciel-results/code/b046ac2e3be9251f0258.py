import numpy as np
import time

def _compute_area_exact(positions, n=64, n_heights=20000):
    """Compute the exact union area using the given quadrature formula."""
    a = (np.arange(n) + 1) / n  # a_i for i=0..n-1
    y = (np.arange(n_heights) + 0.5) / n_heights  # heights
    w = (1.0 - y) / n  # width at each height, shape (n_heights,)
    
    # L_i at each height: L[i, m] = x_i + a_i * y_m
    # positions shape (n,), a shape (n,), y shape (n_heights,)
    L = positions[:, None] + a[:, None] * y[None, :]  # shape (n, n_heights)
    
    # Sort L along axis 0 for each height
    L_sorted = np.sort(L, axis=0)  # shape (n, n_heights)
    
    # Gaps between consecutive sorted L's
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, n_heights)
    
    # Union length at each height: w + sum(min(gaps, w))
    min_gaps = np.minimum(gaps, w[None, :])  # shape (n-1, n_heights)
    union_lengths = w + min_gaps.sum(axis=0)  # shape (n_heights,)
    
    area = union_lengths.mean()
    return area


def _compute_area_smooth(positions, n=64, n_heights=20000, epsilon=0.001):
    """Compute a smooth approximation of the union area using soft-min."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = positions[:, None] + a[:, None] * y[None, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Soft-min: min(gaps, w) ≈ -epsilon * log(exp(-gaps/epsilon) + exp(-w/epsilon))
    # For numerical stability: min(a,b) = -eps*log(exp(-a/eps) + exp(-b/eps))
    # = max(a,b) - eps*log(1 + exp(-(max-min)/eps))
    # Actually: min(a,b) = -eps * log(exp(-a/eps) + exp(-b/eps))
    # Let M = max(a,b), then min(a,b) = M - eps*log(1 + exp(-(M-m)/eps)) where m=min(a,b)
    # But that's circular. Let's use: min(a,b) = -eps*log(exp(-a/eps) + exp(-b/eps))
    # For stability: let c = max(-a/eps, -b/eps), then min = -eps*(c + log(exp(-a/eps-c) + exp(-b/eps-c)))
    
    a_term = gaps / epsilon
    b_term = w[None, :] / epsilon
    
    # soft_min = -epsilon * log(exp(-a_term) + exp(-b_term))
    # For numerical stability:
    c = np.maximum(a_term, b_term)
    soft_min = -epsilon * (c + np.log(np.exp(-a_term - c) + np.exp(-b_term - c)))
    
    union_lengths = w + soft_min.sum(axis=0)
    area = union_lengths.mean()
    return area


def _keich_positions(n=64):
    """Generate Keich construction positions."""
    k = int(np.log2(n))
    positions = np.zeros(n)
    for i in range(n):
        bits = format(i, f'0{k}b')
        eps = np.array([int(b) for b in bits], dtype=float)
        j = np.arange(1, k + 1)
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return positions


def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit, use 630 as hard stop
    
    n = 64
    best_positions = _keich_positions(n)
    best_area = _compute_area_exact(best_positions)
    
    # Try smooth optimization with decreasing epsilon
    epsilons = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for eps in epsilons:
        if time.time() > deadline - 60:  # 60s safety margin
            break
        
        # Use Nelder-Mead for local optimization
        from scipy.optimize import minimize
        
        def objective(x):
            if time.time() > deadline - 30:
                return best_area  # Return current best to stop
            return _compute_area_smooth(x, n=n, epsilon=eps)
        
        try:
            result = minimize(
                objective,
                best_positions,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_area:
                best_positions = result.x.copy()
                best_area = _compute_area_exact(best_positions)
        except Exception:
            pass
    
    # Final polish with exact area and smaller steps
    if time.time() < deadline - 30:
        from scipy.optimize import minimize
        
        def objective_exact(x):
            if time.time() > deadline - 15:
                return best_area
            return _compute_area_exact(x)
        
        try:
            result = minimize(
                objective_exact,
                best_positions,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.fun < best_area:
                best_positions = result.x.copy()
                best_area = _compute_area_exact(best_positions)
        except Exception:
            pass
    
    return np.array(best_positions, dtype=np.float64)
