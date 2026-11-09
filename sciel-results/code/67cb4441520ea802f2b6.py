# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Keich initial construction
    def keich_init():
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)
    
    x0 = keich_init()
    
    # Precompute constants
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    w_base = 1.0 / n
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M  # heights
    y_col = y_vals.reshape(-1, 1)  # (M, 1)
    w_vals = (1.0 - y_vals) / n  # (M,)
    
    def compute_area(x):
        """Compute exact union area."""
        # L[i, m] = x[i] + a[i] * y[m]
        L = x.reshape(-1, 1) + a.reshape(-1, 1) * y_col  # (n, M)
        # Sort along axis 0
        L_sorted = np.sort(L, axis=0)  # (n, M)
        # Differences between consecutive
        diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
        # min(diffs, w)
        w_row = w_vals.reshape(1, -1)  # (1, M)
        union_lengths = w_vals + np.sum(np.minimum(diffs, w_row), axis=0)  # (M,)
        return np.mean(union_lengths)
    
    def smooth_objective(x, t=0.01):
        """Smooth approximation of the area using soft-min."""
        L = x.reshape(-1, 1) + a.reshape(-1, 1) * y_col  # (n, M)
        L_sorted = np.sort(L, axis=0)  # (n, M)
        diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
        w_row = w_vals.reshape(1, -1)  # (1, M)
        # soft_min(d, w) = -t * log(exp(-d/t) + exp(-w/t))
        # For stability: soft_min(d, w) = min(d,w) - t*log(1 + exp(-|d-w|/t)) approximately
        # Actually: soft_min(d,w) = -t*log(exp(-d/t) + exp(-w/t))
        # = min(d,w) - t*log(1 + exp(-|d-w|/t))
        d_min_w = np.minimum(diffs, w_row)
        abs_diff = np.abs(diffs - w_row)
        # For numerical stability
        soft_min = d_min_w - t * np.log1p(np.exp(-abs_diff / t))
        union_lengths = w_vals + np.sum(soft_min, axis=0)
        return np.mean(union_lengths)
    
    def smooth_gradient(x, t=0.01, eps=1e-7):
        """Numerical gradient of smooth objective."""
        g = np.zeros(n)
        f0 = smooth_objective(x, t)
        for i in range(n):
            xp = x.copy()
            xp[i] += eps
            g[i] = (smooth_objective(xp, t) - f0) / eps
        return g
    
    best_x = x0.copy()
    best_area = compute_area(x0)
    
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Phase 1: L-BFGS-B with smooth objective
    try:
        res = minimize(
            smooth_objective, x0, args=(0.01,),
            jac=smooth_gradient, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            area = compute_area(res.x)
            if area < best_area:
                best_x = res.x.copy()
                best_area = area
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Phase 2: Basin hopping with L-BFGS-B polishing
    rng = np.random.default_rng(42)
    for iteration in range(200):
        if time.time() > deadline - 30:
            break
        
        # Perturb
        sigma = 0.1 / (1 + iteration * 0.1)
        x_perturbed = best_x + rng.normal(0, sigma, n)
        
        # Optimize from perturbed point
        try:
            res = minimize(
                smooth_objective, x_perturbed, args=(0.005,),
                jac=smooth_gradient, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                area = compute_area(res.x)
                if area < best_area:
                    best_x = res.x.copy()
                    best_area = area
        except Exception:
            pass
    
    # Phase 3: Final polish with tighter tolerance on exact objective
    if time.time() < deadline - 10:
        try:
            # Use exact objective with numerical gradient for final polish
            def exact_grad(x, eps=1e-8):
                g = np.zeros(n)
                f0 = compute_area(x)
                for i in range(n):
                    xp = x.copy()
                    xp[i] += eps
                    g[i] = (compute_area(xp) - f0) / eps
                return g
            
            res = minimize(
                compute_area, best_x,
                jac=exact_grad, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                area = compute_area(res.x)
                if area < best_area:
                    best_x = res.x.copy()
                    best_area = area
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
