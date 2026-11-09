# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Direct optimization on the probability simplex (100 dimensions).
    2. Use L-BFGS-B with analytical gradients.
    3. Multi-start: Run optimization from multiple diverse initializations 
       (uniform, diagonal, off-diagonal, random sparse, random dense) to 
       avoid local maxima.
    4. Time-boxed: Stop after a safety margin and return the best found.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 600s limit
    
    # Grid coordinates: P[y, x] -> y is row (0..9), x is col (0..9)
    ys, xs = np.mgrid[0:n, 0:n]
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    
    # Linear forms for entropy calculations
    d_val = (xs_flat - ys_flat)
    m0_val = xs_flat
    m1_val = (xs_flat + ys_flat)
    m2_val = (xs_flat + 2 * ys_flat)
    
    # Unique values and inverse mapping for bincount
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    # Precompute bincount weights matrices (sparse-like via dense for speed in small n)
    # M_d: shape (19, 100) where M_d[i, j] = 1 if d_inv[j] == i
    # Actually, bincount is faster, but for gradients we need to know which cells contribute to which bin.
    # Let's use the inverse mapping to construct gradient contributions.
    
    def compute_value_and_grad(p_flat):
        """
        p_flat: shape (100,) normalized probability vector
        Returns: (value, gradient) where value = D/M, gradient is d(value)/d(p_flat)
        """
        # Clip to ensure numerical stability for log
        p_safe = np.maximum(p_flat, 1e-18)
        
        # Calculate bin probabilities
        p_d = np.bincount(d_inv, weights=p_safe, minlength=len(d_vals))
        p_m0 = np.bincount(m0_inv, weights=p_safe, minlength=len(m0_vals))
        p_m1 = np.bincount(m1_inv, weights=p_safe, minlength=len(m1_vals))
        p_m2 = np.bincount(m2_inv, weights=p_safe, minlength=len(m2_vals))
        
        # Shannon Entropy H(p) = -sum(p * log2(p))
        # Gradient of H w.r.t p_flat:
        # dH/dp_flat[j] = -1/ln(2) * (log2(p_bin) + 1) if j belongs to bin
        
        def entropy_and_grad(p_dist, inv_map, num_bins):
            # p_dist: (num_bins,)
            # inv_map: (100,)
            # Returns H and grad_H (100,)
            p_nonzero = p_dist > 1e-18
            H = 0.0
            if np.any(p_nonzero):
                H = -np.sum(p_dist[p_nonzero] * np.log2(p_dist[p_nonzero]))
            
            # Gradient of H w.r.t p_dist
            # dH/dp_dist[i] = -1/ln(2) * (log2(p_dist[i]) + 1)
            grad_p_dist = np.zeros(num_bins)
            if np.any(p_nonzero):
                grad_p_dist[p_nonzero] = -(np.log2(p_dist[p_nonzero]) + 1.0) / np.log(2)
            
            # Map back to p_flat
            # grad_H[j] = grad_p_dist[inv_map[j]]
            grad_H = grad_p_dist[inv_map]
            return H, grad_H

        H_d, grad_H_d = entropy_and_grad(p_d, d_inv, len(d_vals))
        H_m0, grad_H_m0 = entropy_and_grad(p_m0, m0_inv, len(m0_vals))
        H_m1, grad_H_m1 = entropy_and_grad(p_m1, m1_inv, len(m1_vals))
        H_m2, grad_H_m2 = entropy_and_grad(p_m2, m2_inv, len(m2_vals))
        
        M = max(H_m0, H_m1, H_m2)
        
        if M < 1e-9:
            # Invalid or singular, return large negative value and zero gradient
            return -1e9, np.zeros_like(p_flat)
        
        value = H_d / M
        
        # Gradient of value = (M * grad_H_d - H_d * grad_M) / M^2
        # grad_M is the gradient of the max function.
        # Since M = max(H_m0, H_m1, H_m2), the gradient is the gradient of the active term(s).
        # If multiple are equal, we can take the sum or average. For L-BFGS-B, taking the one with max value is standard subgradient.
        # To be smooth, we could use a soft-max, but standard max is usually fine for L-BFGS-B if we handle ties.
        
        if H_m0 >= H_m1 and H_m0 >= H_m2:
            grad_M = grad_H_m0
        elif H_m1 >= H_m0 and H_m1 >= H_m2:
            grad_M = grad_H_m1
        else:
            grad_M = grad_H_m2
            
        # If there are ties, the gradient is not unique. We pick one. 
        # A more robust approach for optimization is to sum gradients of all active maxes if we want to stay in the "middle", 
        # but standard practice for max is to pick one. Let's stick to one for simplicity and speed.
        
        grad_value = (M * grad_H_d - H_d * grad_M) / (M * M)
        
        return value, grad_value

    def objective_neg(p_flat):
        """
        Objective for minimization: -D/M
        """
        val, grad = compute_value_and_grad(p_flat)
        return -val, -grad

    # Generate diverse initializations
    rng = np.random.default_rng(42)
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(100) / 100.0)
    
    # 2. Diagonal (x=y)
    diag = np.zeros(100)
    diag[np.arange(10)*10 + np.arange(10)] = 1.0/10.0
    initializations.append(diag)
    
    # 3. Off-diagonal (x!=y)
    off_diag = np.ones(100)
    off_diag[np.arange(10)*10 + np.arange(10)] = 0.0
    initializations.append(off_diag / np.sum(off_diag))
    
    # 4. Top-Left block
    block = np.zeros(100)
    block[0:25] = 1.0
    initializations.append(block / 25.0)
    
    # 5-10. Random sparse (few non-zeros)
    for _ in range(6):
        p = np.zeros(100)
        k = rng.integers(5, 20)
        idx = rng.choice(100, k, replace=False)
        p[idx] = 1.0
        initializations.append(p / k)
        
    # 11-15. Random dense
    for _ in range(5):
        p = rng.random(100)
        initializations.append(p / np.sum(p))
        
    # 16-20. Structured patterns (e.g., stripes)
    for slope in [-1, 0, 1, 2, -2]:
        p = np.zeros(100)
        # Add weight to cells where x + slope*y is close to a target
        # Simple: just use the linear form values as weights
        vals = xs_flat + slope * ys_flat
        p = np.exp(vals)
        initializations.append(p / np.sum(p))

    best_p = None
    best_value = -np.inf
    
    # Optimize from each initialization
    for i, p0 in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                objective_neg, 
                p0, 
                method='L-BFGS-B', 
                jac=True, # We provide the gradient
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 20}
            )
            
            if res.success or res.fun < -best_value:
                val, _ = compute_value_and_grad(res.x)
                if val > best_value:
                    best_value = val
                    best_p = res.x.copy()
                    
        except Exception:
            continue
            
    # If no valid result was found, fallback to uniform
    if best_p is None:
        best_p = np.ones(100) / 100.0
        best_value = compute_value_and_grad(best_p)[0]
        
    # Final check and normalization
    P = best_p.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
