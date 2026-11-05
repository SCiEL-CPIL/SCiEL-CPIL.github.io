import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    # Objective function
    # We want to minimize V = | 2 * N * max(conv) / (sum(h))^2 |
    # Since sum(h) is in denominator, we can fix sum(h) = 1 to simplify optimization?
    # No, the value scales as 1/S^2. If we scale h by k, V scales by 1/k.
    # Wait.
    # V(h) = 2N * max(conv(h)) / (sum(h))^2
    # conv(kh) = k^2 conv(h)
    # sum(kh) = k sum(h)
    # V(kh) = 2N * k^2 max(conv(h)) / (k^2 (sum(h))^2) = V(h)
    # The value is scale invariant!
    # So we can constrain sum(h) = 1.
    
    def objective(h):
        # h is normalized such that sum(h) is close to 1, but we don't enforce it strictly in obj
        # to allow the optimizer to explore. But scale invariance means we can fix it.
        # Let's fix sum(h) = 1 by projecting?
        # Or just compute V directly.
        s = np.sum(h)
        if np.abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        return val

    # Initializations
    candidates = []
    
    # 1. Constant (Seed)
    candidates.append(np.ones(N))
    
    # 2. Random Gaussian
    np.random.seed(42)
    candidates.append(np.random.randn(N))
    
    # 3. Sinusoid
    t = np.arange(N)
    candidates.append(np.cos(2 * np.pi * t / N))
    
    # 4. Comb
    h_comb = np.zeros(N)
    h_comb[::2] = 1.0
    candidates.append(h_comb)
    
    # 5. Linear Recurrence Perturbation (as per directive)
    # h[n] = A + sum c_i r_i^n
    # Let's try a simple AR(1) process: h[n] = rho * h[n-1] + noise
    # Or just a decaying exponential perturbation on baseline.
    h_base = np.ones(N)
    h_pert = np.exp(-0.1 * np.arange(N))
    candidates.append(h_base + 0.1 * h_pert)
    
    best_val = 1e10
    best_h = None
    
    for h0 in candidates:
        # Normalize initial guess to sum=1 to start in good region
        s0 = np.sum(h0)
        if np.abs(s0) < 1e-6:
            continue
        h0 = h0 / s0
        
        # Check time
        if time.time() > deadline - 5:
            break
            
        try:
            # Use L-BFGS-B
            # Bounds: None (unconstrained)
            res = minimize(objective, h0, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-8})
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            pass

    # If no optimization succeeded or found better, return seed
    if best_h is None:
        return np.ones(N)
    
    # Final check
    s = np.sum(best_h)
    if np.abs(s) < 1e-6:
        return np.ones(N)
        
    return best_h
