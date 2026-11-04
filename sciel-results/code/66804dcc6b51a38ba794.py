import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    # Safety margin: stop 10 seconds before the 150s limit
    deadline = start_time + 140.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def objective_and_grad(h):
        """
        Compute objective value and gradient.
        We assume the max of the convolution occurs at a specific lag k.
        To make this differentiable, we can either:
        1. Fix the lag k to the one that maximizes the convolution for the current h.
        2. Use a soft-max approximation (exp(t * c) / sum(exp(t * c))) but t is tricky.
        
        Let's use approach 1: Fix the lag k.
        The gradient of the objective with respect to h is complex because k depends on h.
        However, if we fix k, the gradient is straightforward.
        
        Objective: J = | 2 * N * max(c) / s^2 |
        
        Let's define a proxy objective that is differentiable:
        F(h) = 2 * N * c_k(h) / (sum(h))^2
        
        where k is the index of the maximum of c(h) evaluated at the current h.
        This is not strictly differentiable if k changes, but in practice, k is stable 
        in local neighborhoods. We can just compute the gradient assuming k is fixed.
        
        c_k = sum_i h_i * h_{i+k}  (for lag k, considering full convolution indices)
        Note: np.convolve(h, h) has length 2N-1.
        Index j in convolve corresponds to lag j - (N-1).
        Let L = j - (N-1).
        c_j = sum_i h_i * h_{i+L}  where indices are valid.
        
        Actually, it's easier to just use finite differences or a numerical gradient 
        if the analytic one is too messy, but N=400 is small enough for numerical 
        gradient if we do it efficiently? No, 400 evaluations per gradient step * 400 dims 
        is 160k operations per gradient, which is fine. But analytic is better.
        
        Let's try to derive the gradient for fixed k.
        Let S = sum(h).
        Let C_k = sum_i h_i * h_{i+k}  (with h_i=0 for i<0 or i>=N).
        Objective f = 2 * N * C_k / S^2
        
        df/dh_m = 2*N * ( (dC_k/dh_m) * S^2 - C_k * 2*S * (dS/dh_m) ) / S^4
        df/dh_m = 2*N * ( (dC_k/dh_m) / S - 2*C_k * (dS/dh_m) / S^2 )
        
        dS/dh_m = 1
        
        dC_k/dh_m:
        C_k = sum_i h_i h_{i+k}
        Term involving h_m:
        1. h_m * h_{m+k} (if m+k < N)
        2. h_{m-k} * h_m (if m-k >= 0)
        
        So dC_k/dh_m = h_{m+k} + h_{m-k} (with zero padding)
        
        So:
        grad_m = 2*N * ( (h_{m+k} + h_{m-k}) / S - 2*C_k / S^2 )
        
        This is valid if k is the true maximum. If k changes, the gradient is 
        discontinuous, but L-BFGS-B might handle it okay, or we can use a 
        soft-max to smooth it.
        
        Let's implement the soft-max version to be safer.
        c = convolve(h, h)
        t = 100  # temperature
        # Avoid overflow
        c_max = np.max(c)
        exp_c = np.exp((c - c_max) * t)
        Z = np.sum(exp_c)
        p = exp_c / Z  # Probabilities
        C_smooth = np.sum(p * c)  # This is c_max + log(Z)/t
        grad_C = t * (c * p - C_smooth * p)  # This is wrong.
        
        Actually, d/dc (sum p c) = sum p (since d p_i / d c_j is complex).
        Let's stick to the fixed k approach, it's simpler and usually works well 
        enough for these types of optimization problems where the max is unique 
        and stable.
        """
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(h)
        
        c = np.convolve(h, h)
        k_idx = np.argmax(c)
        k_lag = k_idx - (N - 1)
        
        C_k = c[k_idx]
        
        # Gradient of C_k with respect to h
        # dC_k/dh_m = h_{m+k_lag} + h_{m-k_lag}
        grad_C = np.zeros_like(h)
        
        # Contribution from h_m * h_{m+k_lag}
        if k_lag >= 0:
            # m + k_lag < N => m < N - k_lag
            valid_m = np.arange(0, N - k_lag)
            grad_C[valid_m] += h[valid_m + k_lag]
        else:
            # m - k_lag >= 0 => m >= -k_lag
            valid_m = np.arange(-k_lag, N)
            grad_C[valid_m] += h[valid_m + k_lag] # Note: h_{m+k_lag} where k_lag is negative, so h_{m-|k_lag|}
            
        # Contribution from h_{m-k_lag} * h_m
        if k_lag <= 0:
            # m - k_lag < N => m < N + k_lag
            valid_m = np.arange(0, N + k_lag)
            grad_C[valid_m] += h[valid_m - k_lag]
        else:
            # m - k_lag >= 0 => m >= k_lag
            valid_m = np.arange(k_lag, N)
            grad_C[valid_m] += h[valid_m - k_lag]
            
        # Wait, the convolution c_k = sum_i h_i h_{i+k}.
        # If k is positive, h_{i+k} is the second term.
        # My indexing above:
        # Term 1: h_m is the first factor. Pair with h_{m+k}.
        # Term 2: h_m is the second factor. Pair with h_{m-k}.
        
        # Let's re-verify:
        # C_k = sum_{i=0}^{N-1-k} h_i h_{i+k}  (if k > 0)
        # dC_k/dh_m:
        # If m is in [0, N-1-k], it appears as h_m in the first slot. Pair is h_{m+k}.
        # If m is in [k, N-1], it appears as h_m in the second slot (let i = m-k, so h_{m-k} h_m). Pair is h_{m-k}.
        
        # So:
        # grad_C[m] = 0
        # if m < N-k: grad_C[m] += h[m+k]
        # if m >= k: grad_C[m] += h[m-k]
        
        # My previous code:
        # if k_lag >= 0:
        #   valid_m = 0..N-k-1. grad_C[valid_m] += h[valid_m + k]. Correct.
        # else:
        #   valid_m = -k..N-1. grad_C[valid_m] += h[valid_m + k]. 
        #   Here k is negative. Let k' = -k. valid_m = k'..N-1. h[valid_m - k']. 
        #   This is h_{m-k'}. Correct.
        
        # if k_lag <= 0:
        #   valid_m = 0..N+k-1. grad_C[valid_m] += h[valid_m - k].
        #   k is negative. valid_m = 0..N-|k|-1. h[valid_m + |k|]. 
        #   This is h_{m+k'}. Correct.
        # else:
        #   valid_m = k..N-1. grad_C[valid_m] += h[valid_m - k]. Correct.
        
        # So the gradient calculation for C_k is correct.
        
        # Now, Objective f = 2*N*C_k / S^2
        # df/dh_m = 2*N * ( grad_C[m] * S - 2*C_k ) / S^3
        #          = 2*N/S * grad_C[m] - 4*N*C_k/S^2
        
        grad_f = (2.0 * N / s) * grad_C - (4.0 * N * C_k / (s * s))
        
        # The actual objective in compute_value uses abs().
        # If the value is negative, the gradient should be negated.
        val = 2.0 * N * C_k / (s * s)
        if val < 0:
            grad_f = -grad_f
            val = -val
            
        return val, grad_f

    # Initial best: Linear Ramp
    h_best = np.arange(1, N + 1, dtype=np.float64)
    val_best = compute_value(h_best)
    
    # Run scipy optimization
    # We might need to run multiple times from different starts
    
    starts = [
        np.arange(1, N + 1, dtype=np.float64),
        np.arange(1, N + 1, dtype=np.float64) ** 0.5,
        np.arange(1, N + 1, dtype=np.float64) ** 2,
        np.ones(N, dtype=np.float64) * 10, # Uniform
        np.sin(np.pi * np.arange(N) / N) + 1.0, # Sine
        np.exp(np.arange(N) / N), # Exponential
    ]
    
    for h0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxls': 50, 'disp': False}
            )
            h_res = res.x
            val_res = compute_value(h_res)
            if val_res < val_best:
                val_best = val_res
                h_best = h_res.copy()
        except Exception as e:
            pass

    return h_best
