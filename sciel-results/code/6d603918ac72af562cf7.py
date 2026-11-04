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
        
        Given the nature of autocorrelation for smooth functions, the max is often at lag 0 (if h is positive) 
        or near 0. However, h can be negative.
        
        Let's look at the structure: 
        value = 2*N * max(c) / S^2.
        
        If we fix the lag k* that gives the max c_k*, the objective is f(h) = 2*N * c_{k*}(h) / S(h)^2.
        The gradient w.r.t h_i is:
        d(c_k)/dh_i = h_{k-i} + h_{k-i+N-1} (depending on indexing, but essentially convolution terms)
        Actually, c_k = sum_j h_j h_{j+k}.
        d(c_k)/dh_i = h_{i-k} (if i-k in range) + h_{i+k} (if i+k in range) ... wait.
        c_k = sum_{j} h_j h_{j+k}.
        If we vary h_i, it appears as h_i * h_{i+k} (if i+k valid) and h_{i-k} * h_i (if i-k valid).
        So d(c_k)/dh_i = h_{i-k} + h_{i+k} (with bounds check).
        
        d(S^2)/dh_i = 2 S.
        
        So grad f_i = (2N * (h_{i-k} + h_{i+k}) * S^2 - 2N * c_k * 2 S) / S^4
                    = 2N * (h_{i-k} + h_{i+k}) / S - 4N * c_k / S^2
                    = (2N/S) * (h_{i-k} + h_{i+k} - 2 * c_k / S)
        
        This gradient is valid ONLY if k remains the argmax. If the argmax switches, the function is not differentiable.
        However, L-BFGS-B uses line search and might handle small jumps or we can just accept that it's a heuristic.
        A safer approach for the optimizer: 
        Since N=400 is small, we can compute the full gradient by checking which lag is max, 
        and if multiple lags are close, it's noisy. 
        But for a first try, let's assume the argmax is stable enough or use the gradient of the max term directly.
        
        Alternative: Use finite differences? Too slow for 400 vars * many steps.
        
        Let's implement the analytical gradient assuming the current argmax k* is the one to differentiate.
        """
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, np.zeros_like(h)
        
        c = np.convolve(h, h)
        k_max = np.argmax(c)
        c_max = c[k_max]
        
        val = abs(2.0 * N * c_max / (s * s))
        
        # Calculate gradient of c_k w.r.t h
        # c_k = sum_{j=0}^{N-1-k} h_j h_{j+k}
        # d(c_k)/dh_i = sum_{j} (delta_{j,i} h_{j+k} + h_j delta_{j+k,i})
        # Term 1: j=i -> h_{i+k} (if i+k < N)
        # Term 2: j+k=i -> j=i-k -> h_{i-k} (if i-k >= 0)
        
        grad_c = np.zeros(N)
        # h_{i+k}
        if k_max > 0:
            grad_c[:N-k_max] += h[k_max:]
        # h_{i-k}
        if k_max > 0:
            grad_c[k_max:] += h[:N-k_max]
        # If k_max == 0:
        # c_0 = sum h_i^2. d/dh_i = 2 h_i.
        # The formula above: h_{i+0} + h_{i-0} = 2h_i. Correct.
        
        # f = 2N * c_max / s^2
        # df/dh_i = 2N * ( (d c_max / dh_i) * s^2 - c_max * 2 s * 1 ) / s^4
        #         = 2N * ( d c_max / dh_i * s - 2 * c_max ) / s^2
        #         = (2N/s^2) * ( s * d c_max / dh_i - 2 * c_max )
        
        grad_val = (2.0 * N / (s * s)) * (s * grad_c - 2.0 * c_max)
        
        # Note: The objective is abs(...). 
        # If 2*N*c_max/s^2 is positive, grad is as above.
        # If negative, grad is -grad_val.
        # Since we want to minimize the absolute value, we minimize the magnitude.
        # Usually c_max > 0 for typical signals. If c_max < 0, the value is negative inside abs, 
        # so we want to make it less negative (increase it).
        # Let's just return the gradient of the signed quantity q = 2*N*c_max/s^2.
        # If we minimize |q|, the gradient is sign(q) * grad(q).
        # But L-BFGS-B minimizes the function. If we pass |q|, the gradient is sign(q)*grad(q).
        # However, |q| is not differentiable at q=0.
        # Let's assume we are far from 0 (score ~1.1 means value ~1.8, so q is likely positive or negative but not 0).
        # Actually, if q is positive, we minimize q. If q is negative, we maximize q (minimize -q).
        # Let's just return the gradient of q, and let the optimizer handle the sign? 
        # No, minimize expects the gradient of the function being minimized.
        # If we define the function to minimize as q (signed), we might drive q to -infinity if possible?
        # But q is bounded? 
        # Let's check the seed: h=1, c_max=N, s=N, q = 2*N*N/N^2 = 2.
        # If we flip signs, c_max might become negative? 
        # If h is all -1, c_max = N, s = -N, s^2 = N^2. q = 2.
        # If h is [1, -1, 1, -1], c_0 = N, c_1 = -N. max is N.
        # It seems c_max is often positive.
        # Let's assume we minimize q = 2*N*c_max/s^2.
        # If q becomes negative, the abs value increases? No, abs(-5) = 5.
        # If q is negative, we want to increase it towards 0.
        # So if q < 0, we should minimize -q.
        # Gradient of -q is -grad_q.
        # So: if q > 0, return grad_q. If q < 0, return -grad_q.
        
        if c_max > 0:
            return val, grad_val
        else:
            return val, -grad_val

    # Initial best
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    # Try a few different initializations for L-BFGS-B
    rng = np.random.default_rng(42)
    
    # Initialization 1: Random noise
    # Initialization 2: Ramp
    # Initialization 3: Sine wave
    
    initializations = [
        np.ones(N),
        np.arange(1, N+1, dtype=np.float64),
        np.sin(np.linspace(0, 2*np.pi, N)),
        rng.normal(0, 1, N),
        rng.uniform(-1, 1, N)
    ]
    
    for h_init in initializations:
        if time.time() > deadline:
            break
            
        # Scale initialization to have similar magnitude
        # The objective is scale invariant?
        # value = 2N * max(c) / s^2.
        # If h -> k*h, c -> k^2 c, s -> k s, s^2 -> k^2 s^2.
        # value is invariant to scaling.
        # So we can fix the scale, e.g., sum(h) = N or norm(h) = 1.
        # L-BFGS-B might struggle if scale is arbitrary.
        # Let's normalize h_init to sum = N.
        s_init = np.sum(h_init)
        if abs(s_init) > 1e-9:
            h_init = h_init * (N / s_init)
        else:
            h_init = h_init + 1e-6 # avoid zero sum
            
        try:
            # L-BFGS-B is good for large N
            res = minimize(
                objective_and_grad, 
                h_init, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'gtol': 1e-5, 'disp': False}
            )
            
            h_candidate = res.x
            val_candidate = compute_value(h_candidate)
            
            if val_candidate < val_best:
                val_best = val_candidate
                h_best = h_candidate.copy()
                
        except Exception:
            pass

    # If time remains, do a quick local refinement around the best found
    if time.time() < deadline - 20:
        # Use the best found as start
        h_refine = h_best.copy()
        # Normalize
        s_ref = np.sum(h_refine)
        if abs(s_ref) > 1e-9:
            h_refine = h_refine * (N / s_ref)
        
        try:
            res = minimize(
                objective_and_grad, 
                h_refine, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'gtol': 1e-8, 'disp': False}
            )
            h_candidate = res.x
            val_candidate = compute_value(h_candidate)
            if val_candidate < val_best:
                val_best = val_candidate
                h_best = h_candidate.copy()
        except Exception:
            pass

    return h_best
