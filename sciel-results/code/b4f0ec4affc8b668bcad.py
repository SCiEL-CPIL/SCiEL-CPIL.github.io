"""
Optimized solution for the third autocorrelation inequality task.

The key insight is that the autoconvolution at lag 0 is sum(h^2), and by
Cauchy-Schwarz, sum(h^2) >= (sum(h))^2 / N. However, the maximum of the
autoconvolution can be much smaller than sum(h^2) if the h values have
significant negative parts that cause destructive interference in the
autoconvolution at other lags.

The strategy: use a vector with a large DC offset (to make sum(h) large)
combined with an antisymmetric or oscillatory component (to make the
autoconvolution peaks at non-zero lags small or negative).

Specifically, if h = c + a where c is a constant and a is antisymmetric
(a_i = -a_{N-1-i}), then:
- sum(h) = N*c + sum(a) = N*c (since sum of antisymmetric is 0)
- The autoconvolution (h*h)_k involves cross terms that can be made small.

We use a gradient-based approach to minimize the objective directly.
"""
import numpy as np
import time

N = 400

def _evaluate(h):
    """Evaluate the objective: abs(2*N*max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _autoconv_max(h):
    """Compute the maximum of the autoconvolution."""
    return np.max(np.convolve(h, h))

def _objective(h):
    """The objective to minimize."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    return abs(2.0 * N * _autoconv_max(h) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Start with a good initial guess: DC offset + antisymmetric component
    # h = c * ones + a where a is antisymmetric
    # This makes sum(h) = N*c, and the autoconvolution can be controlled
    
    # Initial candidate: large DC + moderate antisymmetric
    best_h = None
    best_val = 1e10
    
    # Try several DC levels and antisymmetric patterns
    for c in [1.0, 2.0, 5.0, 10.0, 50.0]:
        # Antisymmetric pattern: sin with odd symmetry
        t = np.arange(N)
        a = np.sin(np.pi * (t + 0.5) / N)  # antisymmetric about center
        # Make it properly antisymmetric: a[i] = -a[N-1-i]
        a = a - a[::-1]  # ensure antisymmetry
        h = c * np.ones(N) + a * c * 0.5
        val = _evaluate(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Also try pure antisymmetric with small DC
    for c in [0.1, 0.5, 1.0, 2.0]:
        t = np.arange(N)
        # Various antisymmetric bases
        for freq in [1, 2, 3, 5]:
            a = np.sin(2 * np.pi * freq * (t + 0.5) / N)
            # Make antisymmetric
            a = a - a[::-1]
            h = c * np.ones(N) + a
            val = _evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Gradient descent refinement
    if best_h is None:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
    
    # Use scipy if available for better optimization
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        # Define the objective for scipy (need to handle the non-smooth max)
        # We'll use a soft-max approximation for differentiability
        def soft_max(x, beta=100.0):
            """Soft maximum approximation."""
            m = np.max(x)
            return m + np.log(np.sum(np.exp(x - m)) / len(x)) / beta
        
        def objective_scipy(h_flat):
            h = h_flat.reshape(N)
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h)
            # Use soft max for smoothness
            sm = soft_max(conv, beta=50.0)
            return abs(2.0 * N * sm / (s * s))
        
        # Try multiple starting points
        for start_idx in range(5):
            if time.time() > deadline - 20.0:
                break
            
            if start_idx == 0:
                h0 = best_h.copy()
            else:
                # Perturbed version
                h0 = best_h + rng.standard_normal(N) * 0.1 * np.std(best_h)
            
            try:
                result = minimize(
                    objective_scipy,
                    h0.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                h_opt = result.x.reshape(N)
                val_opt = _evaluate(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Custom gradient descent with momentum
    if time.time() < deadline - 15.0:
        # Use a subgradient approach for the max function
        h_curr = best_h.copy()
        lr = 0.01
        momentum = np.zeros(N)
        beta_m = 0.9
        
        for iteration in range(2000):
            if time.time() > deadline - 10.0:
                break
            
            s = np.sum(h_curr)
            if abs(s) < 1e-10:
                break
            
            conv = np.convolve(h_curr, h_curr)
            max_idx = np.argmax(conv)
            
            # Gradient of max(conv) w.r.t. h
            # conv[k] = sum_i h[i] * h[i+k] (with appropriate indexing for full conv)
            # For full convolve, conv[k] = sum_i h[i] * h[i-(k-N+1)] for k in 0..2N-2
            # The gradient of conv[k] w.r.t. h is a shifted version of h
            
            # For the max at index max_idx:
            # grad = 2 * h shifted appropriately
            # In full convolve, conv[k] = sum_{i} h[i] * h[i - (k - N + 1)]
            # Let shift = k - N + 1
            # conv[k] = sum_i h[i] * h[i - shift]
            # d(conv[k])/d(h[j]) = h[j-shift] + h[j+shift] (if in range)
            
            shift = max_idx - N + 1
            
            grad_conv = np.zeros(N)
            for j in range(N):
                val = 0.0
                if 0 <= j - shift < N:
                    val += h_curr[j - shift]
                if 0 <= j + shift < N:
                    val += h_curr[j + shift]
                grad_conv[j] = val
            
            # Objective: abs(2*N*max_conv / s^2)
            # For small changes, gradient of f = 2*N*max_conv/s^2 (ignoring sign)
            # df/dh[j] = 2*N/s^2 * grad_conv[j] - 2*N*max_conv/s^3 * 2*s * 1
            #          = 2*N/s^2 * grad_conv[j] - 4*N*max_conv/s^2
            #          = 2*N/s^2 * (grad_conv[j] - 2*max_conv)
            
            max_conv = conv[max_idx]
            grad_obj = (2.0 * N / (s * s)) * (grad_conv - 2.0 * max_conv)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad_obj)
            if grad_norm > 1e-10:
                grad_obj = grad_obj / grad_norm
            
            # Momentum update
            momentum = beta_m * momentum + (1 - beta_m) * grad_obj
            h_curr = h_curr - lr * momentum
            
            # Project: keep sum(h) away from zero
            s_new = np.sum(h_curr)
            if abs(s_new) < 0.1:
                h_curr += (0.1 - s_new) / N
            
            val_new = _evaluate(h_curr)
            if val_new < best_val:
                best_val = val_new
                best_h = h_curr.copy()
                lr *= 1.1  # increase learning rate slightly on improvement
            else:
                lr *= 0.95  # decrease learning rate
            
            # Reset if getting worse
            if val_new > best_val * 2:
                h_curr = best_h + rng.standard_normal(N) * 0.05 * np.std(best_h)
                momentum = np.zeros(N)
                lr = 0.01
    
    # Final local search with random perturbations
    if time.time() < deadline - 5.0:
        for _ in range(500):
            if time.time() > deadline - 3.0:
                break
            h_new = best_h + rng.standard_normal(N) * 0.01 * np.std(best_h)
            val_new = _evaluate(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    
    # Safety check
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
    
    return best_h
