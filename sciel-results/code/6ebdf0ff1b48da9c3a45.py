"""
Strategy: Full 400-dim direct optimization with differentiable softmax surrogate for max,
iterative reweighting, and periodic exact restarts.

The objective is:
    value = abs(2 * N * max(conv(h,h)) / sum(h)^2)

We minimize this by:
1. Using a softmax approximation: max(x) ≈ (1/T) * log(sum(exp(T*x))) for large T
   This makes the objective differentiable w.r.t. h.
2. Computing gradients via automatic differentiation (manual, since conv is linear).
3. Iteratively increasing T (annealing) to sharpen the max approximation.
4. Periodically evaluating the exact objective and using it for restarts.
5. Adding a small regularization to keep h well-conditioned.

The gradient of the objective w.r.t. h can be computed analytically:
- Let c = conv(h, h), M = max(c) (or softmax approx)
- dM/dh = 2 * conv(grad_M, h) where grad_M is the gradient of M w.r.t. c
- For exact max: grad_M[i] = 1 if c[i] == M, else 0
- For softmax: grad_M[i] = exp(T*c[i]) / sum(exp(T*c))

Then the full gradient of value w.r.t. h is computed via chain rule.
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0  # wall-clock limit with safety margin

def _exact_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _softmax_max_and_grad(c, T):
    """
    Compute softmax approximation of max and its gradient w.r.t. c.
    Returns (max_approx, grad_wrt_c)
    """
    # Numerical stability: subtract max
    c_shifted = c - np.max(c)
    exp_c = np.exp(T * c_shifted)
    sum_exp = np.sum(exp_c)
    if sum_exp == 0 or not np.isfinite(sum_exp):
        return np.max(c), np.zeros_like(c)
    grad = T * exp_c / sum_exp
    max_approx = np.max(c) + (1.0 / T) * np.log(sum_exp)
    return max_approx, grad

def _objective_and_grad(h, T):
    """
    Compute the surrogate objective and its gradient w.r.t. h.
    Uses softmax approximation for the max of autoconvolution.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    M, grad_M = _softmax_max_and_grad(conv, T)
    
    # value = abs(2*N*M / s^2)
    # For simplicity, assume 2*N*M/s^2 > 0 (we'll handle sign if needed)
    # d(value)/dM = 2*N/s^2 (if positive) or -2*N/s^2 (if negative)
    # d(value)/ds = -2 * 2*N*M / s^3 = -4*N*M/s^3
    
    val = abs(2.0 * N * M / (s * s))
    
    # Gradient of M w.r.t. h:
    # M = sum_i grad_M[i] * c[i], c = conv(h,h)
    # dM/dh[j] = sum_i grad_M[i] * dc[i]/dh[j]
    # dc[i]/dh[j] = h[i-j] + h[j-i] (for valid indices in full conv)
    # More precisely: c[k] = sum_j h[j] * h[k-j]
    # dc[k]/dh[j] = h[k-j] + h[j-k] (when both terms exist)
    # This is equivalent to: grad_M_convolved = conv(grad_M, h) + conv(h, grad_M)
    # But since conv is commutative for real signals: = 2 * conv(grad_M, h)
    # Wait, let me be more careful.
    # c[k] = sum_j h[j] * h[k-j]
    # dc[k]/dh[j] = h[k-j] + (if j' = k-j exists) h[k-j'] ... 
    # Actually: dc[k]/dh[j] = h[k-j] + h[j-k] ... no.
    # c[k] = sum_j h[j] h[k-j]
    # dc[k]/dh[j] = h[k-j] + sum_{j'} [if j'=k-j then 1 else 0] ... 
    # Let me just use: the gradient of conv(h,h) w.r.t. h is 2*conv(h, grad) where grad is the gradient w.r.t. the conv output.
    # Actually for c = conv(h, h), dc/dh = conv(h, grad_c) + conv(grad_c, h) = 2*conv(h, grad_c)
    # But wait, conv is not exactly symmetric in this sense for the gradient.
    # Let's think: c[k] = sum_j h[j] h[k-j]
    # dc[k]/dh[m] = h[k-m] + h[m-k] ... no, that's not right either.
    # dc[k]/dh[m] = sum_j [delta_{j,m} h[k-j] + h[j] delta_{k-j,m}]
    #             = h[k-m] + h[k-m] ... 
    # Wait: first term: j=m gives h[k-m]
    # Second term: k-j=m means j=k-m, gives h[k-m]
    # So dc[k]/dh[m] = 2*h[k-m] if we're careful about boundaries, but in full convolution:
    # Actually for full convolution of length 2N-1:
    # c[k] = sum_j h[j] h[k-j] for k=0..2N-2, j=0..N-1, k-j=0..N-1
    # dc[k]/dh[m] = h[k-m] (from first factor) + h[k-m] (from second factor, when j=k-m)
    # But we need k-m in [0,N-1] for both.
    # The clean way: dc/dh[m] = sum_k grad_c[k] * dc[k]/dh[m]
    #                  = sum_k grad_c[k] * (h[k-m] + h[m-k] if valid)
    # This is equivalent to: (conv(grad_c, h))[m] + (conv(h, grad_c))[m] with appropriate handling
    # For real signals, conv(a,b) = conv(b,a), so this is 2*conv(grad_c, h) but with the understanding
    # that we need to handle the full convolution properly.
    
    # Let's just compute it directly using the convolution property:
    # The Jacobian-vector product: for a perturbation v in h,
    # d<grad_c, c>/dv = 2 * <grad_c, conv(h, v)> = 2 * <conv(grad_c, h), v>
    # So dM/dh = 2 * conv(grad_M, h)
    
    dM_dh = 2.0 * np.convolve(grad_M, h, mode='full')
    # dM_dh has length 2N-1, but h has length N. We need to extract the N-length part.
    # The gradient w.r.t. h[j] is the sum over k of grad_c[k] * dc[k]/dh[j]
    # which corresponds to the central N elements of conv(grad_M, h) * 2? 
    # Let me verify: if h has length N, conv(grad_M, h) has length (2N-1)+(N-1) = 3N-2? No.
    # grad_M has length 2N-1, h has length N. conv(grad_M, h) has length (2N-1)+N-1 = 3N-2.
    # That's not right for a gradient. Let me reconsider.
    
    # Actually, the proper way: M = sum_k grad_M[k] * c[k] where c = conv(h,h)
    # dM/dh[j] = sum_k grad_M[k] * dc[k]/dh[j]
    # dc[k]/dh[j] = h[k-j] + h[j-k] (with the convention that h[m]=0 for m outside [0,N-1])
    # = h[k-j] + h[k-j] ... no.
    
    # Let me just compute it numerically via the FFT approach for efficiency:
    # Use the fact that for c = conv(h, h), the gradient w.r.t. h is:
    # grad_h = ifft(fft(h) * conj(fft(h)) * something) ... this is getting complicated.
    
    # Simplest correct approach: 
    # dM/dh[j] = 2 * sum_k grad_M[k] * h[k-j] (with h[m]=0 for m not in [0,N-1])
    # This is 2 * (grad_M * h)[j] where * denotes cross-correlation, or equivalently:
    # = 2 * conv(grad_M, h)[j + (N-1)] ... let me just compute it directly.
    
    # Direct computation:
    # dM_dh[j] = 2 * sum_{k} grad_M[k] * h[k-j] for k-j in [0, N-1]
    # This is the cross-correlation of grad_M and h, evaluated at lag j.
    # np.correlate(grad_M, h, mode='full')[N-1 + j] ... 
    
    # Actually, let's just use a simpler but correct method:
    # The autoconvolution c = conv(h, h). The gradient of sum_k g[k]*c[k] w.r.t. h is:
    # 2 * conv(g, h) truncated to length N (taking the appropriate N samples)
    # Specifically, if g has length 2N-1 and h has length N:
    # The result of conv(g, h) has length 3N-2. The gradient w.r.t. h[j] is:
    # sum_k g[k] * (dc[k]/dh[j])
    # dc[k]/dh[j] = h[k-j] if 0<=k-j<N else 0, plus h[k-j] if ... 
    
    # I'll use the FFT-based approach for clarity and correctness:
    # For c = conv(h, h), we have C = FFT(h) * FFT(h) (with zero-padding)
    # dM/dh = IFFT(conj(FFT(h)) * FFT(2*grad_M_padded)) ... 
    
    # Let me just do it the straightforward way with np.convolve and slicing:
    # grad_M has length 2N-1. We want dM/dh[j] for j=0..N-1.
    # dM/dh[j] = sum_k grad_M[k] * dc[k]/dh[j]
    # dc[k]/dh[j] = h[k-j] (if 0<=k-j<N) + h[k-j] (if 0<=k-j<N, from the other term)
    # Wait, c[k] = sum_i h[i]*h[k-i], so dc[k]/dh[j] = h[k-j] (from i=j) + h[k-j] (from k-i=j, i.e., i=k-j)
    # So dc[k]/dh[j] = 2*h[k-j] if 0<=k-j<N, else 0 (with boundary corrections)
    # Actually it's h[k-j] + h[k-j] = 2*h[k-j] only when both i=j and i=k-j are valid,
    # which is when 0<=j<N and 0<=k-j<N. But these are the same condition.
    # However, when j = k-j, i.e., k=2j, we get 2*h[j] (the same term counted twice? No, 
    # c[k] = sum_i h[i]h[k-i], the term i=j gives h[j]h[k-j], and i=k-j gives h[k-j]h[j],
    # which are the same term if j=k-j. So actually dc[k]/dh[j] = h[k-j] + h[k-j] = 2*h[k-j]
    # always (when valid), because the two contributions are from the two factors of h in the product.
    
    # So: dM/dh[j] = 2 * sum_{k: 0<=k-j<N} grad_M[k] * h[k-j]
    # Let m = k-j, so k = j+m, m in [0, N-1]:
    # dM/dh[j] = 2 * sum_{m=0}^{N-1} grad_M[j+m] * h[m]
    # This is 2 * (cross-correlation of grad_M and h at lag j)
    # = 2 * np.correlate(grad_M, h, mode='full')[N-1+j] ... 
    
    # np.correlate(a, b, mode='full')[i] = sum_k a[i+k] * b[k] for appropriate ranges
    # Let's just compute it directly for correctness:
    
    dM_dh = np.zeros(N)
    for j in range(N):
        # sum over m=0..N-1 of grad_M[j+m] * h[m], where j+m < 2N-1
        max_m = min(N, 2*N-1 - j)
        if max_m > 0:
            dM_dh[j] = 2.0 * np.dot(grad_M[j:j+max_m], h[:max_m])
    
    # Now the full gradient of value = |2*N*M/s^2| w.r.t. h:
    # Assuming 2*N*M/s^2 > 0 (we'll check and handle sign):
    # d(value)/dh = (2*N/s^2) * dM_dh - (4*N*M/s^3) * 1  (the -1 is d(s)/dh = 1 for each component)
    
    ratio = 2.0 * N * M / (s * s)
    sign_ratio = 1.0 if ratio >= 0 else -1.0
    
    grad_val = sign_ratio * (2.0 * N / (s * s)) * dM_dh - sign_ratio * (4.0 * N * M / (s**3)) * np.ones(N)
    
    return val, grad_val

def _exact_grad(h):
    """Compute exact gradient using subgradient of max (uniform over argmax)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf, np.zeros(N)
    
    conv = np.convolve(h, h)
    M = np.max(conv)
    # Subgradient: uniform over indices where conv == M
    mask = (conv == M)
    n_max = np.sum(mask)
    if n_max == 0:
        return np.inf, np.zeros(N)
    grad_M = mask.astype(float) / n_max
    
    # dM/dh[j] = 2 * sum_{m=0}^{N-1} grad_M[j+m] * h[m]
    dM_dh = np.zeros(N)
    for j in range(N):
        max_m = min(N, 2*N-1 - j)
        if max_m > 0:
            dM_dh[j] = 2.0 * np.dot(grad_M[j:j+max_m], h[:max_m])
    
    ratio = 2.0 * N * M / (s * s)
    sign_ratio = 1.0 if ratio >= 0 else -1.0
    
    grad_val = sign_ratio * (2.0 * N / (s * s)) * dM_dh - sign_ratio * (4.0 * N * M / (s**3)) * np.ones(N)
    
    return abs(ratio), grad_val

def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    Uses full 400-dim gradient-based optimization with softmax surrogate.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initialize with ones (seed)
    best_h = np.ones(N, dtype=float)
    best_val = _exact_value(best_h)  # Should be 2.0
    
    # Also try a few diverse initializations
    rng = np.random.default_rng(42)
    inits = [
        np.ones(N),
        rng.normal(1.0, 0.3, N),
        np.ones(N) * np.exp(-0.5 * ((np.arange(N) - N/2) / (N/4))**2),  # Gaussian bump
        np.ones(N) * (1.0 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)),  # Sinusoidal modulation
    ]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        v = _exact_value(init)
        if np.isfinite(v) and v < best_val:
            best_val = v
            best_h = init.copy()
    
    # Main optimization loop with annealing T
    # Start with moderate T, increase gradually
    T_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for T in T_schedule:
        if time.time() > deadline - 15:
            break
        
        h = best_h.copy()
        # Run gradient descent for this T
        lr = 0.001
        for step in range(500):
            if time.time() > deadline - 10:
                break
            
            val, grad = _objective_and_grad(h, T)
            
            if not np.isfinite(val) or not np.all(np.isfinite(grad)):
                break
            
            # Gradient step with clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                step_size = min(lr, 0.01 / grad_norm)
                h_new = h - step_size * grad
            
            # Also try a small random perturbation to escape local minima
            if step % 50 == 49:
                h_new = h_new + rng.normal(0, 0.005, N)
            
            # Check exact value
            v_exact = _exact_value(h_new)
            if np.isfinite(v_exact) and v_exact < best_val:
                best_val = v_exact
                best_h = h_new.copy()
            
            h = h_new
        
        # After each T stage, do a few exact gradient steps
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            val, grad = _exact_grad(best_h)
            if not np.all(np.isfinite(grad)):
                break
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            step_size = min(0.001, 0.005 / grad_norm)
            h_new = best_h - step_size * grad
            v_exact = _exact_value(h_new)
            if np.isfinite(v_exact) and v_exact < best_val:
                best_val = v_exact
                best_h = h_new
    
    # Final phase: coordinate descent refinement
    if time.time() < deadline - 5:
        for _ in range(20000):
            if time.time() > deadline - 3:
                break
            idx = rng.integers(0, N)
            h_new = best_h.copy()
            # Try several perturbations
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                h_try = h_new.copy()
                h_try[idx] += delta
                v_try = _exact_value(h_try)
                if np.isfinite(v_try) and v_try < best_val - 1e-12:
                    best_val = v_try
                    best_h = h_try
                    break
    
    # Final validation
    final_val = _exact_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
