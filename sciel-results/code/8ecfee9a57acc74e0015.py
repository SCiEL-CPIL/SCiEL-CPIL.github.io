"""
Spectral Phase Optimization via Stochastic Gradient Descent.

Strategy:
1. Fix the magnitude spectrum of the DFT to be constant (flat). This ensures the 
   'energy' is spread uniformly, which is a necessary condition for minimizing 
   autoconvolution peaks in the ideal limit.
2. Optimize only the phase vector $\phi$ to minimize the maximum absolute value 
   of the off-diagonal autoconvolution, normalized by the squared integral.
3. Use Stochastic Gradient Descent (SGD) with a smooth max surrogate (log-sum-exp) 
   to handle the non-differentiable max operation.
4. Compute gradients analytically using the chain rule: 
   d(c_k)/d(phi_i) involves the derivative of the IFFT of |X|^2 with respect to phase.
   Actually, since we fix |X|, we can just compute the gradient of the time-domain 
   signal x with respect to phi, and then the gradient of the autoconvolution 
   with respect to x.
   
   Let X = M * exp(i*phi). 
   x = IFFT(X).
   c = autoconv(x).
   We want to minimize L = max(|c_k| for k!=0) / (sum(x))^2.
   
   We use a soft-max surrogate: S = log(sum(exp(beta * |c_k|))) / beta.
   Gradient of S w.r.t. x can be computed.
   Gradient of x w.r.t. phi: dx/dphi = i * (d IFFT / d X) * X.
   Since x is real, we take the real part.
"""

import numpy as np
import time

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _phase_to_signal(phi, mag):
    """Convert phase vector and fixed magnitude to time-domain signal."""
    # phi and mag are for the full complex FFT (length N)
    X = mag * np.exp(1j * phi)
    x = np.fft.ifft(X).real
    return x

def _compute_objective_and_grad(phi, mag, beta=20.0):
    """
    Compute the soft-max objective and its gradient with respect to phi.
    
    Objective: J = log(sum_{k!=0} exp(beta * |c_k|)) / beta / (sum(x))^2
    We minimize this.
    
    Note: The normalization (sum(x))^2 is constant if we fix mag[0] and phi[0]=0,
    but sum(x) can vary slightly if we allow other phases to change? 
    No, sum(x) = X[0] = mag[0] * exp(i*phi[0]). 
    If we fix phi[0]=0 and mag is fixed, sum(x) is constant.
    So we can ignore the derivative of the normalization term.
    """
    n = len(phi)
    x = _phase_to_signal(phi, mag)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros(n)
    
    c = _autoconv_lin(x)
    abs_c = np.abs(c[1:]) # Exclude lag 0
    
    if len(abs_c) == 0:
        return 0.0, np.zeros(n)
        
    m = np.max(abs_c)
    # Soft max: log(sum(exp(beta * (abs_c - m)))) / beta + m
    # To avoid overflow, subtract max
    exp_terms = np.exp(beta * (abs_c - m))
    soft_max_val = m + np.log(np.sum(exp_terms)) / beta
    obj_val = soft_max_val / (s * s)
    
    # Gradient of soft_max w.r.t. abs_c
    # d/d(abs_c_j) [ m + log(sum(exp(beta*(abs_c - m)))) / beta ]
    # Note: m depends on abs_c, but in the "log-sum-exp" formulation often used,
    # if we treat m as the max element, the gradient is 1 for the max element and 0 otherwise 
    # if beta -> inf. For finite beta, the gradient is proportional to exp(beta*abs_c_j)/sum(...).
    # Let's use the standard gradient for log-sum-exp:
    # G_j = exp(beta * abs_c_j) / sum(exp(beta * abs_c))
    # But we used the shifted version. 
    # d/dy_j log(sum exp(beta(y_i - y_max))) = exp(beta(y_j - y_max)) / sum exp(beta(y_i - y_max))
    # This is a probability distribution p_j.
    p = exp_terms / np.sum(exp_terms)
    
    # Gradient of |c_k| w.r.t. c_k is sign(c_k)
    # We only have abs_c, so we need sign of c[1:]
    sign_c = np.sign(c[1:])
    
    # Gradient of soft_max w.r.t. c[1:] is p * sign_c
    grad_c = p * sign_c
    
    # Gradient of c w.r.t. x
    # c = autoconv(x)
    # c_k = sum_j x_j x_{j+k}
    # d c_k / d x_i = x_{i+k} + x_{i-k} (with boundary handling)
    
    # We need grad_x = sum_k grad_c[k] * (d c_{k+1} / d x)
    # Let's compute this efficiently.
    # grad_x_i = sum_{k=1}^{2n-2} grad_c[k-1] * (x_{i+k} + x_{i-k})
    # where x_j = 0 if j < 0 or j >= n.
    
    grad_x = np.zeros(n)
    
    # We can compute this via FFT or direct convolution.
    # Since n=400, direct is O(N^2) = 160k ops, which is fine for 400 iterations.
    # But let's try to be smarter or just use the direct loop if it's fast enough.
    # Actually, 400*400 = 160,000 operations per gradient step.
    # 400 steps * 160,000 = 64,000,000 ops. This is < 1 second in Python/NumPy if vectorized.
    # Let's vectorize the accumulation.
    
    # grad_x[i] = sum_{k} grad_c[k-1] * x[i+k] + sum_{k} grad_c[k-1] * x[i-k]
    # This is a cross-correlation of grad_c with x.
    # Let g = grad_c (length 2n-1, but we only use indices 1 to 2n-2, i.e., length 2n-2)
    # Actually, c has length 2n-1. We use c[1:] so indices 1 to 2n-2.
    # grad_c corresponds to these indices.
    
    # Term 1: sum_{k=1}^{2n-2} grad_c[k-1] x[i+k]
    # Let j = i+k => k = j-i.
    # Sum over j such that 1 <= j-i <= 2n-2.
    # This is a convolution of grad_c with x?
    # Let's just use the direct formula with np.roll or slicing for speed.
    
    # Simpler: Compute gradient of c w.r.t x using FFT?
    # d c / d x is a linear operator. 
    # c = IFFT(FFT(x)^2) is not quite right for linear conv.
    # Linear autoconv c_k = sum_j x_j x_{j+k}.
    # d c_k / d x_i = x_{i+k} + x_{i-k}.
    
    # Let's compute grad_x by summing contributions.
    # grad_x = sum_{k=1}^{2n-2} grad_c[k-1] * (shift(x, -k) + shift(x, k))
    # where shift(x, k) is x shifted by k (zero-padded).
    
    # We can compute this as:
    # grad_x = convolve(grad_c, x) + convolve(grad_c_rev, x) ?
    # Let's stick to a loop over k if vectorization is tricky, but 400 iterations is small.
    # However, doing it in a Python loop over k (400 times) inside the gradient calc (400 times) 
    # means 160,000 Python loop iterations, which might be slow (0.1s-1s).
    # Let's try to vectorize using FFT for the correlation.
    
    # grad_x[i] = sum_{k} g[k] x[i+k] + sum_{k} g[k] x[i-k]
    # Let A[i] = sum_{k} g[k] x[i+k]. This is the cross-correlation of g and x, but g is only non-zero for k>0.
    # Let's pad g to length 2n.
    # Actually, let's just use the property that the gradient of sum_k g_k c_k w.r.t x 
    # is equivalent to applying the adjoint of the autoconv operator to g.
    
    # The autoconv operator T: x -> c.
    # <g, T x> = sum_k g_k sum_j x_j x_{j+k}
    # = sum_j x_j (sum_k g_k x_{j+k})
    # So grad_x_j = sum_k g_k x_{j+k} + sum_k g_k x_{j-k} (due to symmetry of autoconv? No, autoconv is not symmetric in the same way as correlation with a fixed kernel, but c_k = c_{-k} for real x? No, c_k = sum x_j x_{j+k}. c_{-k} = sum x_j x_{j-k} = sum x_{i+k} x_i = c_k. Yes, it is symmetric.)
    # So grad_x = 2 * CrossCorr(g, x) ?
    # Let's check:
    # sum_k g_k c_k = sum_k g_k sum_j x_j x_{j+k}
    # Let i = j+k => j = i-k.
    # = sum_k g_k sum_i x_{i-k} x_i
    # = sum_i x_i (sum_k g_k x_{i-k})
    # So grad_x_i = sum_k g_k x_{i-k}.
    # This is the convolution of g and x!
    # Note: g is defined for k=1..2n-2. We should treat g[k] as 0 for k<=0 or k>=2n-1.
    
    # So grad_x = convolve(g, x) (linear convolution).
    # Wait, the index in g is the lag k.
    # g vector in our code is grad_c, which corresponds to lags 1 to 2n-2.
    # Let's create a full lag vector G of length 2n-1, with G[0]=0, G[1:]=grad_c, G[2n-1]=0.
    # Then grad_x = convolve(G, x, 'full')[:n] ?
    # Let's verify.
    # (G * x)_i = sum_k G[k] x[i-k].
    # We want sum_{k=1}^{2n-2} G[k] x[i-k].
    # Yes, this matches.
    
    G = np.zeros(2 * n - 1)
    G[1:] = grad_c
    
    # Compute linear convolution of G and x
    # G has length 2n-1, x has length n.
    # Result length 3n-2.
    # We want the first n elements?
    # (G*x)_i for i=0..n-1.
    # i=0: sum_k G[k] x[-k] = 0 (since x is 0 for negative indices)
    # i=n-1: sum_k G[k] x[n-1-k].
    
    conv_res = np.convolve(G, x, mode='full')
    # The term sum_k G[k] x[i-k] is indeed the convolution.
    # But we must be careful with the definition of convolution.
    # np.convolve(a, b)[k] = sum_i a[i] b[k-i].
    # So conv_res[i] = sum_k G[k] x[i-k].
    # This is exactly what we want for grad_x.
    
    grad_x = conv_res[:n]
    
    # Now, gradient of x w.r.t. phi.
    # x = Re( IFFT( M * exp(i phi) ) )
    # dx/dphi_j = Re( IFFT( M * i * exp(i phi) * e_j ) )
    # where e_j is the unit vector with 1 at index j.
    # IFFT( Y * e_j ) = 1/N * sum_m Y_m exp(2 pi i m j / N)
    # So dx/dphi_j = Re( 1/N * sum_m (M_m i exp(i phi_m)) exp(2 pi i m j / N) )
    # Let Y = M * i * exp(i phi) = i X.
    # dx/dphi_j = Re( 1/N * IFFT(Y)_j ? No.
    # IFFT(Y) gives x.
    # The derivative of x_r w.r.t. phi_j is:
    # x_r = 1/N sum_m X_m exp(2 pi i m r / N)
    # dx_r / dphi_j = 1/N X_j i exp(2 pi i j r / N)
    # So dx/dphi is a matrix where column j is i/N * X_j * exp(2 pi i j r / N) (real part).
    
    # We want grad_phi = sum_r grad_x_r * dx_r / dphi
    # grad_phi_j = sum_r grad_x_r * Re( 1/N * i * X_j * exp(2 pi i j r / N) )
    # = Re( 1/N * i * X_j * sum_r grad_x_r * exp(2 pi i j r / N) )
    # sum_r grad_x_r * exp(2 pi i j r / N) is the DFT of grad_x at frequency j.
    # Let GFT = FFT(grad_x).
    # Then grad_phi_j = Re( 1/N * i * X_j * GFT_j )
    
    GFT = np.fft.fft(grad_x)
    X = mag * np.exp(1j * phi)
    
    # grad_phi = Re( (1j * X * GFT) / N )
    grad_phi = np.real((1j * X * GFT) / N)
    
    # Fix DC phase gradient to 0
    grad_phi[0] = 0.0
    
    return obj_val, grad_phi

def _optimize_phase_sgd(phi_init, mag, max_iters=500, time_budget=100.0):
    """
    Stochastic Gradient Descent on phase.
    """
    start = time.time()
    phi = phi_init.copy()
    best_phi = phi.copy()
    best_score = _metric_offdiag(_phase_to_signal(phi, mag))
    
    phi[0] = 0.0
    
    lr = 0.05
    momentum = 0.9
    v = np.zeros_like(phi)
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        obj, grad = _compute_objective_and_grad(phi, mag, beta=20.0)
        
        # Momentum step
        v = momentum * v - lr * grad
        phi_new = phi + v
        phi_new[0] = 0.0
        
        x_new = _phase_to_signal(phi_new, mag)
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-10:
            lr *= 0.5
            v *= 0.5
            continue
        
        sc_new = _metric_offdiag(x_new)
        
        if sc_new < best_score:
            best_score = sc_new
            best_phi = phi_new.copy()
            phi = phi_new
            # Slowly decay learning rate
            if it % 50 == 0:
                lr *= 0.9
        else:
            # If it got worse, revert and reduce learning rate
            phi = phi.copy() # revert
            lr *= 0.8
            v *= 0.5
            
    return best_phi, best_score

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    Uses spectral phase optimization with SGD.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_x = None
    best_score = float('inf')
    
    # Fixed flat magnitude spectrum
    # To ensure sum(x) is non-zero and positive, we set mag[0] to a large value.
    # If mag is all 1s, sum(x) = N * mag[0] * exp(i phi[0]) / N ?
    # x = IFFT(M exp(i phi)).
    # sum(x) = X[0] = M[0] exp(i phi[0]).
    # If M[0] = N, then sum(x) = N (if phi[0]=0).
    # This gives a good baseline integral.
    mag = np.ones(N)
    mag[0] = N
    
    # Multiple random initializations
    seeds = [42, 123, 999, 314, 271]
    
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
            
        rng = np.random.default_rng(seed)
        phi = rng.uniform(-np.pi, np.pi, N)
        phi[0] = 0.0
        
        # Initial check
        x_init = _phase_to_signal(phi, mag)
        sc_init = _metric_offdiag(x_init)
        if sc_init < best_score:
            best_score = sc_init
            best_x = x_init.copy()
            
        # Run SGD optimization
        try:
            phi_opt, score_opt = _optimize_phase_sgd(
                phi, mag, 
                max_iters=300, 
                time_budget=20.0
            )
            x_opt = _phase_to_signal(phi_opt, mag)
            if abs(np.sum(x_opt)) > 1e-10:
                sc_opt = _metric_offdiag(x_opt)
                if sc_opt < best_score:
                    best_score = sc_opt
                    best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish: Try thresholding to ±1
    if best_x is not None:
        x_thresh = np.sign(best_x)
        if np.all(x_thresh == 1):
            x_thresh[::2] = -1
        elif np.all(x_thresh == -1):
            x_thresh[::2] = 1
        
        s_thresh = np.sum(x_thresh)
        if abs(s_thresh) > 1e-10:
            score_thresh = _metric_offdiag(x_thresh)
            if score_thresh < best_score:
                best_score = score_thresh
                best_x = x_thresh.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x

if __name__ == "__main__":
    x = construct()
    print("Score:", _metric_offdiag(x))
