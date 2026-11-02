"""
Strategy: Spectral Coefficient Optimization via Cosine Basis.
We parametrize the step function a as the squared magnitude of the IDCT of a non-negative vector c.
Actually, a simpler and more robust spectral approach for non-negative functions is to represent 
the function as a sum of non-negative weighted cosines (or just optimize the coefficients of a 
cosine series directly with a non-negativity penalty).

Let a_i = sum_k w_k * cos(pi * k * (i - N/2) / N)  ... this is tricky with non-negativity.

Better: Let a be generated from a set of "spectral weights" S_k (k=0..M) such that 
a(t) = | sum_k c_k e^{i 2 pi k t / L} |^2 ? No, that's complex.

Let's stick to the directive: "Parametrize the step function as the squared magnitude of the inverse discrete cosine transform (IDCT) of a non-negative vector".
Let c be a non-negative vector of length N.
a = (IDCT(c))^2.
Then a is automatically non-negative.
We optimize c to minimize the normalized max autoconvolution of a.

Why this might work:
The IDCT basis functions are orthogonal. Optimizing in this basis might allow the optimizer to 
more easily find global structures (like the two-block structure) because the basis functions 
are global and smooth, whereas the time-domain representation forces the optimizer to navigate 
a highly correlated landscape of adjacent bins.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
h = 0.5 / N

def _autoconvolve_max(a):
    """Compute the normalized maximum autoconvolution."""
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    max_conv = np.max(conv)
    norm = h**2 * (np.sum(a))**2
    return max_conv / norm

def _dct_matrix(N):
    """Construct the DCT-II matrix for IDCT usage."""
    # IDCT: a = dct_inv(c)
    # We can use scipy.fftpack.dct with type=2 and inverse=True, or build manually.
    # Let's use scipy.fftpack for speed and accuracy.
    pass

from scipy.fftpack import dct, idct

def _a_from_c(c):
    """
    Compute time-domain signal a from spectral vector c.
    a = (idct(c))**2
    c is non-negative.
    """
    # idct expects the input to be the DCT coefficients.
    # Standard IDCT formula: a_i = sum_k c_k * cos(pi * k * (i + 0.5) * 2/N) ... varies by definition.
    # scipy.fftpack.idct(x, type=2) computes the inverse DCT-II.
    # Let's ensure normalization is consistent.
    # a = idct(c, type=2)
    # Then square it.
    a_raw = idct(c, type=2)
    a = a_raw ** 2
    # Normalize so that sum(a) * h = 1? 
    # The score formula is max_conv / (h^2 * (sum a)^2).
    # If we scale a by alpha, max_conv scales by alpha^2, norm scales by alpha^4? No.
    # conv(a) scales by alpha^2.
    # sum(a) scales by alpha.
    # norm = h^2 * alpha^2 * (sum a_original)^2.
    # So score = (alpha^2 * max_conv_orig) / (h^2 * alpha^2 * (sum a_orig)^2) = max_conv_orig / (h^2 * (sum a_orig)^2).
    # The score is scale-invariant! 
    # So we can just return a_raw**2. The normalization in _autoconvolve_max handles it.
    return a

def _obj_and_grad(c):
    """
    Objective: normalized max autoconvolution of a = (idct(c))^2.
    c: non-negative vector.
    We use a smooth approximation for the max.
    """
    a = _a_from_c(c)
    
    # Penalty for negative values in a? 
    # a = (idct(c))^2 is always non-negative. So no penalty needed for non-negativity of a.
    # However, we must ensure c is non-negative. We enforce this via bounds in the optimizer.
    
    # Compute smooth max of autoconvolution
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    
    # Smooth max approximation
    beta = 20.0
    max_c = np.max(conv)
    if max_c <= 0:
        return 0.0, np.zeros_like(c)
    
    shifted = conv / max_c
    # Log-sum-exp
    log_sum_exp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
    smooth_max = max_c * (1.0 + log_sum_exp)
    
    # We need the gradient of smooth_max wrt c.
    # smooth_max depends on conv, which depends on a, which depends on c.
    # Chain rule: d(smooth_max)/dc = d(smooth_max)/da * da/dc
    
    # 1. Gradient of smooth_max wrt a
    exp_terms = np.exp(beta * (shifted - 1.0))
    weights = exp_terms / np.sum(exp_terms)
    
    grad_conv = weights * max_c  # d(smooth_max)/d(conv[k])
    
    # grad_a = d(conv)/da
    # conv[k] = sum_m a[m] * a[m + k - (n-1)]
    # This is the same gradient logic as before.
    grad_a = np.zeros(n)
    for k in range(2*n-1):
        w = grad_conv[k]
        if w < 1e-12:
            continue
        idx1 = k - (n - 1)
        # First term: a[j + idx1]
        if idx1 >= 0:
            if n - 1 - idx1 >= 0:
                grad_a[:n-idx1] += w * a[idx1:]
        else:
            if -idx1 < n:
                grad_a[-idx1:] += w * a[:n+idx1]
        # Second term: a[j - idx1]
        idx2 = -idx1
        if idx2 >= 0:
            if n - 1 - idx2 >= 0:
                grad_a[:n-idx2] += w * a[idx2:]
        else:
            if -idx2 < n:
                grad_a[-idx2:] += w * a[:n+idx2]
        
    # 2. Gradient of a wrt c
    # a = (idct(c))^2
    # Let u = idct(c)
    # a = u^2
    # da/dc = 2 * u * (du/dc)
    # du/dc is the IDCT matrix.
    # u = idct(c)
    # We can compute the gradient by applying the adjoint of the IDCT (which is the DCT) to (2*u)
    # Actually, d(u_i)/d(c_k) = M[i,k] where M is the IDCT matrix.
    # grad_c = (da/dc)^T grad_a
    # da/dc = 2 * u * I * M  (element-wise multiplication by u, then matrix multiplication)
    # So grad_c = M^T * (2 * u * grad_a)
    # M^T is the DCT matrix (up to scaling factors depending on definition).
    # scipy.fftpack.dct(x, type=2) is the forward DCT.
    # Let's verify the relationship.
    # idct(dct(x)) = x.
    # So if u = idct(c), then c = dct(u) * scaling?
    # In scipy, dct and idct are inverses up to a factor of N/2?
    # Let's just compute the Jacobian-vector product numerically if analytical is too tricky, 
    # but analytical is faster.
    
    # Let's use the property that IDCT is linear.
    # u = T c, where T is the IDCT matrix.
    # a_i = u_i^2.
    # d(a_i)/d(c_k) = 2 u_i * T[i,k].
    # grad_c_k = sum_i grad_a_i * 2 u_i * T[i,k]
    # grad_c = T^T (2 * u * grad_a)
    
    u = idct(c, type=2)
    vec = 2.0 * u * grad_a
    # T^T corresponds to dct
    # Need to check scaling.
    # scipy.fftpack.idct(x, type=2) computes:
    # y_n = sqrt(2/N) * sum_{k=0}^{N-1} x_k * cos(pi*(n+0.5)*k/N)
    # The inverse operation (DCT) is:
    # x_k = sqrt(2/N) * sum_{n=0}^{N-1} y_n * cos(pi*(n+0.5)*k/N)
    # So T is symmetric and orthogonal (with the sqrt(2/N) factor).
    # Thus T^T = T.
    # So grad_c = idct(2 * u * grad_a, type=2)
    
    grad_c = idct(vec, type=2)
    
    return smooth_max, grad_c

def construct():
    deadline = time.time() + 130.0
    
    # Initialize c
    # We want a to be non-negative. 
    # If c is all zeros, a is zero. 
    # Let's start with a few modes.
    # c[0] corresponds to the DC component (constant).
    # Let's try initializing c with a single non-zero component at k=0 (uniform a)
    # or k=1 (cosine bump).
    
    best_c = None
    best_score = float('inf')
    
    # Initial guesses for c
    inits = []
    
    # 1. DC component only (uniform a)
    c0 = np.zeros(N)
    c0[0] = 1.0
    inits.append(c0)
    
    # 2. First few modes
    for k in range(1, 10):
        ck = np.zeros(N)
        ck[k] = 1.0
        inits.append(ck)
        
    # 3. Combination of low modes
    c_low = np.zeros(N)
    c_low[:5] = 1.0
    inits.append(c_low)
    
    # 4. Random non-negative sparse vectors
    for _ in range(5):
        cr = np.random.rand(N)
        # Sparsify
        mask = np.random.rand(N) < 0.1
        cr[~mask] = 0
        if np.sum(cr) > 0:
            inits.append(cr)

    # Evaluate and optimize from each initial guess
    for c_init in inits:
        if time.time() > deadline - 30:
            break
            
        # Optimize
        # We need to ensure c stays non-negative.
        # L-BFGS-B supports bounds.
        bounds = [(0, None) for _ in range(N)]
        
        try:
            result = minimize(_obj_and_grad, c_init, jac=True, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50})
            
            c_opt = result.x
            a_opt = _a_from_c(c_opt)
            
            # Check if a is effectively non-zero
            if np.sum(a_opt) < 1e-10:
                continue
                
            score_opt = _autoconvolve_max(a_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_c = c_opt.copy()
                
        except Exception as e:
            pass

    # If optimization failed or didn't find anything better than a baseline, 
    # we should return a known good baseline.
    # The uniform distribution (c0) gives a specific score.
    # Let's calculate the score for the best found or fallback to uniform.
    
    if best_c is None:
        # Fallback to uniform
        a_uni = np.ones(N)
        return list(a_uni)
        
    a_final = _a_from_c(best_c)
    
    # Optional: Polish with time-domain optimization if time permits
    # This allows us to refine the specific bins that the spectral optimization might have missed.
    if time.time() < deadline - 20:
        # Use the result from spectral optimization as the starting point for time-domain optimization
        a_current = a_final.copy()
        # Normalize to sum 1 for the logit optimization? 
        # The score is scale invariant, so we can work with a directly.
        # But the previous code used logit for non-negativity. 
        # Let's just do a few steps of L-BFGS-B on the time domain with a penalty for negative values.
        
        def time_obj(a):
            # Penalty for negative values
            neg_penalty = np.sum(np.minimum(a, 0)**2)
            # Smooth max
            n = len(a)
            nfft = 1 << (2 * n - 1).bit_length()
            A = np.fft.rfft(a, n=nfft)
            conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
            max_c = np.max(conv)
            if max_c <= 0:
                return 1e10, np.zeros(n)
            shifted = conv / max_c
            beta = 20.0
            log_sum_exp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
            smooth_max = max_c * (1.0 + log_sum_exp)
            
            # Gradient
            exp_terms = np.exp(beta * (shifted - 1.0))
            weights = exp_terms / np.sum(exp_terms)
            grad_conv = weights * max_c
            grad_a = np.zeros(n)
            for k in range(2*n-1):
                w = grad_conv[k]
                if w < 1e-12: continue
                idx1 = k - (n - 1)
                if idx1 >= 0:
                    if n - 1 - idx1 >= 0: grad_a[:n-idx1] += w * a[idx1:]
                else:
                    if -idx1 < n: grad_a[-idx1:] += w * a[:n+idx1]
                idx2 = -idx1
                if idx2 >= 0:
                    if n - 1 - idx2 >= 0: grad_a[:n-idx2] += w * a[idx2:]
                else:
                    if -idx2 < n: grad_a[-idx2:] += w * a[:n+idx2]
            
            # Add penalty gradient
            grad_a += 2.0 * np.minimum(a, 0)
            
            return smooth_max + neg_penalty, grad_a

        try:
            result_time = minimize(time_obj, a_current, jac=True, method='L-BFGS-B',
                                   options={'maxiter': 50, 'gtol': 1e-6, 'maxls': 30})
            a_refined = result_time.x
            # Clip negative values
            a_refined = np.maximum(a_refined, 0)
            if np.sum(a_refined) > 1e-10:
                score_refined = _autoconvolve_max(a_refined)
                if score_refined < best_score:
                    return list(a_refined)
        except Exception:
            pass

    return list(a_final)
