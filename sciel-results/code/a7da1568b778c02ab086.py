"""Optimized step function to minimize normalized max autoconvolution on [-1/4, 1/4]."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
# Interval length
h = (0.25 - (-0.25)) / N  # 0.5 / 600 = 1/1200

def _autoconvolve_max(a):
    """
    Compute the normalized maximum autoconvolution.
    Input a: array of heights.
    Returns: normalized max value.
    """
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    max_conv = np.max(conv)
    norm = h**2 * (np.sum(a))**2
    return max_conv / norm

def _objective_and_grad(a):
    """Objective and gradient for optimization."""
    # Compute autoconvolution using FFT
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
    
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    # Score = max_conv / (h^2 * (sum a)^2)
    s = np.sum(a)
    score = max_val / (h**2 * s**2)
    
    # Gradient: d(score)/da
    # score = max_conv / (h^2 * s^2)
    # d(score)/da_j = (1/(h^2 * s^2)) * d(max_conv)/da_j - (2*max_conv/(h^2 * s^3)) * d(s)/da_j
    # d(s)/da_j = 1
    # d(max_conv)/da_j: max_conv is the max of conv, and at the argmax,
    # conv[idx] = sum_i a_i * a_{i+idx} (for linear convolution)
    # d(conv[idx])/da_j = a_{idx-j} + a_{j-idx} (if in range)
    # Actually, conv[k] = sum_i a_i * a_{i+k}
    # d(conv[k])/da_j = a_{k-j} (if 0 <= k-j < n) + a_{j-k} (if 0 <= j-k < n)
    # Wait, conv[k] = sum_i a_i a_{i+k}. 
    # d/d a_j: term where i = j gives a_{j+k}, term where i+k = j gives a_{j-k}.
    # So d(conv[k])/da_j = a_{j+k} (if j+k < n) + a_{j-k} (if j-k >= 0)
    
    k = idx
    grad_conv = np.zeros(n)
    # a_{j+k} for j such that j+k < n, i.e., j < n-k
    if k < n:
        grad_conv[:n-k] += a[k:]
    # a_{j-k} for j such that j-k >= 0, i.e., j >= k
    if k < n:
        grad_conv[k:] += a[:n-k]
    
    # Note: if k=0, both terms give a_j, so grad_conv[j] = 2*a_j, which is correct
    # since conv[0] = sum a_i^2, d/da_j = 2*a_j.
    
    ds = 1.0  # d(s)/da_j = 1 for all j
    
    grad_score = (1.0 / (h**2 * s**2)) * grad_conv - (2.0 * max_val / (h**2 * s**3)) * ds
    
    return score, grad_score

def construct():
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    # Start with a reasonable initial guess
    # Uniform distribution is a baseline. Let's try a smoother profile.
    # A Gaussian-like profile might reduce the peak autoconvolution.
    x = np.linspace(-0.25, 0.25, N)
    # Try a raised cosine or Gaussian
    # Gaussian with sigma chosen to spread out
    sigma = 0.15
    a = np.exp(-x**2 / (2 * sigma**2))
    a = np.maximum(a, 0)
    # Normalize to sum = 1
    a = a / np.sum(a)
    
    best_a = a.copy()
    best_score = _autoconvolve_max(a)
    
    # Use L-BFGS-B for optimization
    # We minimize score = max(f*f) / ||f||_1^2 subject to a >= 0
    # We don't fix sum(a) = 1 because the objective is scale-invariant,
    # but L-BFGS-B needs a well-conditioned problem. Let's fix sum(a) = 1
    # by using a reparameterization or just include the constraint.
    # Actually, since the objective is scale-invariant, the optimizer might
    # drift. Let's fix sum(a) = 1 by normalizing after each step,
    # or use the constraint.
    
    # Better: minimize max_conv subject to sum(a) = 1, a >= 0.
    # Since score = max_conv / (h^2 * s^2) and we fix s = 1, score = max_conv / h^2.
    # So minimizing score is equivalent to minimizing max_conv with s=1.
    
    def obj(a):
        s = np.sum(a)
        # If s is very small, return large value
        if s < 1e-10:
            return 1e10
        n = len(a)
        nfft = 1 << (2 * n - 1).bit_length()
        A = np.fft.rfft(a, n=nfft)
        conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
        return np.max(conv)
    
    def obj_grad(a):
        s = np.sum(a)
        if s < 1e-10:
            return 1e10, np.zeros_like(a)
        n = len(a)
        nfft = 1 << (2 * n - 1).bit_length()
        A = np.fft.rfft(a, n=nfft)
        conv = np.fft.irfft(A * A, n=nfft)[:2*n-1]
        max_val = np.max(conv)
        idx = np.argmax(conv)
        k = idx
        grad_conv = np.zeros(n)
        if k < n:
            grad_conv[:n-k] += a[k:]
            grad_conv[k:] += a[:n-k]
        return max_val, grad_conv
    
    # Use L-BFGS-B with equality constraint sum(a) = 1
    # scipy.optimize.minimize doesn't directly support equality constraints in L-BFGS-B
    # We can use SLSQP or trust-constr, but they might be slower.
    # Alternative: reparameterize a = exp(b) / sum(exp(b))
    # This ensures a >= 0 and sum(a) = 1.
    
    def logit_obj(b):
        # Numerical stability: shift b to prevent overflow in exp
        b_max = np.max(b)
        a = np.exp(b - b_max)
        s = np.sum(a)
        if s < 1e-300:
            return 1e10, np.zeros_like(b)
        a = a / s
        val, grad_a = obj_grad(a)
        # Chain rule: da/db_j = a_j * (1 - a_j) (since a_j = exp(b_j)/sum(exp(b)))
        # d(a_j)/d(b_k) = a_j * (delta_jk - a_k)
        # grad_b_k = sum_j grad_a_j * d(a_j)/d(b_k) = sum_j grad_a_j * a_j * (delta_jk - a_k)
        # = a_k * grad_a_k - a_k * sum_j (grad_a_j * a_j)
        dot = np.sum(grad_a * a)
        grad_b = a * (grad_a - dot)
        return val, grad_b
    
    # Initial b
    b = np.log(a + 1e-10)
    
    try:
        result = minimize(logit_obj, b, jac=True, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50})
        if result.success or result.fun < obj(a):
            b_opt = result.x
            a_opt = np.exp(b_opt)
            a_opt = a_opt / np.sum(a_opt)
            score_opt = _autoconvolve_max(a_opt)
            if score_opt < best_score:
                best_a = a_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 20:
        return list(best_a)
    
    # Try a few different initial profiles and optimize each
    profiles = []
    # Gaussian with different sigmas
    for sigma in [0.1, 0.15, 0.2, 0.25]:
        a_g = np.exp(-x**2 / (2 * sigma**2))
        a_g = a_g / np.sum(a_g)
        profiles.append(a_g)
    # Raised cosine
    a_cos = 0.5 * (1 + np.cos(2 * np.pi * x / 0.5))
    a_cos = np.maximum(a_cos, 0)
    a_cos = a_cos / np.sum(a_cos)
    profiles.append(a_cos)
    # Uniform
    a_uni = np.ones(N) / N
    profiles.append(a_uni)
    # Triangular
    a_tri = np.maximum(1 - np.abs(x) / 0.25, 0)
    a_tri = a_tri / np.sum(a_tri)
    profiles.append(a_tri)
    
    for a_init in profiles:
        if time.time() > deadline - 15:
            break
        
        b_init = np.log(a_init + 1e-10)
        try:
            result = minimize(logit_obj, b_init, jac=True, method='L-BFGS-B',
                              options={'maxiter': 100, 'gtol': 1e-5, 'maxls': 30})
            if result.fun < obj(a_init):
                a_opt = np.exp(result.x)
                a_opt = a_opt / np.sum(a_opt)
                score_opt = _autoconvolve_max(a_opt)
                if score_opt < best_score:
                    best_a = a_opt
                    best_score = score_opt
        except Exception:
            pass
    
    # Final polish: try local search on the best solution
    if time.time() < deadline - 5:
        # Perturb and re-optimize
        a_perturb = best_a + 0.01 * np.random.randn(N)
        a_perturb = np.maximum(a_perturb, 0)
        a_perturb = a_perturb / np.sum(a_perturb)
        b_perturb = np.log(a_perturb + 1e-10)
        try:
            result = minimize(logit_obj, b_perturb, jac=True, method='L-BFGS-B',
                              options={'maxiter': 50, 'gtol': 1e-6, 'maxls': 20})
            if result.fun < obj(a_perturb):
                a_opt = np.exp(result.x)
                a_opt = a_opt / np.sum(a_opt)
                score_opt = _autoconvolve_max(a_opt)
                if score_opt < best_score:
                    best_a = a_opt
                    best_score = score_opt
        except Exception:
            pass
    
    return list(best_a)
