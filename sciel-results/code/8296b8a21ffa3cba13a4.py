"""Rewritten program: smooth surrogate with correct gradient, diverse init, local polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_chebyt

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute Chebyshev basis on [-1, 1]
_x = np.linspace(-1, 1, N)
_num_basis = 14
_basis = np.array([eval_chebyt(k, _x) for k in range(_num_basis)])  # (14, 400)
_basis_T = _basis.T  # shape (400, 14)


def _coeffs_to_h(c):
    """Convert coefficient vector to step heights."""
    h = _basis_T @ c
    s = np.sum(h)
    if abs(s) < 1e-10:
        h[0] += 1e-6
    return h


def _value_from_h(h):
    """Compute the task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))


def _smooth_objective(c, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for the max of autoconvolution.
    Returns (value, gradient) for use with L-BFGS-B.
    """
    h = _basis_T @ c
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(c)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = conv - m
    # log-sum-exp: m + (1/beta)*log(sum(exp(beta*(conv-m))))
    logsumexp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    val = 2.0 * N * logsumexp / (s * s)
    val = abs(val)
    
    # Gradient computation
    # d(logsumexp)/dc = (1/s^2) * d(logsumexp)/dh * dh/dc
    # d(logsumexp)/dh_i = sum_j (conv[j] weight) where conv[j] involves h_i
    # conv[k] = sum_i h[i]*h[k-i]
    # d(conv[k])/dh[i] = h[k-i] + h[k-i] if k-i != i... actually:
    # conv[k] = sum_j h[j]*h[k-j], so d(conv[k])/dh[i] = h[k-i] + h[i] (if k-i != i)
    # Actually d(conv[k])/dh[i] = h[k-i] + (h[i] if k-2i==0... no)
    # Let's use: d(conv[k])/dh[i] = h[k-i] + h[i] when considering both terms
    # More precisely: conv[k] = sum_j h[j] h[k-j]
    # d/dh[i] = sum_j (delta_{j,i} h[k-j] + h[j] delta_{k-j,i})
    # = h[k-i] + h[k-i] = 2*h[k-i]  (when k-i is in [0,N-1])
    # Wait, that's only if both j=i and k-j=i contribute, which requires k=2i.
    # Actually: d(conv[k])/dh[i] = h[k-i] + h[k-i] if we consider j=i gives h[k-i],
    # and k-j=i gives j=k-i, so h[k-i]*h[i] differentiated w.r.t h[i] gives h[k-i]... 
    # Let me redo: conv[k] = sum_j h[j]*h[k-j]
    # d(conv[k])/dh[i] = sum_j [delta_{j,i}*h[k-j] + h[j]*delta_{k-j,i}]
    # = h[k-i] + h[k-i]  (first term: j=i, second term: k-j=i => j=k-i)
    # = 2*h[k-i]  (valid when 0 <= k-i <= N-1)
    
    # So d(logsumexp)/dh[i] = sum_k w_k * 2*h[k-i] where w_k = exp(beta*(conv[k]-m))/sum(...)
    # This is 2 * (w * h)[i] where w is the weight vector and we do circular-like convolution
    # Actually sum_k w[k] * h[k-i] for k in range where h[k-i] is defined
    # This is (w flipped) convolved with h, evaluated at position i... 
    # Let's just compute it directly for correctness:
    
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)  # normalized
    
    # d(logsumexp)/dh[i] = sum_k weights[k] * 2*h[k-i]
    # For each i, sum over k where 0<=k-i<N: weights[k]*h[k-i]
    # This is a convolution-like operation
    # Let's compute: result[i] = sum_k weights[k] * h[k-i]
    # = (weights * h)[i] in a cross-correlation sense
    
    # Using FFT for efficiency
    # Actually for N=400 and conv length 799, let's just do it with np.convolve
    # result[i] = sum_{k=max(0,i)}^{min(798, i+399)} weights[k] * h[k-i]
    
    # Let's use: for each i, the valid k range is [i, i+N-1] intersected with [0, 2N-2]
    # result[i] = sum_k weights[k]*h[k-i] where 0<=k-i<N and 0<=k<2N-1
    
    # Efficient approach: pad and convolve
    # result = np.convolve(weights, h, mode='full')[N-1:]  -- not quite right
    
    # Let me think differently. We want for each i in [0,N-1]:
    # g[i] = sum_{k=0}^{2N-2} weights[k] * h[k-i] * I(0<=k-i<N)
    # = sum_{j=0}^{N-1} weights[i+j] * h[j]  where i+j < 2N-1
    # = sum_{j=0}^{N-1} weights[i+j] * h[j]
    
    # This is a sliding dot product. We can compute via:
    # Pad weights to length 2N-1 (already is), then for each i: dot(weights[i:i+N], h)
    # This is like a correlation. Use np.correlate or manual.
    
    # For N=400, this is manageable with a loop or FFT
    # Let's use FFT: g = ifft(fft(weights, 2N) * fft(h_padded, 2N))
    
    h_padded = np.zeros(2 * N - 1)
    h_padded[:N] = h
    w_padded = np.zeros(2 * N - 1)
    w_padded[:] = weights
    
    # g[i] = sum_j w[i+j]*h[j] = (w * h)[i] where * is cross-correlation
    # np.correlate(w_padded, h_padded, mode='valid') won't give what we want directly
    # Let's use: g = np.correlate(w_padded, h_padded, mode='full')[N-1:2*N-1]
    # Actually: (w * h)[n] = sum_k w[n-k]*h[k], so g[i] = sum_j w[i+j]*h[j] = sum_k w[i+k]*h[k]
    # This is the cross-correlation at lag -i... 
    
    # Simpler: just loop since N=400
    g = np.zeros(N)
    for i in range(N):
        k_start = i
        k_end = min(i + N, 2 * N - 1)
        g[i] = np.dot(weights[k_start:k_end], h[:k_end - k_start])
    
    # d(logsumexp)/dh = 2 * g / s^2 ... wait, logsumexp doesn't depend on s
    # d(val)/dc where val = 2*N*logsumexp/s^2
    # d(val)/dc = 2*N * (d(logsumexp)/dc * s^2 - logsumexp * 2*s*ds/dc) / s^4
    # = 2*N * (d(logsumexp)/dc * s - 2*logsumexp*ds/dc) / s^3
    
    ds_dc = np.sum(_basis_T, axis=0)  # d(sum(h))/dc = sum over rows of basis_T
    
    dlogsumexp_dc = _basis_T.T @ (2.0 * g)  # d(logsumexp)/dc = sum_i d(logsumexp)/dh[i] * dh[i]/dc
    
    # val = abs(2*N*logsumexp/s^2)
    # For gradient, assume val > 0 (which it is for our purposes)
    # d(val)/dc = 2*N*(dlogsumexp_dc * s - 2*logsumexp * ds_dc) / s^3
    
    grad = 2.0 * N * (dlogsumexp_dc * s - 2.0 * logsumexp * ds_dc) / (s ** 3)
    
    return val, grad


def _hard_objective(c):
    """Hard objective for final evaluation."""
    h = _coeffs_to_h(c)
    return _value_from_h(h)


def construct():
    t0 = time.time()
    
    # Start with seed: all ones
    c_seed = np.linalg.pinv(_basis_T) @ np.ones(N)
    
    best_c = c_seed.copy()
    best_val = _hard_objective(best_c)
    
    # Phase 1: Differential Evolution over coefficients
    if time.time() - t0 < DEADLINE - 40:
        bounds = [(-5.0, 5.0) for _ in range(_num_basis)]
        
        def de_obj(c):
            if time.time() - t0 > DEADLINE - 25:
                return best_val
            h = _coeffs_to_h(c)
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        try:
            de_result = differential_evolution(
                de_obj, bounds,
                maxiter=300, popsize=20,
                tol=1e-8, seed=42,
                polish=False
            )
            if de_result.fun < best_val:
                best_c = de_result.x.copy()
                best_val = de_result.fun
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B using smooth surrogate
    # Try multiple beta values
    for beta in [30.0, 80.0, 150.0, 300.0]:
        if time.time() - t0 > DEADLINE - 10:
            break
        try:
            def obj_and_grad(c):
                val, grad = _smooth_objective(c, beta)
                return val, grad
            
            res = minimize(
                obj_and_grad,
                best_c,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            hard_val = _hard_objective(res.x)
            if hard_val < best_val:
                best_c = res.x.copy()
                best_val = hard_val
        except Exception:
            pass
    
    # Phase 3: Additional local refinement with very tight tolerance
    if time.time() - t0 < DEADLINE - 5:
        try:
            def obj_and_grad_tight(c):
                val, grad = _smooth_objective(c, 500.0)
                return val, grad
            
            res = minimize(
                obj_and_grad_tight,
                best_c,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            hard_val = _hard_objective(res.x)
            if hard_val < best_val:
                best_c = res.x.copy()
                best_val = hard_val
        except Exception:
            pass
    
    # Final: return the best h found
    h_best = _coeffs_to_h(best_c)
    
    # Safety check
    if not np.all(np.isfinite(h_best)) or abs(np.sum(h_best)) < 1e-15:
        return np.ones(N)
    
    return h_best


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
