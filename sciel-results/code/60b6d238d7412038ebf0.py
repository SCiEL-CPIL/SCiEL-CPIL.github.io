"""
Smoothed L-BFGS optimization for signed step functions minimizing max|autoconv|/(sum x)^2.
Uses a LogSumExp surrogate for the maximum absolute value to enable efficient gradient-based optimization.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _smooth_objective_and_grad(x, mu):
    """
    Compute the smoothed objective F(x) = mu * log(sum(exp(|C[k]|/mu))) / (sum x)^2
    and its gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        # Avoid division by zero
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Numerical stability for LogSumExp
    max_abs = np.max(abs_ac)
    if max_abs < 1e-12:
        # If autoconv is near zero, gradient is near zero
        return 1e-12, np.zeros(N)
    
    # Scale to prevent overflow in exp
    # We want log(sum(exp(abs_ac/mu)))
    # = max_abs/mu + log(sum(exp((abs_ac - max_abs)/mu)))
    shifted = (abs_ac - max_abs) / mu
    # Clip to avoid underflow/overflow issues, though exp(-large) is fine
    exp_terms = np.exp(shifted)
    
    sum_exp = np.sum(exp_terms)
    log_sum_exp = max_abs / mu + np.log(sum_exp)
    
    M_val = mu * log_sum_exp
    F_val = M_val / (s * s)
    
    # Gradient calculation
    # d(M)/dx_j = (1/sum_exp) * sum_k (sign(ac[k]) * exp((abs(ac[k])-max_abs)/mu) * d(ac[k])/dx_j)
    # Note: d/dz |z| = sign(z). The chain rule gives sign(ac[k]) * exp(...) * d(ac[k])/dx_j.
    # Wait, the term in the sum for M is exp(|ac[k]|/mu).
    # d/dx exp(|ac|/mu) = exp(|ac|/mu) * (1/mu) * sign(ac) * d(ac)/dx.
    # So d(M)/dx = mu * (1/sum_exp_original) * sum_k exp(|ac|/mu) * (1/mu) * sign(ac) * d(ac)/dx
    #             = (1/sum_exp_original) * sum_k exp(|ac|/mu) * sign(ac) * d(ac)/dx
    
    # We computed exp_terms = exp((abs_ac - max_abs)/mu) = exp(|ac|/mu) * exp(-max_abs/mu)
    # So exp(|ac|/mu) = exp_terms * exp(max_abs/mu)
    # sum_exp_original = sum(exp(|ac|/mu)) = exp(max_abs/mu) * sum_exp
    # The factor exp(max_abs/mu) cancels in the ratio (1/sum_exp_original) * exp(|ac|/mu)
    # So weight_k = sign(ac[k]) * exp_terms[k] / sum_exp
    
    weights = np.sign(ac) * exp_terms / sum_exp
    
    # Now compute conv = weights * x (linear convolution)
    # d(ac[k])/dx[j] = x[j+k] + x[j-k]
    # sum_k weights[k] * (x[j+k] + x[j-k])
    # This is equivalent to (weights * x)[j] + (weights_rev * x)[j]?
    # Let's use FFT for efficiency.
    # conv[j] = sum_k weights[k] x[j+k] + sum_k weights[k] x[j-k]
    # Term 1: (x * weights)[j] ? No, standard convolution (x*w)[j] = sum_k x[k] w[j-k].
    # We need sum_k w[k] x[j+k]. Let m = j+k => k = m-j. sum_m w[m-j] x[m]. This is correlation?
    # Actually, sum_k w[k] x[j+k] is the cross-correlation of x and w at lag j?
    # Let's just compute the convolution of x and w.
    # Let C1 = x * w (linear). C1[m] = sum_k x[k] w[m-k].
    # We want G[j] = sum_k w[k] x[j+k].
    # Let n = j+k. G[j] = sum_n w[n-j] x[n].
    # This is (w_rev * x)[j] where w_rev[k] = w[-k]?
    # Actually, sum_k w[k] x[j+k] is the convolution of w and x evaluated at j?
    # (w * x)[j] = sum_k w[k] x[j-k].
    # If we replace x with x_rev, (w * x_rev)[j] = sum_k w[k] x_rev[j-k] = sum_k w[k] x[N-1-(j-k)].
    
    # Simpler: Use FFT to compute the sum directly.
    # We want grad_M[j] = sum_k weights[k] (x[j+k] + x[j-k])
    # Let's compute A = weights * x (linear conv). A[m] = sum_k weights[k] x[m-k].
    # Then sum_k weights[k] x[j-k] = A[j].
    # And sum_k weights[k] x[j+k] = sum_k weights[-k] x[j-(-k)]? No.
    # Let's look at indices.
    # Term 2: sum_k weights[k] x[j-k]. This is exactly (weights * x)[j].
    # Term 1: sum_k weights[k] x[j+k]. Let l = -k. sum_l weights[-l] x[j-l].
    # This is (weights_rev * x)[j] where weights_rev[l] = weights[-l].
    # So grad_M = (weights * x) + (weights_rev * x).
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    W_fft = np.fft.rfft(weights, n=L_fft)
    X_fft = np.fft.rfft(x, n=L_fft)
    
    # Convolution 1: weights * x
    conv1 = np.fft.irfft(W_fft * X_fft, n=L_fft)[:L]
    
    # Convolution 2: weights_rev * x
    # weights_rev[k] = weights[-k]. In FFT domain, this is conjugate of W_fft?
    # If w is real, F(w_rev) = conj(F(w)) ?
    # w_rev[k] = w[-k]. DFT: sum_k w[-k] e^{-i 2 pi k n / L}.
    # Let m = -k. sum_m w[m] e^{i 2 pi m n / L} = conj( sum_m w[m] e^{-i 2 pi m n / L} ) = conj(W[n]).
    # So yes, F(w_rev) = conj(W_fft).
    conv2 = np.fft.irfft(np.conj(W_fft) * X_fft, n=L_fft)[:L]
    
    grad_M = conv1 + conv2
    
    # Ensure we only take the first N elements for x
    grad_M = grad_M[:N]
    
    # Total gradient of F = M / s^2
    # dF/dx = grad_M / s^2 - 2 * M * 1 / s^3
    grad_F = grad_M / (s * s) - 2.0 * M_val / (s * s * s) * np.ones(N)
    
    return F_val, grad_F

def _optimize_smooth(x_init, mu_start=1.0, mu_end=0.01, max_iter_per_mu=50):
    """
    Optimize x using L-BFGS with decreasing temperature mu.
    """
    x_best = x_init.copy()
    best_val = _smooth_objective_and_grad(x_best, mu_start)[0]
    
    # Geometric progression of mu
    mus = np.geomspace(mu_start, mu_end, num=10)
    
    for mu in mus:
        if _check_deadline():
            break
        
        def obj_func(x):
            if _check_deadline():
                return 1e10, np.zeros(N)
            val, grad = _smooth_objective_and_grad(x, mu)
            return val, grad
        
        try:
            result = minimize(
                obj_func,
                x_best,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_mu, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_val:
                # Check if the new solution is better in terms of the original objective
                # The smooth objective approximates the max, so we should verify with the true max
                true_val = np.max(np.abs(_autoconv_fft(result.x))) / (np.sum(result.x)**2)
                if true_val < best_val:
                    best_val = true_val
                    x_best = result.x.copy()
                elif result.fun < _smooth_objective_and_grad(x_best, mu)[0]:
                    # Even if true val didn't improve immediately, keep the smooth-improved state
                    # as it might lead to better results with lower mu
                    x_best = result.x.copy()
        except Exception:
            pass
            
    return x_best, best_val

def _coordinate_descent_polish(x, max_iters=20):
    """
    Final polish using coordinate descent on the true objective.
    """
    best_x = x.copy()
    ac = _autoconv_fft(best_x)
    s = np.sum(best_x)
    best_val = np.max(np.abs(ac)) / (s * s)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            if _check_deadline():
                break
            # Try a few step sizes
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                trial = best_x.copy()
                trial[j] += delta
                s_t = np.sum(trial)
                if abs(s_t) < 1e-10:
                    continue
                ac_t = _autoconv_fft(trial)
                val_t = np.max(np.abs(ac_t)) / (s_t * s_t)
                if val_t < best_val:
                    best_val = val_t
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    # Initial guess: Random or Spectral
    # Spectral init: Flat PSD, random phase
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_pos = L_fft // 2 + 1
    
    rng = np.random.RandomState(42)
    
    best_x = None
    best_val = 1e10
    
    # Try a few different initializations
    inits = []
    
    # 1. Random
    inits.append(rng.randn(N))
    
    # 2. Spectral flat
    mag = np.zeros(n_pos)
    mag[0] = 1.0
    mag[1:] = 1.0 / np.sqrt(n_pos - 1)
    phases = rng.uniform(0, 2*np.pi, n_pos)
    phases[0] = 0.0
    X = mag * np.exp(1j * phases)
    x_spec = np.fft.irfft(X, n=L_fft)[:N]
    s = np.sum(x_spec)
    if abs(s) > 1e-10:
        x_spec = x_spec / s
    inits.append(x_spec)
    
    # 3. Sinusoid
    inits.append(np.sin(2 * np.pi * np.arange(N) / 50.0))
    
    for x0 in inits:
        if _check_deadline():
            break
        
        # Normalize initial guess
        s0 = np.sum(x0)
        if abs(s0) < 1e-10:
            continue
        x0_norm = x0 / s0
        
        # Run smooth optimization
        x_opt, val_opt = _optimize_smooth(x0_norm, mu_start=1.0, mu_end=0.05, max_iter_per_mu=30)
        
        # Polish
        x_pol, val_pol = _coordinate_descent_polish(x_opt, max_iters=10)
        
        if val_pol < best_val:
            best_val = val_pol
            best_x = x_pol.copy()
    
    if best_x is None:
        # Fallback
        best_x = np.ones(N) / N
    
    return best_x
# EVOLVE-BLOCK-END
