"""
Structural change: Direct L-BFGS-B optimization of the time-domain signal x,
using a smooth log-sum-exp surrogate for the max absolute autoconvolution,
with analytic gradients computed via FFT.

This is materially different from the current program's approach of:
1. Optimizing phases in frequency domain with fixed magnitudes (Nelder-Mead)
2. Then doing gradient descent on a smooth surrogate
3. Then local search
4. Then coordinate descent polish

The new approach directly optimizes x in time domain using L-BFGS-B, which
is a quasi-Newton method that can converge much faster than Nelder-Mead for
this smooth problem. The gradient is computed analytically via FFT-based
autoconvolution.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _soft_max_abs(conv, beta=500.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=500.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _build_basis():
    """Build a basis of Legendre polynomials truncated to length N."""
    # Use scipy.special.eval_legendre if available, otherwise implement
    # For simplicity, we'll use a simple polynomial basis with orthogonalization
    # But to keep it simple and fast, let's use a truncated Fourier basis or
    # a simple polynomial basis.
    
    # Actually, let's use a basis of polynomials: x^0, x^1, ..., x^D
    # But we need to be careful with the domain. Let's map x to [-1, 1].
    # Or better, use a basis that is naturally orthogonal on [0, N-1].
    
    # For simplicity and speed, let's use a basis of sines and cosines
    # (Fourier basis) which is orthogonal and easy to handle.
    # But the task asks for Legendre. Let's implement Legendre.
    
    # Legendre polynomials P_n(x) are orthogonal on [-1, 1].
    # We can map the index i in [0, N-1] to t in [-1, 1] via t = 2*i/(N-1) - 1.
    
    D = 20  # Degree of polynomials
    t = 2.0 * np.arange(N) / (N - 1) - 1.0
    
    basis = np.zeros((D + 1, N))
    
    # P_0(x) = 1
    basis[0] = 1.0
    
    if D >= 1:
        # P_1(x) = x
        basis[1] = t
    
    # Recurrence: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
    for n in range(1, D):
        basis[n + 1] = ((2 * n + 1) * t * basis[n] - n * basis[n - 1]) / (n + 1)
    
    # Orthonormalize the basis (Gram-Schmidt) to make optimization easier
    # Since Legendre polynomials are orthogonal but not orthonormal, we normalize
    for i in range(D + 1):
        norm = np.sqrt(np.sum(basis[i] ** 2))
        if norm > 1e-10:
            basis[i] /= norm
    
    return basis


def _objective_coeffs(coeffs, basis, beta=200.0):
    """
    Objective function for coefficient optimization.
    coeffs: array of coefficients for the basis
    basis: 2D array of basis functions
    """
    x = coeffs @ basis
    
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return 1e10
    
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    return soft_max / (s * s)


def _gradient_coeffs(coeffs, basis, beta=200.0):
    """
    Gradient of the objective with respect to coefficients.
    """
    x = coeffs @ basis
    
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return np.zeros(len(coeffs))
    
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of soft_max with respect to conv[k]
    # d(soft_max)/d(conv[k]) = sign(conv[k]) * weights[k]
    # But we need d(soft_max)/d(x[j])
    
    # soft_max = log(sum(exp(beta * |conv[k]|))) / beta + max_val
    # d(soft_max)/d(conv[k]) = sign(conv[k]) * exp(beta * |conv[k]|) / sum(exp(beta * |conv[l]|))
    # = sign(conv[k]) * weights[k]
    
    # d(conv[k])/d(x[j]) = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
    
    grad_x = np.zeros(N)
    for k in range(2 * N - 1):
        if weights[k] < 1e-10:
            continue
        sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
        for j in range(N):
            contrib = 0.0
            if j + k < N:
                contrib += x[j + k]
            if j - k >= 0:
                contrib += x[j - k]
            grad_x[j] += 2.0 * sign_k * weights[k] * contrib
    
    soft_max_val = _soft_max_abs(conv, beta)
    grad_x *= -2.0 * soft_max_val / (s ** 3)
    
    # Chain rule: grad_coeffs[i] = sum_j grad_x[j] * basis[i][j]
    grad_coeffs = basis @ grad_x
    
    return grad_coeffs


def _optimize_legendre(x_init, deadline, rng):
    """
    Optimize coefficients in the Legendre polynomial basis.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    basis = _build_basis()
    D = basis.shape[0] - 1
    
    # Project initial x onto the basis
    # coeffs = basis @ x (since basis is orthonormal)
    coeffs_init = basis @ x_init
    
    # Normalize to ensure sum is 1
    s_init = np.sum(x_init)
    if abs(s_init) > 1e-12:
        coeffs_init /= s_init
    
    num_runs = 3
    time_per_run = (deadline - time.time()) / num_runs
    
    for run_idx in range(num_runs):
        if time.time() > deadline - 10:
            break
        
        if run_idx == 0:
            coeffs_current = coeffs_init.copy()
        else:
            # Add random perturbation
            coeffs_current = coeffs_init + rng.normal(0, 0.1, size=D + 1)
        
        start_time = time.time()
        try:
            result = minimize(
                _objective_coeffs,
                coeffs_current,
                args=(basis, 200.0),
                jac=_gradient_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            x_final = result.x @ basis
            val_final = _metric(x_final)
            
            if val_final < best_val:
                best_val = val_final
                best_x = x_final.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > time_per_run:
            pass

    return best_x, best_val


def _local_search_coeffs(x_init, deadline, rng, max_rounds=50000):
    """Local search in the coefficient space."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    basis = _build_basis()
    D = basis.shape[0] - 1
    
    coeffs_current = basis @ best_x
    s_init = np.sum(best_x)
    if abs(s_init) > 1e-12:
        coeffs_current /= s_init
    
    no_improve = 0
    step_scale = 0.05
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Pick a random coefficient to perturb
        idx = int(rng.integers(0, D + 1))
        delta = rng.normal(0, step_scale)
        
        coeffs_test = coeffs_current.copy()
        coeffs_test[idx] += delta
        
        x_test = coeffs_test @ basis
        val_test = _metric(x_test)
        
        if val_test < best_val:
            best_val = val_test
            best_x = x_test.copy()
            coeffs_current = coeffs_test
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent in time domain."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Legendre polynomial coefficient optimization
    if time.time() < deadline - 40:
        x_leg, val_leg = _optimize_legendre(best_x, deadline - 40, rng)
        if val_leg < best_val:
            best_val = val_leg
            best_x = x_leg
    
    # Phase 2: Local search in coefficient space
    if time.time() < deadline - 15:
        x_ls, val_ls = _local_search_coeffs(best_x, deadline - 15, rng, max_rounds=30000)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls
    
    # Phase 3: Final polish in time domain
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
