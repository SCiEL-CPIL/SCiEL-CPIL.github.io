"""
Legendre Polynomial Coefficient Optimization for Signed Autocorrelation Minimization.

Strategy:
1. Represent the signal x of length N as a linear combination of the first M Legendre polynomials 
   evaluated at a grid corresponding to the signal indices.
   x[n] = sum_{k=0}^{M-1} c_k * P_k(t_n)
   where t_n = 2*n/(N-1) - 1 for n=0..N-1.
   
   The Legendre polynomials are orthogonal with respect to the discrete inner product on the grid 
   (approximating the continuous orthogonality). This parameterization constrains the signal to a 
   low-complexity subspace, inherently suppressing high-frequency noise that might cause sharp 
   peaks in the autoconvolution.

2. The metric is scale-invariant: max(|autoconv(x)|) / (sum(x))^2.
   We can fix the scale by normalizing the coefficients such that the sum of the signal is 1, 
   or simply optimize the coefficients and normalize at the end. 
   However, since the metric is homogeneous of degree 0, we can just minimize the numerator 
   assuming the denominator is fixed, or minimize the ratio directly.
   Let's minimize the ratio directly using a smooth approximation (soft-max) for differentiability.

3. Use gradient descent (L-BFGS-B or Adam-like) on the coefficients c to minimize the objective.
   The dimensionality M is much smaller than N (e.g., M=20 vs N=400), making the optimization 
   landscape smoother and easier to navigate than optimizing 400 raw time-domain samples.

4. After optimization, we might refine with a few steps of local search in the coefficient space 
   or even project back to time domain for final polishing if time permits, but the core is the 
   polynomial fit.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_legendre

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Precompute the Legendre basis matrix
# t grid from -1 to 1
_t_grid = 2.0 * np.arange(N) / (N - 1) - 1.0

def _build_basis(M):
    """Build the M x N matrix of Legendre polynomial values."""
    B = np.zeros((M, N))
    for k in range(M):
        B[k, :] = eval_legendre(k, _t_grid)
    return B

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

def _soft_max_abs(conv, beta=100.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _objective_coeffs(c, B, beta=100.0):
    """
    Objective function for coefficient optimization.
    c: array of coefficients
    B: basis matrix (M x N)
    """
    x = c @ B
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    # Compute autoconvolution
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    soft_max = _soft_max_abs(conv, beta)
    return soft_max / (s * s)

def _gradient_coeffs(c, B, beta=100.0, eps=1e-5):
    """
    Numerical gradient for coefficient optimization.
    """
    grad = np.zeros_like(c)
    f0 = _objective_coeffs(c, B, beta)
    
    for i in range(len(c)):
        c_plus = c.copy()
        c_plus[i] += eps
        f_plus = _objective_coeffs(c_plus, B, beta)
        
        c_minus = c.copy()
        c_minus[i] -= eps
        f_minus = _objective_coeffs(c_minus, B, beta)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
        
    return grad

def _optimize_legendre(M, deadline, rng):
    """
    Optimize the Legendre coefficients.
    """
    B = _build_basis(M)
    
    # Initial coefficients: random small values
    c_init = rng.normal(0, 1.0, size=M)
    
    best_c = c_init.copy()
    best_val = _objective_coeffs(best_c, B, 50.0)
    
    # Try increasing beta for sharper max approximation
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 5:
            break
        
        # Use L-BFGS-B for optimization
        # We can use numerical gradient to avoid implementing analytic gradient for soft-max
        res = minimize(
            _objective_coeffs,
            best_c,
            args=(B, beta),
            method='L-BFGS-B',
            jac=_gradient_coeffs,
            options={'maxiter': 100, 'disp': False, 'ftol': 1e-9, 'gtol': 1e-5}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    
    # Convert coefficients to time-domain signal
    x = best_c @ B
    
    # Normalize to sum=1 (or handle sign)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return None, float('inf')
    
    x = x / s
    
    # Evaluate exact metric
    val = _metric(x)
    
    return x, val

def _local_search_coeffs(x, c, B, deadline, rng, max_rounds=2000):
    """
    Refine coefficients with local search.
    """
    best_c = c.copy()
    best_val = _metric(x)
    no_improve = 0
    step_scale = 0.1
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 50:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Perturb a random subset of coefficients
        k = int(rng.integers(1, min(3, M)))
        indices = rng.choice(len(best_c), size=k, replace=False)
        
        best_delta = 0
        best_c_new = None
        for trial in range(2):
            c_test = best_c.copy()
            for idx in indices:
                c_test[idx] += rng.normal(0, step_scale)
            
            x_test = c_test @ B
            s = np.sum(x_test)
            if abs(s) < 1e-12:
                continue
            x_test = x_test / s
            v = _metric(x_test)
            
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_c_new = c_test
        
        if best_c_new is not None and best_delta < 0:
            best_val += best_delta
            best_c = best_c_new
            no_improve = 0
        else:
            no_improve += 1
            
    x_final = best_c @ B
    s = np.sum(x_final)
    if abs(s) < 1e-12:
        return x, best_val
    x_final = x_final / s
    return x_final, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Try different degrees of Legendre polynomials
    # Low degrees capture smooth trends, high degrees add detail
    # M=10, 20, 50
    for M in [10, 20, 50]:
        if time.time() > deadline - 20:
            break
        
        try:
            x_opt, val_opt = _optimize_legendre(M, deadline - 10, rng)
            if x_opt is not None and val_opt < best_val:
                best_x = x_opt
                best_val = val_opt
                
                # Refine with local search in coefficient space
                # We need to recover the coefficients from x_opt? 
                # Actually, _optimize_legendre returns x, not c. 
                # Let's modify it to return c as well, or just re-estimate c via least squares.
                B = _build_basis(M)
                # Least squares fit of x_opt to the basis
                c_fit, _, _, _ = np.linalg.lstsq(B, best_x, rcond=None)
                
                x_ref, val_ref = _local_search_coeffs(best_x, c_fit, B, deadline - 5, rng)
                if val_ref < best_val:
                    best_x = x_ref
                    best_val = val_ref
                    
        except Exception:
            pass

    return best_x
