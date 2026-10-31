import numpy as np
import time
from scipy.optimize import minimize

def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 (physicists' Hermite)."""
    # H0 = 1
    H0 = np.array([1.0])
    # H4 = 16x^4 - 48x^2 + 12
    H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])  # c0 + c1*x + ... + c4*x^4
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    H8 = np.array([168.0, 0.0, -1344.0, 0.0, 2688.0, 0.0, -1344.0, 0.0, 256.0])
    # H12 = 4096x^12 - 46080x^10 + 230400x^8 - 518400x^6 + 518400x^4 - 21600x^2 + 2310
    H12 = np.array([2310.0, 0.0, -21600.0, 0.0, 518400.0, 0.0, -518400.0, 0.0, 230400.0, 0.0, -46080.0, 0.0, 4096.0])
    return H0, H4, H8, H12

def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients at x."""
    result = np.zeros_like(x, dtype=float)
    for c in coeffs[::-1]:
        result = result * x + c
    return result

def _compute_c4(a0, a4, a8):
    """Compute the upper bound C4 given three free coefficients."""
    H0, H4, H8, H12 = _hermite_coeffs()
    # a12 from g(0) = 0
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 168.0) / 2310.0
    
    # Build g(x) polynomial coefficients
    # g = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Max degree is 12
    g_coeffs = np.zeros(13)
    g_coeffs += a0 * np.pad(H0, (0, 12))
    g_coeffs += a4 * np.pad(H4, (0, 8))
    g_coeffs += a8 * np.pad(H8, (0, 4))
    g_coeffs += a12 * H12
    
    # f(x) = g(x) * exp(-x^2/2)
    # A(f) = integral of |f(x)| dx
    # A(F) = integral of |F_hat(xi)| dxi
    # Use numerical integration over a sufficient range
    
    x = np.linspace(-8, 8, 40001)
    dx = x[1] - x[0]
    
    g_vals = _poly_eval(g_coeffs, x)
    f_vals = g_vals * np.exp(-x**2 / 2.0)
    
    A_f = np.sum(np.abs(f_vals)) * dx
    
    # Fourier transform: F_hat(xi) = integral f(x) exp(-2*pi*i*x*xi) dx
    # For f(x) = g(x) exp(-x^2/2), use the fact that FT of x^n exp(-x^2/2)
    # can be computed via derivatives of exp(-2*pi^2*xi^2)
    # Or just compute numerically
    
    # Actually, let's use the property: FT[x^n * exp(-x^2/2)] 
    # With convention F_hat(xi) = int f(x) exp(-2*pi*i*x*xi) dx
    # FT[exp(-x^2/2)] = exp(-2*pi^2*xi^2)
    # FT[x*exp(-x^2/2)] = i/(2*pi) * d/dxi [exp(-2*pi^2*xi^2)] = i*xi*exp(-2*pi^2*xi^2)
    # In general, FT[x^n * exp(-x^2/2)] = (i/(2*pi))^n * d^n/dxi^n [exp(-2*pi^2*xi^2)]
    
    # This gets complicated. Let me just compute the FT numerically.
    # Use FFT for efficiency
    # f(x) on x grid, compute FT
    
    # Actually for the bound, we might not need the exact FT.
    # Let me think about what C4 actually is.
    
    # Perhaps C4 is computed as: C4 = A(f) * A(F) where A is L1 norm
    # And we need to minimize this product.
    
    # Let me compute A(F) numerically via FFT
    # Need to be careful with the FFT normalization
    
    N = len(x)
    # Use np.fft for the FT
    # F_hat[k] = sum f[n] exp(-2*pi*i*n*k/N) * dx
    # This approximates int f(x) exp(-2*pi*i*x*xi_k) dx where xi_k = k/(N*dx)
    
    F_hat = np.fft.fft(f_vals) * dx
    # Frequency: xi_k = k / (N * dx) for k=0,...,N/2; negative for k>N/2
    xi = np.fft.fftfreq(N, d=dx)
    
    A_F = np.sum(np.abs(F_hat)) * (1.0 / (N * dx))  # dxi = 1/(N*dx)
    
    # Wait, the spacing in frequency: dxi = 1/(N*dx)
    A_F = np.sum(np.abs(F_hat)) * (1.0 / (N * dx))
    
    return A_f * A_F

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # Seed value
    seed_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    seed_value = _compute_c4(*seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    # Log-magnitude parameterization
    # Each coefficient = sign * exp(log_mag)
    # This allows exploration of different scales
    
    def _objective(params):
        # params: [log|a0|, sign0, log|a4|, sign4, log|a8|, sign8]
        # Or simpler: params are the actual coefficients, but we use a transformed space
        a0, a4, a8 = params
        val = _compute_c4(a0, a4, a8)
        if not np.isfinite(val):
            return 1e10
        return val
    
    def _log_mag_objective(params):
        # params: [ln|a0|, ln|a4|, ln|a8|, s0, s4, s8] where s in [-1,1]
        ln_a0, ln_a4, ln_a8, s0, s4, s8 = params
        a0 = np.sign(s0) * np.exp(ln_a0) if abs(s0) > 1e-10 else 0.0
        a4 = np.sign(s4) * np.exp(ln_a4) if abs(s4) > 1e-10 else 0.0
        a8 = np.sign(s8) * np.exp(ln_a8) if abs(s8) > 1e-10 else 0.0
        val = _compute_c4(a0, a4, a8)
        if not np.isfinite(val):
            return 1e10
        return val
    
    # Strategy: multiple random restarts with local optimization
    rng = np.random.default_rng(42)
    
    # Initial point from seed
    a0_s, a4_s, a8_s = seed_coeffs
    init_log = np.array([
        np.log(abs(a0_s)) if abs(a0_s) > 1e-15 else -5.0,
        np.log(abs(a4_s)) if abs(a4_s) > 1e-15 else -5.0,
        np.log(abs(a8_s)) if abs(a8_s) > 1e-15 else -5.0,
        1.0, 1.0, 1.0  # signs
    ])
    
    n_restarts = 0
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x0 = init_log
        else:
            # Random restart in log-magnitude space
            x0 = np.array([
                rng.uniform(-8, 3),
                rng.uniform(-8, 3),
                rng.uniform(-8, 3),
                rng.choice([-1.0, 1.0]),
                rng.choice([-1.0, 1.0]),
                rng.choice([-1.0, 1.0]),
            ])
        
        try:
            result = minimize(
                _log_mag_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            n_restarts += 1
            
            if result.fun < best_value:
                ln_a0, ln_a4, ln_a8, s0, s4, s8 = result.x
                a0 = np.sign(s0) * np.exp(ln_a0) if abs(s0) > 1e-10 else 0.0
                a4 = np.sign(s4) * np.exp(ln_a4) if abs(s4) > 1e-10 else 0.0
                a8 = np.sign(s8) * np.exp(ln_a8) if abs(s8) > 1e-10 else 0.0
                best_coeffs = np.array([a0, a4, a8])
                best_value = result.fun
        except Exception:
            continue
    
    # Polish with high-precision local optimization on the best found point
    if time.time() < deadline - 10.0:
        a0_b, a4_b, a8_b = best_coeffs
        x_polish = np.array([
            np.log(abs(a0_b)) if abs(a0_b) > 1e-15 else -5.0,
            np.log(abs(a4_b)) if abs(a4_b) > 1e-15 else -5.0,
            np.log(abs(a8_b)) if abs(a8_b) > 1e-15 else -5.0,
            np.sign(a0_b) if abs(a0_b) > 1e-15 else 1.0,
            np.sign(a4_b) if abs(a4_b) > 1e-15 else 1.0,
            np.sign(a8_b) if abs(a8_b) > 1e-15 else 1.0,
        ])
        
        try:
            result = minimize(
                _log_mag_objective,
                x_polish,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_value:
                ln_a0, ln_a4, ln_a8, s0, s4, s8 = result.x
                a0 = np.sign(s0) * np.exp(ln_a0) if abs(s0) > 1e-10 else 0.0
                a4 = np.sign(s4) * np.exp(ln_a4) if abs(s4) > 1e-10 else 0.0
                a8 = np.sign(s8) * np.exp(ln_a8) if abs(s8) > 1e-10 else 0.0
                best_coeffs = np.array([a0, a4, a8])
                best_value = result.fun
        except Exception:
            pass
    
    return best_coeffs
