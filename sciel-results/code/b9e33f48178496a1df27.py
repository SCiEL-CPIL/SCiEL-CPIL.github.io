"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x) as array [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    coeffs = np.zeros(n + 1)
    coeffs[n] = 2.0 ** n
    if n >= 2:
        coeffs[n - 2] = -2 * n * (n - 1) * 2.0 ** (n - 2)
    # General recurrence: H_n(x) = 2x*H_{n-1}(x) - 2(n-1)*H_{n-2}(x)
    h_prev2 = np.array([1.0])  # H_0
    h_prev1 = np.array([0.0, 2.0])  # H_1
    for k in range(2, n + 1):
        h_curr = np.zeros(k + 1)
        # 2x * H_{k-1}(x)
        h_curr[1:] += 2.0 * h_prev1[:k]
        # -2(k-1) * H_{k-2}(x)
        h_curr[:k-1] -= 2.0 * (k - 1) * h_prev2[:k-1]
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_prev1


def _poly_eval(coeffs, x):
    """Evaluate polynomial with coeffs [a0, a1, ..., an] at points x."""
    result = np.zeros_like(x, dtype=float)
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result


def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 for given coefficients c0, c4, c8."""
    # Hermite polynomial coefficients
    h0 = _hermite_coeffs(0)   # [1]
    h4 = _hermite_coeffs(4)   # H4(x) = 16x^4 - 48x^2 + 12
    h8 = _hermite_coeffs(8)
    h12 = _hermite_coeffs(12)
    
    # c12 from constraint g(0) = 0
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    h0_0 = _poly_eval(h0, np.array([0.0]))[0]
    h4_0 = _poly_eval(h4, np.array([0.0]))[0]
    h8_0 = _poly_eval(h8, np.array([0.0]))[0]
    h12_0 = _poly_eval(h12, np.array([0.0]))[0]
    
    if abs(h12_0) < 1e-15:
        return 1e10  # degenerate
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Build g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # We need to compute integrals related to the uncertainty bound
    
    # For the Fourier uncertainty inequality with Gaussian-weighted Hermite functions,
    # the relevant quantity involves moments of |g(x)|^2 and |G(k)|^2
    
    # Let's compute numerically
    x = np.linspace(-20, 20, 200000)
    dx = x[1] - x[0]
    
    g_poly = c0 * _poly_eval(h0, x) + c4 * _poly_eval(h4, x) + c8 * _poly_eval(h8, x) + c12 * _poly_eval(h12, x)
    g = g_poly * np.exp(-x**2 / 2)
    
    # L2 norm squared
    norm_sq = np.sum(g**2) * dx
    
    if norm_sq < 1e-15:
        return 1e10
    
    # Mean and variance of |g|^2
    x2 = x**2
    weight = g**2 / norm_sq
    
    mean_x = np.sum(x * weight) * dx
    var_x = np.sum((x - mean_x)**2 * weight) * dx
    
    # Fourier transform of g
    # G(k) = integral g(x) exp(-i*k*x) dx
    # For efficiency, use FFT
    n_pts = len(x)
    if n_pts % 2 == 1:
        g = g[:-1]
        x = x[:-1]
        n_pts = len(x)
    
    G = np.fft.fftshift(np.fft.fft(g * dx)) * np.sqrt(dx)
    k = 2 * np.pi * np.fft.fftfreq(n_pts, d=dx)
    k = np.fft.fftshift(k)
    
    norm_G_sq = np.sum(G**2) * dx
    
    if norm_G_sq < 1e-15:
        return 1e10
    
    weight_G = G**2 / norm_G_sq
    mean_k = np.sum(k * weight_G) * dx
    var_k = np.sum((k - mean_k)**2 * weight_G) * dx
    
    # The uncertainty product bound
    # A(f) * A(F) ~ sqrt(var_x) * sqrt(var_k) or similar
    # C4 is the constant in the inequality
    
    # The "upper bound" we're trying to minimize is likely:
    # C4 = (some functional of g that bounds the uncertainty product)
    
    # For the standard Heisenberg-type inequality:
    # var_x * var_k >= 1/4
    # But here we have a specific test function and the bound C4 depends on it
    
    # Based on the task, C4 is computed from the test function properties
    # Let's use: C4 = (integral of x^2 |g|^2 / integral |g|^2) * (integral of k^2 |G|^2 / integral |G|^2)
    # This is the product of variances (assuming zero mean)
    
    second_moment_x = np.sum(x2 * weight) * dx
    second_moment_k = np.sum(k**2 * weight_G) * dx
    
    C4 = second_moment_x * second_moment_k
    
    return C4


def construct():
    t_start = time.time()
    deadline = t_start + 190.0  # safety margin before 210s kill
    
    # Starting point: known good coefficients
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(best_c[0], best_c[1], best_c[2])
    
    # Try local optimization with Nelder-Mead
    def objective(c):
        return _compute_bound(c[0], c[1], c[2])
    
    # First, try a few perturbed points to get a sense of the landscape
    rng = np.random.default_rng(42)
    
    for i in range(20):
        if time.time() > deadline:
            break
        perturbation = rng.normal(0, 0.1, size=3)
        c_trial = best_c + perturbation
        val = _compute_bound(c_trial[0], c_trial[1], c_trial[2])
        if val < best_val:
            best_val = val
            best_c = c_trial.copy()
    
    # Now try Nelder-Mead from the best point found
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Try a few more local optimizations from different starting points
    for i in range(5):
        if time.time() > deadline - 10:
            break
        c_start = best_c + rng.normal(0, 0.05, size=3)
        try:
            result = minimize(
                objective,
                c_start,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
