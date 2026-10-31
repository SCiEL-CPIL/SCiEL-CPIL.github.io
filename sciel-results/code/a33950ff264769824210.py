"""Optimized program for uncertainty inequality test function tuning."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _integrate(y, x):
    """Compute integral using numpy trapezoid rule, compatible with NumPy 2.0+."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    else:
        return np.trapz(y, x)

def _hermite_prob(x, n):
    """Compute probabilists' Hermite polynomial Hn(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    elif n == 2:
        return x**2 - 1
    elif n == 3:
        return x**3 - 3*x
    elif n == 4:
        return x**4 - 6*x**2 + 3
    elif n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    elif n == 12:
        return x**12 - 66*x**10 + 1540*x**8 - 15400*x**6 + 69300*x**4 - 118800*x**2 + 945
    else:
        raise ValueError(f"Unsupported Hermite degree: {n}")

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound for given coefficients."""
    # Derive a12 from g(0) = 0
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=-945 (probabilists)
    a12 = (a0 + 3*a4 + 105*a8) / 945.0
    
    # Define the test function on a grid
    x = np.linspace(-10, 10, 10001)
    dx = x[1] - x[0]
    
    # Gaussian envelope: exp(-x^2/4)
    gauss = np.exp(-x**2 / 4.0)
    
    # Hermite polynomials
    H0 = _hermite_prob(x, 0)
    H4 = _hermite_prob(x, 4)
    H8 = _hermite_prob(x, 8)
    H12 = _hermite_prob(x, 12)
    
    # Test function
    g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
    
    # Compute moments of g
    g2 = g**2
    N0 = _integrate(g2, x)
    if abs(N0) < 1e-15:
        return 1e10  # Degenerate
    
    N2 = _integrate(x**2 * g2, x)
    N4 = _integrate(x**4 * g2, x)
    
    # Fourier transform via FFT (pad for accuracy)
    N_fft = 2**16
    x_fft = np.linspace(-10, 10, N_fft)
    dx_fft = x_fft[1] - x_fft[0]
    gauss_fft = np.exp(-x_fft**2 / 4.0)
    H0_fft = _hermite_prob(x_fft, 0)
    H4_fft = _hermite_prob(x_fft, 4)
    H8_fft = _hermite_prob(x_fft, 8)
    H12_fft = _hermite_prob(x_fft, 12)
    g_fft = (a0*H0_fft + a4*H4_fft + a8*H8_fft + a12*H12_fft) * gauss_fft
    
    # FFT to get Fourier transform (using exp(-2πixξ) convention)
    G_fft = np.fft.fftshift(np.fft.fft(g_fft * dx_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dx_fft))
    
    G2 = np.abs(G_fft)**2
    M0 = _integrate(G2, freqs)
    if abs(M0) < 1e-15:
        return 1e10
    
    M2 = _integrate(freqs**2 * G2, freqs)
    M4 = _integrate(freqs**4 * G2, freqs)
    
    # Uncertainty product: A(g) * A(F(g))
    # A(f) = sqrt(∫x^2|f|^2 / ∫|f|^2)
    A_g = np.sqrt(N2 / N0)
    A_Fg = np.sqrt(M2 / M0)
    
    bound = A_g * A_Fg
    return bound

def construct():
    """Find optimal Hermite coefficients to minimize the uncertainty bound."""
    deadline = time.time() + 180.0  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _compute_bound(*seed)
    best_params = seed.copy()
    
    # Log-magnitude parameterization: a_i = sign_i * exp(m_i)
    # Handle zero/negative by using sign and log-abs
    def log_params_to_coeffs(lp):
        """Convert (log_mag0, log_mag4, log_mag8, sign0, sign4, sign8) to coefficients."""
        m0, m4, m8, s0, s4, s8 = lp
        a0 = s0 * np.exp(m0)
        a4 = s4 * np.exp(m4)
        a8 = s8 * np.exp(m8)
        return np.array([a0, a4, a8])
    
    def coeffs_to_log_params(coeffs):
        """Convert coefficients to log-magnitude representation."""
        mags = np.abs(coeffs)
        signs = np.sign(coeffs)
        # Handle zeros
        mags = np.maximum(mags, 1e-10)
        signs = np.where(signs == 0, 1.0, signs)
        return np.concatenate([np.log(mags), signs])
    
    # Objective in log-parameter space
    def objective(lp):
        if time.time() > deadline - 5.0:
            return best_score
        coeffs = log_params_to_coeffs(lp)
        # Bound the coefficients to avoid numerical issues
        if np.any(np.abs(coeffs) > 100):
            return 1e10
        return _compute_bound(*coeffs)
    
    # Start from seed in log-parameter space
    initial_lp = coeffs_to_log_params(seed)
    
    # Try multiple optimization strategies
    # 1. Nelder-Mead in log-parameter space
    try:
        if time.time() < deadline - 30.0:
            res = minimize(objective, initial_lp, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = log_params_to_coeffs(res.x)
    except Exception:
        pass
    
    # 2. Powell method
    try:
        if time.time() < deadline - 30.0:
            res = minimize(objective, initial_lp, method='Powell',
                          options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = log_params_to_coeffs(res.x)
    except Exception:
        pass
    
    # 3. Bounded differential evolution (coarse global search)
    try:
        if time.time() < deadline - 60.0:
            # Bounds in log-parameter space
            # Coefficients roughly in [-10, 10] -> log_mag in [log(1e-10), log(10)]
            bounds = [(-23, 2.3)] * 3 + [(-1, 1)] * 3
            res = differential_evolution(objective, bounds, seed=42, 
                                         maxiter=200, tol=1e-10,
                                         popsize=15, polish=True)
            if res.fun < best_score:
                best_score = res.fun
                best_params = log_params_to_coeffs(res.x)
    except Exception:
        pass
    
    # 4. Local refinement from best found
    try:
        if time.time() < deadline - 10.0:
            best_lp = coeffs_to_log_params(best_params)
            res = minimize(objective, best_lp, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = log_params_to_coeffs(res.x)
    except Exception:
        pass
    
    return best_params

# EVOLVE-BLOCK-END
