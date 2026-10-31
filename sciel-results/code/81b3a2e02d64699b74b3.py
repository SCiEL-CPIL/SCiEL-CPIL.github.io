"""Optimized program for uncertainty inequality test function tuning."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_prob(x, n):
    """Compute probabilists' Hermite polynomial Hn(x) using recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    Hn2 = np.ones_like(x)
    Hn1 = x.copy()
    for k in range(2, n + 1):
        Hn = k * x * Hn1 - (k - 1) * Hn2
        Hn2 = Hn1
        Hn1 = Hn
    return Hn1


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound A(g)*A(F(g)) for given coefficients."""
    # Derive a12 from g(0) = 0
    a12 = (a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Position-space grid
    x = np.linspace(-15.0, 15.0, 30001)
    dx = x[1] - x[0]

    gauss = np.exp(-x**2 / 2.0)
    H0 = _hermite_prob(x, 0)
    H4 = _hermite_prob(x, 4)
    H8 = _hermite_prob(x, 8)
    H12 = _hermite_prob(x, 12)
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss

    g2 = g**2
    N0 = np.sum(g2) * dx
    if abs(N0) < 1e-15:
        return 1e10
    N2 = np.sum(x**2 * g2) * dx
    A_g = np.sqrt(N2 / N0)

    # Fourier transform via FFT
    N_fft = 2**15
    x_fft = np.linspace(-15.0, 15.0, N_fft)
    dx_fft = x_fft[1] - x_fft[0]
    gauss_fft = np.exp(-x_fft**2 / 2.0)
    H0_fft = _hermite_prob(x_fft, 0)
    H4_fft = _hermite_prob(x_fft, 4)
    H8_fft = _hermite_prob(x_fft, 8)
    H12_fft = _hermite_prob(x_fft, 12)
    g_fft = (a0 * H0_fft + a4 * H4_fft + a8 * H8_fft + a12 * H12_fft) * gauss_fft

    G = np.fft.fftshift(np.fft.fft(g_fft * dx_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dx_fft))

    G2 = np.abs(G)**2
    dxi = freqs[1] - freqs[0]
    M0 = np.sum(G2) * dxi
    if abs(M0) < 1e-15:
        return 1e10
    M2 = np.sum(freqs**2 * G2) * dxi
    A_Fg = np.sqrt(M2 / M0)

    return A_g * A_Fg


def construct():
    """Return three Hermite coefficients (degrees 0, 4, 8) minimizing the bound."""
    deadline = time.time() + 170.0

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _compute_bound(*seed)
    best_params = seed.copy()

    def lp_to_coeffs(lp):
        m0, m4, m8, s0, s4, s8 = lp
        return np.array([s0 * np.exp(m0), s4 * np.exp(m4), s8 * np.exp(m8)])

    def coeffs_to_lp(c):
        mags = np.maximum(np.abs(c), 1e-10)
        signs = np.where(np.sign(c) == 0, 1.0, np.sign(c))
        return np.concatenate([np.log(mags), signs])

    def objective(lp):
        if time.time() > deadline - 3.0:
            return best_score
        c = lp_to_coeffs(lp)
        if np.any(np.abs(c) > 100):
            return 1e10
        return _compute_bound(*c)

    lp0 = coeffs_to_lp(seed)

    try:
        if time.time() < deadline - 50.0:
            r = minimize(objective, lp0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if r.fun < best_score:
                best_score, best_params = r.fun, lp_to_coeffs(r.x)
    except Exception:
        pass

    try:
        if time.time() < deadline - 50.0:
            r = minimize(objective, lp0, method='Powell',
                         options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-12})
            if r.fun < best_score:
                best_score, best_params = r.fun, lp_to_coeffs(r.x)
    except Exception:
        pass

    try:
        if time.time() < deadline - 30.0:
            rng = np.random.RandomState(123)
            for _ in range(10):
                if time.time() > deadline - 25.0:
                    break
                lp_s = lp0 + rng.normal(0, 1.0, 6)
                lp_s[:3] = np.clip(lp_s[:3], -15, 5)
                lp_s[3:] = np.clip(lp_s[3:], -1, 1)
                r = minimize(objective, lp_s, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                if r.fun < best_score:
                    best_score, best_params = r.fun, lp_to_coeffs(r.x)
    except Exception:
        pass

    try:
        if time.time() < deadline - 5.0:
            lp_best = coeffs_to_lp(best_params)
            r = minimize(objective, lp_best, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if r.fun < best_score:
                best_score, best_params = r.fun, lp_to_coeffs(r.x)
    except Exception:
        pass

    return best_params

# EVOLVE-BLOCK-END
