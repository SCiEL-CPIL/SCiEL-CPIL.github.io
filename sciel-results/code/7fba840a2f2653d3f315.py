"""Rewritten program: numerically optimize the Hermite-Gaussian coefficients
to reduce the uncertainty (Fourier concentration) upper bound C4.

New approach family: instead of returning fixed analytic constants, we run a
numerical local optimization (L-BFGS-B) over the three free coefficients,
starting from the Goncalves et al. (2017) state-of-the-art point.  The
objective reconstructs the test function g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)
exp(-x^2/2) with a12 fixed by g(0)=0, then measures the product of the L1 mass
of g and of its Fourier transform, a standard uncertainty concentration
quantity that the C4 bound is built from.  Lower product => tighter bound.
"""
# EVOLVE-BLOCK-START
import numpy as np


def _phys_hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _build_g(coeffs, x):
    """Return g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2).

    a12 is chosen so that g(0) = 0.
    """
    a0, a4, a8 = coeffs
    H0 = _phys_hermite(0, x)
    H4 = _phys_hermite(4, x)
    H8 = _phys_hermite(8, x)
    H12 = _phys_hermite(12, x)
    H0_0, H4_0, H8_0, H12_0 = 1.0, 3.0, 105.0, 10395.0
    # g(0) = (a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0) * 1 = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    return poly * np.exp(-0.5 * x * x)


def _objective(coeffs):
    """Uncertainty concentration proxy: A(g) * A(Fourier(g))."""
    x = np.linspace(-8.0, 8.0, 4096)
    g = _build_g(coeffs, x)
    # L1 mass of g
    A_g = np.trapz(np.abs(g), x)
    # Fourier transform (unitary-ish, constant factors cancel in the product
    # up to a fixed scale, so we keep a consistent convention).
    g_hat = np.fft.fftshift(np.fft.fft(g * np.exp(1j * 0.0) ))
    # Build frequency axis matching x spacing
    dx = x[1] - x[0]
    n = len(x)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    # Use a proper continuous FT via trapezoid in the integral form for accuracy
    # (FFT of a non-periodic truncated function has edge artifacts; do direct
    #  integration on a coarser frequency grid to stay faithful).
    xi = np.linspace(-12.0, 12.0, 4096)
    # Direct FT: g_hat(xi) = int g(x) exp(-2 pi i x xi) dx
    # Vectorized in chunks to bound memory.
    g_hat = np.zeros_like(xi, dtype=complex)
    chunk = 256
    for i0 in range(0, len(xi), chunk):
        xi_c = xi[i0:i0 + chunk]
        phase = np.exp(-2j * np.pi * np.outer(x, xi_c))  # (len(x), chunk)
        g_hat[i0:i0 + chunk] = np.trapz(g[:, None] * phase, x, axis=0)
    A_ghat = np.trapz(np.abs(g_hat), xi)
    return A_g * A_ghat


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    try:
        from scipy.optimize import minimize
        base = _objective(seed)
        best = seed
        best_val = base
        # A few restarts around the seed to escape any poor local basin.
        rng = np.random.default_rng(0)
        for attempt in range(4):
            if attempt == 0:
                x0 = seed
            else:
                x0 = seed + rng.normal(0.0, 0.05, size=3)
            res = minimize(
                _objective,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.success or res.fun < best_val:
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _objective(cand)
                    if np.isfinite(v) and v < best_val:
                        best_val = v
                        best = cand
        return best
    except Exception:
        # Fall back to the known-good state of the art if anything fails.
        return seed
# EVOLVE-BLOCK-END
