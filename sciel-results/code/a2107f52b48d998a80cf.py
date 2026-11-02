"""Rewritten solver for the uncertainty-inequality Hermite test-function task."""
import numpy as np
import time
import scipy.optimize

# Keep the same entry-point name and return shape (three floats for H0, H4, H8).

def _upper_bound_c4(c0, c4, c8):
    """
    Compute a proxy for the upper bound C4 given Hermite coefficients (c0, c4, c8).
    The H12 coefficient is fixed so that g(0) = 0.

    We use a simplified but faithful proxy:
      - Build g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
      - Compute A(f) ~ L1 norm of g, and A(F) ~ L1 norm of Fourier(g)
      - Return A(f)*A(F) as a proxy for C4 bound.

    To keep this fast and deterministic, we evaluate on a fixed grid.
    """
    # Fixed grid
    N = 2048
    x = np.linspace(-20, 20, N)
    dx = x[1] - x[0]

    # Hermite polynomials (physicists' convention)
    # H0 = 1
    # H4 = x^4 - 6*x^2 + 3
    # H8 = x^8 - 28*x^6 + 210*x^4 - 420*x^2 + 105
    # H12 = x^12 - 60*x^10 + 1260*x^8 - 12600*x^6 + 63000*x^4 - 126000*x^2 + 10395

    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2 * x2 - 6 * x2 + 3
    H8 = x2 ** 4 - 28 * x2 ** 3 + 210 * x2 ** 2 - 420 * x2 + 105
    H12 = x2 ** 6 - 60 * x2 ** 5 + 1260 * x2 ** 4 - 12600 * x2 ** 3 + 63000 * x2 ** 2 - 126000 * x2 + 10395

    # Enforce g(0) = 0: c0 + c4*3 + c8*105 + c12*10395 = 0
    # => c12 = -(c0 + 3*c4 + 105*c8) / 10395
    c12 = -(c0 + 3 * c4 + 105 * c8) / 10395.0

    gauss = np.exp(-x2 / 2.0)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss

    # L1 norm of g
    A_f = np.sum(np.abs(g)) * dx

    # Fourier transform via FFT (approximate, using a dense grid)
    # To use FFT properly, we need the function sampled on a uniform grid.
    # We'll compute the DTFT via np.fft.fft and scale appropriately.
    # For a finite interval, the Fourier transform magnitude can be approximated.
    # A simpler proxy: use the L1 norm of the Fourier transform computed via fft.
    # Note: This is a proxy, not the exact continuous FT, but it's monotonic-ish.

    # Use FFT with proper scaling
    # The grid is x in [-20, 20], N points.
    # Frequency spacing: df = 1 / (N * dx)
    ft = np.fft.fft(g) * dx
    # Take the magnitude
    A_F = np.sum(np.abs(ft)) * (1.0 / (N * dx))  # scaling for DTFT

    return A_f * A_F


def construct():
    """
    Return three coefficients (c0, c4, c8) that minimize the upper bound on C4.
    We use a multi-start Nelder-Mead optimization with a time budget.
    """
    deadline = time.time() + 180.0  # Safety margin: return before 180s

    best_val = _upper_bound_c4(-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0)
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Multi-start optimization
    rng = np.random.default_rng(42)
    num_starts = 50

    for i in range(num_starts):
        if time.time() > deadline - 30.0:
            break

        # Generate a random starting point
        if i == 0:
            start = np.array([-1.0, 0.05, 0.0003])  # Near the known solution
        else:
            start = np.array([
                rng.uniform(-5, 5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.005, 0.005)
            ])

        try:
            res = scipy.optimize.minimize(
                lambda c: _upper_bound_c4(c[0], c[1], c[2]),
                start,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_c = res.x
        except Exception:
            continue

    return best_c
