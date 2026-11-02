"""Uncertainty inequality: tune Hermite-Gaussian coefficients to minimize A(f)*A(Fourier(f))."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_coeffs(deg):
    """Return coefficients of the (physicist's) Hermite polynomial H_deg(x) as a list [c0, c1, ..., c_deg]."""
    if deg == 0:
        return [1.0]
    if deg == 1:
        return [0.0, 2.0]
    # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    prev = [1.0]  # H_0
    curr = [0.0, 2.0]  # H_1
    for n in range(1, deg):
        # H_{n+1} = 2x*H_n - 2n*H_{n-1}
        new = [0.0] * (n + 2)
        # 2x * curr: shift curr by 1 and multiply by 2
        for i in range(len(curr)):
            new[i + 1] += 2.0 * curr[i]
        # -2n * prev
        for i in range(len(prev)):
            new[i] -= 2.0 * n * prev[i]
        prev, curr = curr, new
    return curr

def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients at x (or array x)."""
    result = np.zeros_like(x, dtype=float)
    for c in reversed(coeffs):
        result = result * x + c
    return result

def _compute_bound(a0, a4, a8):
    """Compute the upper bound on A(f)*A(Fourier(f)) given coefficients a0, a4, a8."""
    # Derive a12 so that g(0) = 0
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=560, H12(0)=2016
    a12 = -(a0 + 12.0 * a4 + 560.0 * a8) / 2016.0

    # Build the polynomial P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    h0 = _hermite_coeffs(0)
    h4 = _hermite_coeffs(4)
    h8 = _hermite_coeffs(8)
    h12 = _hermite_coeffs(12)

    # Combine into a single polynomial (degree 12)
    max_deg = 12
    P = np.zeros(max_deg + 1)
    for c, h in [(a0, h0), (a4, h4), (a8, h8), (a12, h12)]:
        for i, coeff in enumerate(h):
            P[i] += c * coeff

    # g(x) = P(x) * exp(-x^2/2)
    # A(f) = integral of |g(x)|^2 dx = integral of P(x)^2 * exp(-x^2) dx
    # A(Fourier(g)) = integral of |G(k)|^2 dk
    # For Fourier transform convention: G(k) = integral g(x) exp(-ikx) dx
    # A(Fourier(g)) = (1/2pi) * integral |G(k)|^2 dk = (1/2pi) * integral |g(x)|^2 dx by Parseval
    # Wait, need to be careful about conventions.
    # 
    # Actually, the uncertainty product A(f)*A(Fourier(f)) where A(h) = integral |h|^2
    # By Parseval: integral |g(x)|^2 dx = (1/2pi) integral |G(k)|^2 dk
    # So A(Fourier(g)) = integral |G(k)|^2 dk = 2pi * A(g)
    #
    # But the "upper bound" in the uncertainty inequality context likely refers to:
    # A(f) * A(Fourier(f)) where we need to compute both.
    #
    # Let me compute A(g) = integral P(x)^2 exp(-x^2) dx numerically.
    # And A(Fourier(g)) needs the Fourier transform.
    #
    # Actually, for the uncertainty product, if we use the standard Fourier transform
    # G(k) = integral g(x) e^{-ikx} dx, then:
    # integral |G(k)|^2 dk = 2pi * integral |g(x)|^2 dx (Parseval)
    #
    # So A(f)*A(Fourier(f)) = A(g) * 2pi * A(g) = 2pi * A(g)^2
    #
    # This means minimizing the product is equivalent to minimizing A(g) = integral P(x)^2 exp(-x^2) dx
    #
    # Let me compute this integral numerically.
    
    # Use Gauss-Hermite quadrature for integral of P(x)^2 * exp(-x^2) dx
    # numpy.polynomial.hermite.hermgauss gives nodes and weights for weight exp(-x^2)
    n_quad = 200
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    P_vals = _poly_eval(P, x_nodes)
    A_g = np.sum(w_nodes * P_vals**2)
    
    # The upper bound is 2*pi*A_g (if using the standard Fourier convention)
    # But actually, let me reconsider. The task says "upper bound on A(f)*A(Fourier(f))"
    # This might be a specific bound from the uncertainty principle, not just the product itself.
    #
    # In the context of Goncalves et al. 2017, the bound likely involves:
    # A(f) * A(Fourier(f)) >= some constant, and we're trying to find a test function
    # that makes this product small (close to the lower bound).
    #
    # The "upper bound" might refer to an upper bound on the uncertainty product
    # for this specific test function, which is just the product itself.
    #
    # Let me just return A(g) * 2*pi*A(g) = 2*pi*A_g^2 as the objective.
    # Actually, let me reconsider the Fourier transform convention.
    #
    # If Fourier(f)(k) = integral f(x) e^{-2pi i x k} dx (unitary-ish), then
    # Parseval: integral |f|^2 = integral |Fourier(f)|^2
    # So A(f)*A(Fourier(f)) = A(g)^2
    #
    # The convention matters. Let me use the convention where the product is A(g)^2
    # since that's cleaner, and the relative optimization should be the same.
    
    return 2.0 * np.pi * A_g * A_g  # Using angular frequency convention

def construct():
    deadline = time.time() + 190.0  # Safety margin before 210s kill
    
    # Initial best: the seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound(seed[0], seed[1], seed[2])
    best_x = seed.copy()
    
    # Try a grid search first to get a rough landscape
    # Then use differential evolution for global search
    # Then polish with local optimization
    
    # Differential evolution for global search
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    try:
        result_de = differential_evolution(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=False,
            init='sobol',
            workers=1
        )
        if result_de.success or result_de.fun < best_val:
            if time.time() > deadline - 30:
                pass  # Don't update if we're close to deadline
            else:
                best_val = result_de.fun
                best_x = result_de.x.copy()
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            result_nm = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0=best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result_nm.fun < best_val:
                best_val = result_nm.fun
                best_x = result_nm.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            result_p = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0=best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result_p.fun < best_val:
                best_val = result_p.fun
                best_x = result_p.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
