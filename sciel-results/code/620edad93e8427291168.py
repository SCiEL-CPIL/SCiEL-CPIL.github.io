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


def _objective_fast(coeffs):
    """Fast uncertainty concentration proxy using analytical FT of Hermite-Gaussians."""
    a0, a4, a8 = coeffs
    # Calculate a12 such that g(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
    
    # Grid for x
    x = np.linspace(-10.0, 10.0, 4096)
    
    # Calculate g(x)
    H0 = _phys_hermite(0, x)
    H4 = _phys_hermite(4, x)
    H8 = _phys_hermite(8, x)
    H12 = _phys_hermite(12, x)
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-0.5 * x * x)
    
    A_g = np.trapz(np.abs(g), x)
    
    # Calculate g_hat(xi) using analytical FT
    # Convention: F(xi) = int g(x) exp(-2 pi i x xi) dx
    # Identity: F[H_n(x) exp(-x^2/2)](xi) = sqrt(2*pi) * (-i)^n * H_n(2*pi*xi) * exp(-(2*pi*xi)^2/2)
    # For even n, (-i)^n = 1.
    
    xi = np.linspace(-10.0, 10.0, 4096)
    xi_scaled = 2.0 * np.pi * xi
    gauss_hat = np.exp(-0.5 * xi_scaled * xi_scaled)
    
    H0_hat = _phys_hermite(0, xi_scaled)
    H4_hat = _phys_hermite(4, xi_scaled)
    H8_hat = _phys_hermite(8, xi_scaled)
    H12_hat = _phys_hermite(12, xi_scaled)
    
    poly_hat = a0 * H0_hat + a4 * H4_hat + a8 * H8_hat + a12 * H12_hat
    g_hat = np.sqrt(2.0 * np.pi) * poly_hat * gauss_hat
    
    A_ghat = np.trapz(np.abs(g_hat), xi)
    
    return A_g * A_ghat


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    try:
        from scipy.optimize import minimize, differential_evolution
        
        # Define bounds for the coefficients
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (-1.0, 1.0)]
        
        # 1. Try Differential Evolution for a global-ish search
        de_res = differential_evolution(
            _objective_fast,
            bounds,
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-6
        )
        
        best_x = de_res.x
        best_val = de_res.fun
        
        # 2. Refine with L-BFGS-B from the best DE point
        res = minimize(
            _objective_fast,
            best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8}
        )
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
        # 3. Also try refining from the original seed
        res_seed = minimize(
            _objective_fast,
            seed,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8}
        )
        if res_seed.fun < best_val:
            best_x = res_seed.x
            best_val = res_seed.fun
            
        return best_x

    except Exception:
        return seed
# EVOLVE-BLOCK-END
