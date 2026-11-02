"""Optimized program: numerical minimization of the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _precompute_hermite(x):
    """Precompute H0, H4, H8, H12 on grid x."""
    h0 = np.ones_like(x)
    h1 = x.copy()
    h2 = x * h1 - 1.0 * h0  # H2
    h3 = x * h2 - 2.0 * h1  # H3
    h4 = x * h3 - 3.0 * h2  # H4
    h5 = x * h4 - 4.0 * h3  # H5
    h6 = x * h5 - 5.0 * h4  # H6
    h7 = x * h6 - 6.0 * h5  # H7
    h8 = x * h7 - 7.0 * h6  # H8
    h9 = x * h8 - 8.0 * h7  # H9
    h10 = x * h9 - 9.0 * h8  # H10
    h11 = x * h10 - 10.0 * h9  # H11
    h12 = x * h11 - 11.0 * h10  # H12
    return h0, h4, h8, h12


def construct():
    """
    Numerically optimize the three free Hermite coefficients (a0, a4, a8)
    to minimize the upper bound on C4 = A(g) * A(ĝ), with the constraint
    g(0) = 0 enforced by solving for a12.
    """
    # Precompute grid and Hermite polynomials
    # Use a sufficiently large grid for accuracy, but not too large for speed
    x = np.linspace(-12, 12, 10001)
    dx = x[1] - x[0]
    h0, h4, h8, h12 = _precompute_hermite(x)
    gauss = np.exp(-x ** 2 / 2.0)
    
    def objective(coeffs):
        a0, a4, a8 = coeffs
        try:
            # Correct constraint: g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
            # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
            # a0 + 3*a4 + 105*a8 + 945*a12 = 0
            a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
            
            poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
            g = poly * gauss
            
            # Calculate L2 norm
            norm_sq = np.dot(g, g) * dx
            if norm_sq < 1e-12:
                return 1e10
            
            norm = np.sqrt(norm_sq)
            g_norm = g / norm
            
            # Sup norm
            A_g = np.max(np.abs(g_norm))
            
            # Since g is composed of H0, H4, H8, H12 (all n divisible by 4),
            # and F[H_n(x)e^{-x^2/2}] = i^n H_n(x)e^{-x^2/2} = H_n(x)e^{-x^2/2} for n%4==0,
            # g is its own Fourier transform (under unitary convention).
            # Thus A(g) == A(ĝ).
            # Objective is A(g) * A(ĝ) = A(g)^2.
            return A_g * A_g
        except Exception:
            return 1e10
    
    # Multiple starting points for global search
    # The optimal solution for minimizing sup-norm of normalized Hermite-Gaussians with g(0)=0
    # is likely near the roots of the derivative or specific symmetric configurations.
    np.random.seed(42)
    starts = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.5, 0.1, 0.0]),
        np.array([-0.8, -0.05, 0.0]),
        np.array([-1.2, 0.05, -0.001]),
        np.array([0.0, 0.0, 0.0]), # Degenerate, but good start
        np.array([-2.0, 0.2, 0.01]),
        np.array([-0.5, -0.1, 0.0]),
    ]
    
    # Add random starts
    for _ in range(15):
        starts.append(np.array([
            np.random.uniform(-3.0, 3.0),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.05, 0.05)
        ]))
    
    best_val = np.inf
    best_x = starts[0].copy()
    
    # Local optimization from each start
    for start in starts:
        try:
            res = minimize(
                objective,
                start,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Differential evolution for global search
    try:
        de_result = differential_evolution(
            objective,
            bounds=[(-5.0, 5.0), (-1.0, 1.0), (-0.2, 0.2)],
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            atol=1e-12,
            polish=True
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_x = de_result.x.copy()
    except Exception:
        pass
    
    # Final high-precision refinement
    try:
        final_result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16}
        )
        if final_result.fun < best_val:
            best_val = final_result.fun
            best_x = final_result.x.copy()
    except Exception:
        pass
    
    # One more Nelder-Mead for very fine tuning
    try:
        nm_result = minimize(
            objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-16}
        )
        if nm_result.fun < best_val:
            best_val = nm_result.fun
            best_x = nm_result.x.copy()
    except Exception:
        pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
