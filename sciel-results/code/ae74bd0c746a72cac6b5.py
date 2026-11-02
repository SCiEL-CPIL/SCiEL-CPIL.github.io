"""Optimized program: numerical minimization of the C4 upper bound using exact quadrature and root finding."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


def construct():
    """
    Numerically optimize the three free Hermite coefficients (a0, a4, a8)
    to minimize the upper bound on C4 = A(g) * A(ĝ), with the constraint
    g(0) = 0 enforced by solving for a12.
    
    Uses Gauss-Hermite quadrature for exact norm calculation and polynomial
    root finding for exact sup norm location.
    """
    # Pre-compute polynomial coefficients for H0, H4, H8, H12 in power basis (lowest degree first)
    # np.polynomial.hermite.herm2poly converts Hermite basis coeffs to power basis coeffs
    c_h0 = np.polynomial.hermite.herm2poly([1])
    c_h4 = np.polynomial.hermite.herm2poly([0, 0, 0, 0, 1])
    c_h8 = np.polynomial.hermite.herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 1])
    c_h12 = np.polynomial.hermite.herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    # Ensure all coefficient vectors have the same length (degree 12)
    max_deg = 12
    c_h0 = np.pad(c_h0, (0, max_deg - len(c_h0) + 1))
    c_h4 = np.pad(c_h4, (0, max_deg - len(c_h4) + 1))
    c_h8 = np.pad(c_h8, (0, max_deg - len(c_h8) + 1))
    c_h12 = np.pad(c_h12, (0, max_deg - len(c_h12) + 1))
    
    # Gauss-Hermite quadrature nodes and weights
    # n=13 is sufficient for exact integration of (degree 12 polynomial)^2 = degree 24
    x_gh, w_gh = np.polynomial.hermite.hermgauss(13)
    
    def objective(coeffs):
        a0, a4, a8 = coeffs
        # Enforce P(0) = 0
        # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
        # a0*1 + a4*3 + a8*105 + a12*10395 = 0
        a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 10395.0
        
        # Construct P(x) coefficients
        c_poly = a0 * c_h0 + a4 * c_h4 + a8 * c_h8 + a12 * c_h12
        
        # Calculate L2 norm squared: ||g||_2^2 = integral P(x)^2 e^(-x^2) dx
        # Using GH quadrature
        p_vals = np.polynomial.polynomial.polyval(x_gh, c_poly)
        norm_sq = np.sum(w_gh * p_vals**2)
        
        if norm_sq < 1e-12:
            return 1e10
            
        norm = np.sqrt(norm_sq)
        
        # Calculate Sup norm
        # g(x) = P(x) e^(-x^2/2)
        # g'(x) = e^(-x^2/2) (P'(x) - x P(x))
        # Find roots of Q(x) = P'(x) - x P(x)
        
        c_dp = np.polynomial.polynomial.polyder(c_poly)
        c_xp = np.polynomial.polynomial.polymul(c_poly, [0, 1])
        
        # Pad c_dp to match length of c_xp
        if len(c_dp) < len(c_xp):
            c_dp = np.pad(c_dp, (0, len(c_xp) - len(c_dp)))
            
        c_q = c_dp - c_xp
        
        # Find roots
        try:
            roots = np.polynomial.polynomial.polyroots(c_q)
        except Exception:
            return 1e10
            
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        
        max_val = 0.0
        for r in real_roots:
            p_r = np.polynomial.polynomial.polyval(r, c_poly)
            g_r = abs(p_r * np.exp(-r**2 / 2.0)) / norm
            if g_r > max_val:
                max_val = g_r
        
        # The bound is A(g)^2
        return max_val**2

    # Initial guess
    # Based on typical optimal shapes, coefficients are small
    seed_coeffs = np.array([1.0, -0.1, 0.01])
    
    # Bounds
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0)]
    
    # Global optimization
    best_val = np.inf
    best_x = seed_coeffs.copy()
    
    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            atol=1e-10,
            init='sobol',
            polish=True
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_x = de_result.x
    except Exception:
        pass
    
    # Local refinement
    try:
        local_result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        if local_result.fun < best_val:
            best_val = local_result.fun
            best_x = local_result.x
    except Exception:
        pass
        
    # Additional local refinement from best point
    try:
        final_result = minimize(
            objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if final_result.fun < best_val:
            best_val = final_result.fun
            best_x = final_result.x
    except Exception:
        pass

    return np.array(best_x)
# EVOLVE-BLOCK-END
