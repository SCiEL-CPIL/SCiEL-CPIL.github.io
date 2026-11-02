"""New program: minimize the uncertainty bound using L¹ norms via adaptive quadrature
and Basin Hopping global search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import basinhopping, minimize
from scipy.integrate import quad


def _compute_g_scalar(x, coeffs):
    """Compute g(x) at scalar x."""
    c0, c1, c2 = coeffs[0], coeffs[1], coeffs[2]
    c3 = -(c0 + c1 * (-12.0) + c2 * 168.0) / 792.0
    h = np.exp(-0.5 * x * x)
    h0 = 1.0
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    g = (c0 * h0 + c1 * h4 + c2 * h8 + c3 * h12) * h
    return g


def _l1_norm(coeffs):
    """Compute L¹ norm of g via adaptive quadrature."""
    def integrand(x):
        g = _compute_g_scalar(x, coeffs)
        return abs(g)
    val, err = quad(integrand, -8.0, 8.0, limit=100, epsabs=1e-10, epsrel=1e-10)
    return val


def _objective(coeffs):
    """Objective: (L¹ norm of g)² since ĝ = g for our Hermite degrees."""
    l1 = _l1_norm(coeffs)
    return l1 * l1


def construct():
    """Construct optimal Hermite coefficients via Basin Hopping."""
    starting_points = [
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.8, 0.05, 0.0]),
        np.array([-1.2, 0.03, -0.0005]),
    ]
    
    best_val = 1e12
    best_x = starting_points[0]
    
    for x0 in starting_points:
        try:
            bh = basinhopping(_objective, x0, niter=50, stepsize=0.4,
                            minimizer_kwargs={'method': 'Nelder-Mead',
                                             'options': {'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11}},
                            accept_test=lambda new, old: new.fun <= old.fun + 1e-7)
            if bh.fun < best_val:
                best_val = bh.fun
                best_x = bh.x.copy()
        except Exception:
            pass
    
    # Final polish
    try:
        r = minimize(_objective, best_x, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13})
        if r.fun < best_val:
            best_x = r.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
