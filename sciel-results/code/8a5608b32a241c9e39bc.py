"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = 210.0
_SAFETY_MARGIN = 15.0  # seconds

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _hermite_gaussian(c0, c4, c8, x):
    """Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)."""
    c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 665280.0
    h0 = eval_hermite(0, x)
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    poly = c0*h0 + c4*h4 + c8*h8 + c12*h12
    return poly * np.exp(-x**2 / 2.0)

def _compute_C4(c0, c4, c8):
    """Compute C4 = A(g) * A(hat_g) where A(f) = integral |f|."""
    def g(x):
        return _hermite_gaussian(c0, c4, c8, x)
    
    # A(g) = integral |g(x)| dx over real line
    A_g, _ = quad(lambda x: abs(g(x)), -np.inf, np.inf, limit=300, epsabs=1e-12, epsrel=1e-12)
    
    # hat_g: since g is a combo of even Hermite functions, hat_g has same structure
    # but with (-i)^n factors. For even n, (-i)^n = 1, so hat_g = g in the same basis.
    # Actually, the Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}
    # for the unitary convention. For even n, this is just H_n(x) e^{-x^2/2}.
    # So hat_g = g. Thus A(hat_g) = A(g).
    # But let me verify numerically by computing the FT of g.
    # For safety, compute hat_g numerically via the fact that it's the same function.
    A_hat_g = A_g  # since hat_g = g for even Hermite combinations
    
    return A_g * A_hat_g

def construct():
    # Initial seed values
    seed_c = np.array([-113/100, 1/25, 1/3240])
    best_c = seed_c.copy()
    best_val = _compute_C4(*best_c)
    
    # Generate diverse starting points around the seed
    np.random.seed(42)
    starts = [best_c.copy()]
    
    # Add perturbations at different scales
    for scale in [0.1, 0.05, 0.02, 0.01, 0.005]:
        for _ in range(3):
            perturb = np.random.normal(0, scale, size=3)
            # Scale perturbations appropriately for each coefficient
            perturb[0] *= 0.5  # c0 has larger magnitude
            perturb[1] *= 1.0  # c4
            perturb[2] *= 1.0  # c8
            x0 = best_c + perturb
            starts.append(x0)
    
    # Also try some hand-crafted diverse points
    starts.extend([
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.05, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00025]),
        np.array([-0.95, 0.055, 0.00045]),
    ])
    
    # Phase 1: Nelder-Mead for broad exploration
    for x0 in starts:
        if _time_left() < 10.0:
            break
        try:
            res = minimize(_compute_C4, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: BFGS for local refinement from best point
    if _time_left() > 15.0:
        try:
            res_bfgs = minimize(_compute_C4, best_c, method='BFGS',
                               options={'maxiter': 200, 'gtol': 1e-10, 'disp': False})
            if res_bfgs.fun < best_val and np.isfinite(res_bfgs.fun):
                best_val = res_bfgs.fun
                best_c = res_bfgs.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish with very tight tolerances
    if _time_left() > 10.0:
        try:
            res_polish = minimize(_compute_C4, best_c, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res_polish.fun < best_val and np.isfinite(res_polish.fun):
                best_val = res_polish.fun
                best_c = res_polish.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_val = _compute_C4(*best_c)
    if np.isfinite(final_val) and final_val < best_val:
        best_val = final_val
    
    return best_c
# EVOLVE-BLOCK-END
