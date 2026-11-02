"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.special import eval_hermite

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 200  # seconds, safety margin before 210s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _hermite_h(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)

def _construct_g(c0, c4, c8, x):
    """
    Construct g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    H0(0) = 1
    H4(0) = -12
    H8(0) = 168
    H12(0) = -4960
    
    So c12 = -(c0*1 + c4*(-12) + c8*168) / (-4960)
         = (c0 - 12*c4 + 168*c8) / 4960
    """
    c12 = (c0 - 12*c4 + 168*c8) / 4960.0
    gaussian = np.exp(-x**2 / 2.0)
    poly = (c0 * _hermite_h(0, x) +
            c4 * _hermite_h(4, x) +
            c8 * _hermite_h(8, x) +
            c12 * _hermite_h(12, x))
    return poly * gaussian

def _fourier_transform_coefficients(c0, c4, c8):
    """
    For a Fourier-invariant setup (Fourier transform convention where 
    F(f)(xi) = integral f(x) e^{-2*pi*i*x*xi} dx), the Fourier transform 
    of H_n(x) * exp(-x^2/2) is related to H_n(xi) * exp(-xi^2/2) up to 
    constants depending on the convention.
    
    Actually, let's use the convention that makes the problem Fourier-invariant.
    With F(f)(xi) = (1/sqrt(2*pi)) * integral f(x) exp(-i*x*xi) dx,
    we have F[exp(-x^2/2)] = exp(-xi^2/2)
    and F[H_n(x) exp(-x^2/2)] = i^n H_n(xi) exp(-xi^2/2)
    
    So F[g](xi) = (c0*H0(xi) + i^4*c4*H4(xi) + i^8*c8*H8(xi) + i^12*c12*H12(xi)) * exp(-xi^2/2)
    = (c0*H0(xi) + c4*H4(xi) + c8*H8(xi) + c12*H12(xi)) * exp(-xi^2/2)
    since i^4 = i^8 = i^12 = 1.
    
    So the Fourier transform has the same functional form with the same coefficients!
    This is the Fourier-invariance.
    """
    # Same coefficients due to Fourier invariance for even Hermite polynomials
    return c0, c4, c8

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 = A(g) * A(F(g)) where A(f) = ||f||_1 / ||f||_2^2 or similar.
    
    Actually, for the uncertainty inequality, we need:
    A(f) = ||f||_1 / ||f||_2 (or some norm ratio)
    
    Let me reconsider. The uncertainty bound is typically:
    C = (integral |f|)^2 / (integral |f|^2) * (integral |F(f)|)^2 / (integral |F(f)|^2)
    
    Or perhaps A(f) = ||f||_1 / ||f||_2 and the bound is A(f) * A(F(f)).
    
    By Plancherel, ||F(f)||_2 = ||f||_2 (with appropriate normalization).
    
    So we need to compute:
    ||g||_1, ||g||_2, ||F(g)||_1, ||F(g)||_2
    
    And the bound is (||g||_1 / ||g||_2) * (||F(g)||_1 / ||F(g)||_2)
    
    Since ||F(g)||_2 = ||g||_2 by Plancherel, this becomes:
    (||g||_1 * ||F(g)||_1) / ||g||_2^2
    """
    # Use adaptive quadrature on a reasonable range
    from scipy.integrate import quad
    
    # Evaluate on a grid for numerical integration
    # g(x) decays like exp(-x^2/2), so integrate over [-15, 15] should be sufficient
    
    x_grid = np.linspace(-15, 15, 10000)
    g_vals = _construct_g(c0, c4, c8, x_grid)
    dg_vals = _construct_g(c0, c4, c8, x_grid)  # F(g) has same form
    
    # Compute norms using trapezoidal rule
    dx = x_grid[1] - x_grid[0]
    norm1_g = np.sum(np.abs(g_vals)) * dx
    norm2_g_sq = np.sum(g_vals**2) * dx
    norm1_fg = np.sum(np.abs(dg_vals)) * dx  # Same as norm1_g due to invariance
    
    if norm2_g_sq <= 0:
        return 1e10  # Invalid
    
    bound = (norm1_g * norm1_fg) / norm2_g_sq
    return bound

def construct():
    """
    Find coefficients c0, c4, c8 that minimize the uncertainty bound.
    Returns np.array([c0, c4, c8]).
    """
    best_result = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_score = _compute_bound(best_result[0], best_result[1], best_result[2])
    
    def objective(params):
        c0, c4, c8 = params
        try:
            return _compute_bound(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Start with the known good point
    x0 = best_result.copy()
    current_best = x0.copy()
    current_best_score = best_score
    
    # Try local optimization from the starting point
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.success and result.fun < current_best_score:
            current_best = result.x.copy()
            current_best_score = result.fun
    except Exception:
        pass
    
    if _time_remaining() < 30:
        return current_best
    
    # Try basin hopping to escape local minima
    try:
        def bh_objective(params):
            return objective(params)
        
        bh_result = basinhopping(bh_objective, x0, niter=20, 
                                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200}},
                                stepsize=0.5, T=1.0, accept_test=lambda e1, e2: e1 < e2)
        if bh_result.fun < current_best_score:
            current_best = bh_result.x.copy()
            current_best_score = bh_result.fun
    except Exception:
        pass
    
    if _time_remaining() < 20:
        return current_best
    
    # Try differential evolution for global search
    try:
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        de_result = differential_evolution(objective, bounds, maxiter=50, 
                                           popsize=15, tol=1e-8, seed=42,
                                           polish=True)
        if de_result.fun < current_best_score:
            current_best = de_result.x.copy()
            current_best_score = de_result.fun
    except Exception:
        pass
    
    return current_best

# EVOLVE-BLOCK-END
