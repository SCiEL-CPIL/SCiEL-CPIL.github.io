"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial

# Deadline guard: 210s limit, use 195s for safety
_DEADLINE = time.time() + 195

def _hermite_at_zero(n):
    if n == 0:
        return 1.0
    if n == 4:
        return 12.0
    if n == 8:
        return 3024.0
    if n == 12:
        return 10321920.0
    return 0.0

def _eval_poly_and_g(x, a0, a4, a8):
    """Evaluate P(x)*exp(-x^2/2) and its components."""
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 3024.0, 10321920.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    H0 = np.ones_like(x)
    H4 = 16.0*x4 - 48.0*x2 + 12.0
    H8 = 256.0*x8 - 4608.0*x6 + 23040.0*x4 - 34560.0*x2 + 10080.0
    H12 = (4096.0*x12 - 147456.0*x10 + 2090112.0*x8 - 
           12902400.0*x6 + 34406400.0*x4 - 34406400.0*x2 + 10321920.0)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-x2 / 2.0)
    return g

def _l1_norm_grid(a0, a4, a8, N=65536, xmax=12.0):
    """Compute ||P(x) exp(-x^2/2)||_1 via uniform grid."""
    x = np.linspace(-xmax, xmax, N)
    dx = x[1] - x[0]
    g = _eval_poly_and_g(x, a0, a4, a8)
    return np.sum(np.abs(g)) * dx

def _l1_norm_smooth(a0, a4, a8, eps=0.01, N=32768, xmax=10.0):
    """Smooth surrogate for L1 norm using sqrt(x^2+eps) instead of |x|."""
    x = np.linspace(-xmax, xmax, N)
    dx = x[1] - x[0]
    g = _eval_poly_and_g(x, a0, a4, a8)
    # Smooth absolute value
    smooth_abs = np.sqrt(g**2 + eps**2)
    return np.sum(smooth_abs) * dx

def _objective_exact(a0, a4, a8):
    """Exact objective: ||g||_1 * ||F[g]||_1.
    Since F[H_n e^{-x^2/2}] = (-i)^n H_n e^{-x^2/2} and n is even,
    |F[g]| has the same functional form as |g|, so ||g||_1 = ||F[g]||_1 (up to const).
    The bound is proportional to ||g||_1^2.
    """
    l1 = _l1_norm_grid(a0, a4, a8)
    if not np.isfinite(l1) or l1 <= 0:
        return 1e10
    return l1 * l1

def _objective_smooth(a0, a4, a8, eps=0.005):
    """Smooth surrogate objective."""
    l1s = _l1_norm_smooth(a0, a4, a8, eps=eps)
    if not np.isfinite(l1s) or l1s <= 0:
        return 1e10
    return l1s * l1s

def _vqa_params_to_coeffs(theta):
    """
    VQA-inspired parameterization: use rotation angles to generate coefficients.
    theta = [t1, t2, t3] maps to (a0, a4, a8) via a sequence of rotations
    that keep the coefficients in a natural bounded region.
    
    This mimics a quantum ansatz: start with a "ground state" and apply
    rotation gates to explore the coefficient space.
    """
    t1, t2, t3 = theta
    
    # Base scale factors
    s0 = 1.0
    s4 = 0.04
    s8 = 0.0005
    
    # Rotation-based parameterization
    # a0 = s0 * cos(t1)
    # a4 = s4 * sin(t1) * cos(t2)
    # a8 = s8 * sin(t1) * sin(t2) * cos(t3)
    # with an additional mixing term
    
    a0 = s0 * np.cos(t1) + 0.3 * s0 * np.sin(t1) * np.cos(t2)
    a4 = s4 * np.sin(t1) * np.cos(t2) + 0.1 * s4 * np.cos(t1) * np.sin(t3)
    a8 = s8 * np.sin(t1) * np.sin(t2) + 0.05 * s8 * np.cos(t1) * np.cos(t3)
    
    return np.array([a0, a4, a8])

def construct():
    # Seed result
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    seed_bound = _objective_exact(*seed_result)
    
    best_result = seed_result.copy()
    best_bound = seed_bound
    
    # Phase 1: Differential evolution with smooth surrogate
    bounds = [(-3.0, 3.0), (-0.2, 0.2), (-0.002, 0.002)]
    
    if time.time() < _DEADLINE - 100:
        try:
            de_result = differential_evolution(
                _objective_smooth,
                bounds,
                seed=42,
                maxiter=80,
                tol=1e-6,
                polish=False,
                init='sobol',
                popsize=15,
                updating='deferred'
            )
            if de_result.fun < best_bound:
                # Verify with exact objective
                exact_fun = _objective_exact(*de_result.x)
                if exact_fun < best_bound:
                    best_bound = exact_fun
                    best_result = de_result.x.copy()
                else:
                    # Use exact for comparison
                    best_result = de_result.x.copy()
                    best_bound = exact_fun
        except Exception:
            pass
    
    # Phase 2: VQA-inspired search using rotation parameters
    if time.time() < _DEADLINE - 60:
        try:
            def _vqa_objective(theta):
                if time.time() > _DEADLINE - 20:
                    return 1e10
                coeffs = _vqa_params_to_coeffs(theta)
                return _objective_smooth(*coeffs)
            
            vqa_bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
            
            vqa_result = differential_evolution(
                _vqa_objective,
                vqa_bounds,
                seed=123,
                maxiter=50,
                tol=1e-5,
                polish=False,
                init='sobol',
                popsize=12
            )
            
            if vqa_result.fun < 1e9:
                vqa_coeffs = _vqa_params_to_coeffs(vqa_result.x)
                exact_fun = _objective_exact(*vqa_coeffs)
                if exact_fun < best_bound:
                    best_bound = exact_fun
                    best_result = vqa_coeffs.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement with Nelder-Mead on smooth surrogate
    starts = [
        best_result,
        seed_result,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
        np.array([-1.5, 0.06, 0.0005]),
        np.array([-0.8, 0.02, 0.0001]),
        np.array([-1.1, 0.045, 0.00035]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 35:
            break
        try:
            res = minimize(
                _objective_smooth,
                start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < 1e9:
                exact_fun = _objective_exact(*res.x)
                if exact_fun < best_bound:
                    best_bound = exact_fun
                    best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Fine-tune with L-BFGS-B on smooth surrogate
    if time.time() < _DEADLINE - 20:
        try:
            res = minimize(
                _objective_smooth,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < 1e9:
                exact_fun = _objective_exact(*res.x)
                if exact_fun < best_bound:
                    best_bound = exact_fun
                    best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final polish with tighter smooth epsilon
    if time.time() < _DEADLINE - 10:
        try:
            def _objective_tight(a0, a4, a8):
                return _l1_norm_smooth(a0, a4, a8, eps=0.001) ** 2
            
            res = minimize(
                _objective_tight,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < 1e9:
                exact_fun = _objective_exact(*res.x)
                if exact_fun < best_bound:
                    best_bound = exact_fun
                    best_result = res.x.copy()
        except Exception:
            pass
    
    # Final: one more exact evaluation to confirm
    if time.time() < _DEADLINE - 5:
        try:
            final_bound = _objective_exact(*best_result)
            if final_bound < best_bound:
                best_bound = final_bound
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
