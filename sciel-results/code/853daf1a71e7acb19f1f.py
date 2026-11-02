import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

# Precompute grid for fast evaluation
_x_grid = np.linspace(-10, 10, 8001)
_dx = _x_grid[1] - _x_grid[0]
_gauss = np.exp(-_x_grid**2 / 2)

# Probabilists' Hermite polynomials precomputed on grid
def _he0(x): return np.ones_like(x)
def _he4(x): return x**4 - 6*x**2 + 3
def _he8(x): return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105
def _he12(x): return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

_H0 = _he0(_x_grid)
_H4 = _he4(_x_grid)
_H8 = _he8(_x_grid)
_H12 = _he12(_x_grid)

# Values at 0 for the constraint g(0)=0
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 10395.0

def _c3_from(c0, c1, c2):
    return -(c0 + _H4_0*c1 + _H8_0*c2) / _H12_0

def _g_values(coeffs):
    """Compute g(x) on the precomputed grid."""
    c0, c1, c2 = coeffs
    c3 = _c3_from(c0, c1, c2)
    return (c0*_H0 + c1*_H4 + c2*_H8 + c3*_H12) * _gauss

def _exact_bound(coeffs):
    """Exact bound: sqrt(2*pi) * L1 / L_inf for Fourier-invariant g."""
    g = _g_values(coeffs)
    l1 = np.sum(np.abs(g)) * _dx
    linf = np.max(np.abs(g))
    if linf < 1e-15:
        return 1e10
    return math.sqrt(2 * math.pi) * l1 / linf

def _smooth_bound(coeffs, temperature=0.01):
    """Smooth surrogate using soft-min via log-sum-exp."""
    g = _g_values(coeffs)
    l1 = np.sum(np.abs(g)) * _dx
    
    abs_g = np.abs(g)
    m = np.max(abs_g)
    if m < 1e-15:
        return 1e10
    scaled = abs_g / m
    # Use temperature for soft-min
    exp_terms = np.exp(-scaled / temperature)
    log_sum_exp = np.log(np.sum(exp_terms))
    soft_min = -temperature * (log_sum_exp - np.log(len(scaled))) * m
    
    if soft_min < 1e-15:
        return 1e10
    return math.sqrt(2 * math.pi) * l1 / soft_min

def _objective_de(coeffs):
    """Objective for differential evolution - uses exact bound."""
    return _exact_bound(coeffs)

def construct():
    t0 = time.time()
    deadline = t0 + 170  # safety margin before 210s kill
    
    # Seed coefficients
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_val = _exact_bound(best)
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 80:
        try:
            bounds = [(-3.0, 0.5), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective_de,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-10,
                atol=1e-12,
                init='sobol',
                polish=True,
                workers=1,
                updating='deferred',
                mutation=(0.3, 0.9),
                recombination=0.7
            )
            if de_result.fun < best_val:
                best = de_result.x.copy()
                best_val = de_result.fun
        except Exception:
            pass
    
    # Phase 2: Local refinement from best found using smooth surrogate
    if time.time() < deadline - 30:
        try:
            res = minimize(_smooth_bound, best, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            if time.time() < deadline - 20 and res.fun < best_val:
                exact_v = _exact_bound(res.x)
                if exact_v < best_val:
                    best = res.x.copy()
                    best_val = exact_v
        except Exception:
            pass
    
    # Phase 3: Multi-start local refinement
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        n_starts = 15
        for i in range(n_starts):
            if time.time() > deadline - 15:
                break
            # Generate diverse starting points
            if i % 3 == 0:
                start = best + rng.normal(0, 0.02, size=3)
            elif i % 3 == 1:
                start = np.array([rng.uniform(-2.0, -0.5), 
                                  rng.uniform(-0.3, 0.3), 
                                  rng.uniform(-0.005, 0.005)])
            else:
                start = np.array([rng.uniform(-1.5, 0.0), 
                                  rng.uniform(-0.2, 0.2), 
                                  rng.uniform(-0.003, 0.003)])
            try:
                res = minimize(_smooth_bound, start, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
                if time.time() > deadline - 15:
                    break
                if res.fun < best_val:
                    exact_v = _exact_bound(res.x)
                    if exact_v < best_val:
                        best = res.x.copy()
                        best_val = exact_v
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish using exact objective
    if time.time() < deadline - 10:
        try:
            res = minimize(_exact_bound, best, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 5: Ultra-fine polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_exact_bound, best, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return np.array(best)
