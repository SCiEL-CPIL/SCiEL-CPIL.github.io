"""Optimized tuner for the Fourier-invariant Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import roots_hermite

_T0 = time.time()
_HARD_DEADLINE = 195.0  # safety margin before the 210 s kill
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x) as array [c0, c1, ..., cn]."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    # Recurrence: H_n = 2x*H_{n-1} - 2(n-1)*H_{n-2}
    h = np.zeros(n + 1)
    h[0] = 1.0
    for k in range(1, n + 1):
        new_h = np.zeros(n + 1)
        new_h[1:] += 2.0 * h[:-1]  # 2x * H_{k-1}
        if k >= 2:
            new_h[:-2] -= 2.0 * (k - 1) * h[2:]  # -2(k-1)*H_{k-2}
        h = new_h
    return h


def _poly_eval(coeffs, x):
    """Evaluate polynomial with coefficients [c0, c1, ..., cn] at points x."""
    result = np.zeros_like(x, dtype=float)
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result


def _build_quadrature():
    """Build Gauss-Hermite quadrature for exact integration of polynomials * exp(-x^2) and exp(-2x^2)."""
    # For I2 = ∫ P(x)^2 * exp(-x^2) dx: use standard GH with nodes x_i, weights w_i
    # For I4 = ∫ P(x)^4 * exp(-2x^2) dx: rescale u = sqrt(2)*x
    n_gauss = 128  # enough for degree-48 polynomials
    x_gauss, w_gauss = roots_hermite(n_gauss)
    
    # Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
    hc = {n: _hermite_coeffs(n) for n in (0, 4, 8, 12)}
    
    # Evaluate each H_n at Gauss nodes
    H_vals = {n: _poly_eval(hc[n], x_gauss) for n in (0, 4, 8, 12)}
    
    # For I4: ∫ P(x)^4 exp(-2x^2) dx = (1/sqrt(2)) ∫ P(u/sqrt(2))^4 exp(-u^2) du
    x_gauss2 = x_gauss / np.sqrt(2.0)
    H_vals2 = {n: _poly_eval(hc[n], x_gauss2) for n in (0, 4, 8, 12)}
    w_gauss2 = w_gauss / np.sqrt(2.0)
    
    return x_gauss, w_gauss, H_vals, x_gauss2, w_gauss2, H_vals2


def _make_objective(w1, H1, w2, H2):
    """Create objective using exact Gauss-Hermite quadrature."""
    def objective(n):
        a0, a4, a8 = n[0], n[1], n[2]
        a12 = -(a0 + a4 / 6.0 + a8 / 108.0)
        
        # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        P1 = a0 * H1[0] + a4 * H1[1] + a8 * H1[2] + a12 * H1[3]
        P2 = a0 * H2[0] + a4 * H2[1] + a8 * H2[2] + a12 * H2[3]
        
        # I2 = ∫ P(x)^2 exp(-x^2) dx
        I2 = np.sum(w1 * P1 * P1)
        
        # I4 = ∫ P(x)^4 exp(-2x^2) dx
        I4 = np.sum(w2 * P2 * P2 * P2 * P2)
        
        if I2 < 1e-30:
            return 1e10
        if I4 < 0:
            I4 = 0.0
        
        return np.sqrt(2.0 * np.pi) * np.sqrt(I4 / I2)
    
    return objective


def construct():
    best_n = _SEED.copy()
    
    # Build quadrature
    x1, w1, H1, x2, w2, H2 = _build_quadrature()
    
    objective = _make_objective(w1, H1, w2, H2)
    
    best_val = objective(best_n)
    
    # Phase 1: Differential evolution with reasonable bounds
    if time.time() < _HARD_DEADLINE - 60.0:
        bounds = [
            (-5.0, 5.0),
            (-1.0, 1.0),
            (-0.1, 0.1)
        ]
        try:
            de_res = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=30,
                tol=1e-10,
                seed=42,
                polish=False,
                init='sobol',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_n = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement
    starts = [best_n.copy(), _SEED.copy()]
    rng = np.random.default_rng(12345)
    for _ in range(20):
        starts.append(best_n + rng.normal(0, 0.1, 3))
    for _ in range(10):
        starts.append(_SEED + rng.normal(0, 0.2, 3))
    
    for s in starts:
        if time.time() > _HARD_DEADLINE - 20.0:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14,
                                    'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < _HARD_DEADLINE - 5.0:
        try:
            res = minimize(objective, best_n, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-15,
                                    'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            pass
    
    return best_n
# EVOLVE-BLOCK-END
