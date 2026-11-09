"""
Uncertainty inequality: tune Hermite-Gaussian test function coefficients to minimize the upper bound on C4.

We parameterize g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is derived from the constraint g(0) = 0.

The score is the upper bound on A(f)*A(F(f)) (or C4-related constant).
We use scipy.optimize to minimize this bound over (a0, a4, a8).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution

DEADLINE_FRACTION = 0.90  # safety margin before 210s

# Precompute on a fixed grid for fast evaluation
_X_GRID = np.linspace(-15, 15, 50000)
_DX = _X_GRID[1] - _X_GRID[0]
_H0 = np.ones_like(_X_GRID)
_H4 = eval_hermite(4, _X_GRID)
_H8 = eval_hermite(8, _X_GRID)
_H12 = eval_hermite(12, _X_GRID)
_GAUSS = np.exp(-0.5 * _X_GRID**2)


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on the uncertainty product for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    """
    # Derive a12 from g(0) = 0
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Build g(x) using precomputed arrays
    poly = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    g = poly * _GAUSS
    
    # ||g||_2^2
    g2 = g**2
    norm2_sq = np.trapz(g2, dx=_DX)
    
    if norm2_sq <= 0:
        return 1e10  # invalid
    
    norm2 = np.sqrt(norm2_sq)
    norm_inf = np.max(np.abs(g))
    
    bound = (norm_inf / norm2)**2
    
    return bound


def construct():
    start_time = time.time()
    deadline = 210.0 * DEADLINE_FRACTION  # 189s
    
    best_val = float('inf')
    best_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])  # seed
    
    # Evaluate seed
    try:
        seed_val = _compute_bound(best_x[0], best_x[1], best_x[2])
        if seed_val < best_val:
            best_val = seed_val
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x
    
    # Fast coarse search: sample a smaller grid around promising regions
    try:
        # Start with a moderate grid centered on the seed
        a0_center, a4_center, a8_center = best_x
        a0_span, a4_span, a8_span = 2.0, 0.2, 0.005
        
        n_a0, n_a4, n_a8 = 10, 8, 6
        
        a0_vals = np.linspace(a0_center - a0_span, a0_center + a0_span, n_a0)
        a4_vals = np.linspace(a4_center - a4_span, a4_center + a4_span, n_a4)
        a8_vals = np.linspace(a0_center - a8_span, a0_center + a8_span, n_a8)
        
        for a0 in a0_vals:
            for a4 in a4_vals:
                for a8 in a8_vals:
                    if time.time() - start_time > deadline - 25:
                        raise TimeoutError
                    try:
                        v = _compute_bound(a0, a4, a8)
                        if v < best_val:
                            best_val = v
                            best_x = np.array([a0, a4, a8])
                    except Exception:
                        pass
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_x
    
    # Differential Evolution for global search
    try:
        bounds = [
            (best_x[0] - 3.0, best_x[0] + 3.0),
            (best_x[1] - 0.3, best_x[1] + 0.3),
            (best_x[2] - 0.01, best_x[2] + 0.01)
        ]
        
        def _de_obj(x):
            return _compute_bound(x[0], x[1], x[2])
        
        result = differential_evolution(
            _de_obj,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            popsize=15,
            polish=False
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        return best_x
    
    # Refine with Nelder-Mead
    try:
        result = minimize(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_x
    
    # Multi-start refinement from perturbed points
    np.random.seed(42)
    try:
        for _ in range(5):
            if time.time() - start_time > deadline - 8:
                break
            scale = 0.05
            x0 = best_x + scale * np.random.randn(3)
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Final high-precision polish
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
