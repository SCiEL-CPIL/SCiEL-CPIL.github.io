"""Piecewise-linear control-point parameterization for Erdos minimum overlap.

Improved version:
1. Increased control points to 50 for better resolution.
2. Uses a penalty-based objective function to handle constraints smoothly,
   allowing SLSQP to converge effectively without non-differentiable projection steps.
3. Combines Differential Evolution for global exploration with SLSQP for local refinement.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
N_CP = 50  # Increased resolution


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _interpolate_cp(cp):
    """
    Interpolate N_CP control points to N samples using linear interpolation.
    """
    cp_pos = np.linspace(0, 1, N_CP)
    sample_pos = np.linspace(0, 1, N)
    h = np.interp(sample_pos, cp_pos, cp)
    return h


def _penalized_objective(cp, lambda_sum=100.0, lambda_bound=100.0):
    """
    Objective function with penalties for constraint violations.
    This makes the function smooth and suitable for gradient-based methods.
    """
    h = _interpolate_cp(cp)
    
    # Base objective
    val = _correlation_value(h)
    
    # Penalty for sum constraint
    sum_h = np.sum(h)
    target_sum = N / 2.0
    sum_err = (sum_h - target_sum) ** 2
    
    # Penalty for bound constraints
    # h should be in [0, 1]
    below = np.clip(-h, 0, None)
    above = np.clip(h - 1, 0, None)
    bound_err = np.sum(below**2) + np.sum(above**2)
    
    return val + lambda_sum * sum_err + lambda_bound * bound_err


def _construct_from_cp(cp):
    """Build h from control points and project to feasible set for final output."""
    h = _interpolate_cp(cp)
    # Hard projection for final result to ensure validity
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    # Simple iterative projection
    for _ in range(100):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    bounds = [(0.0, 1.0)] * N_CP
    
    # Phase 1: Differential Evolution
    # DE doesn't handle penalties as well, but it's good for global search.
    # We use the penalized objective here too, but with lower penalties to guide it.
    def de_objective(cp):
        return _penalized_objective(cp, lambda_sum=10.0, lambda_bound=10.0)
    
    de_deadline = start_time + 40.0
    
    try:
        def timed_de_objective(cp):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(cp)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=False, # We will polish separately
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _construct_from_cp(result.x)
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: SLSQP Local Optimization with Penalty
    # We run SLSQP from multiple starting points:
    # 1. The best from DE
    # 2. Random starts
    # 3. Structured starts
    
    rng = np.random.default_rng(123)
    
    starts = []
    
    # From DE result
    if result.x is not None:
        starts.append(result.x)
    
    # From current best (interpolated back to CPs)
    cp_from_best = np.interp(np.linspace(0, 1, N_CP), np.linspace(0, 1, N), best_h)
    starts.append(cp_from_best)
    
    # Random starts
    for _ in range(20):
        starts.append(rng.uniform(0, 1, N_CP))
    
    # Structured starts
    # Alternating
    starts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N_CP)]))
    # Sine waves
    for freq in [1, 2, 3, 4, 5]:
        t = np.linspace(0, 1, N_CP)
        cp_sin = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t)
        starts.append(np.clip(cp_sin, 0, 1))
    # Cosine waves
    for freq in [1, 2, 3, 4, 5]:
        t = np.linspace(0, 1, N_CP)
        cp_cos = 0.5 + 0.5 * np.cos(2 * np.pi * freq * t)
        starts.append(np.clip(cp_cos, 0, 1))
    # Ramps
    starts.append(np.linspace(0, 1, N_CP))
    starts.append(np.linspace(1, 0, N_CP))
    
    slsqp_deadline = start_time + 105.0
    
    for idx, cp_start in enumerate(starts):
        if time.time() > slsqp_deadline - 5.0:
            break
        
        cp_start = np.clip(cp_start, 0, 1)
        
        try:
            # Use high penalty for SLSQP to ensure feasibility
            def slsqp_objective(cp):
                return _penalized_objective(cp, lambda_sum=1000.0, lambda_bound=1000.0)
            
            res = minimize(
                slsqp_objective,
                cp_start,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'disp': False}
            )
            
            cp_opt = np.clip(res.x, 0, 1)
            h_opt = _construct_from_cp(cp_opt)
            val = _correlation_value(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final projection and check
    best_h = _construct_from_cp(np.interp(np.linspace(0, 1, N_CP), np.linspace(0, 1, N), best_h))
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
        best_h = _construct_from_cp(best_h) # Ensure it's the projected one
        
    return best_h
