"""Piecewise-linear control-point parameterization for Erdos minimum overlap.

Key insight: represent h as a piecewise linear function defined by 10 control
points, linearly interpolated to 100 samples. This low-dimensional smooth
parameterization lets the optimizer explore global shapes (bimodal, multi-modal)
more effectively than optimizing 100 independent variables.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
N_CP = 20  # number of control points, increased for better resolution


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _interpolate_cp(cp):
    """
    Interpolate N_CP control points to N samples using linear interpolation.
    Control points are defined at equally spaced positions.
    """
    # Control point positions: 0, 1/(N_CP-1), 2/(N_CP-1), ..., 1
    cp_pos = np.linspace(0, 1, N_CP)
    # Sample positions: 0, 1/(N-1), 2/(N-1), ..., 1
    sample_pos = np.linspace(0, 1, N)
    # Linear interpolation
    h = np.interp(sample_pos, cp_pos, cp)
    return h


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-14:
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


def _construct_from_cp(cp):
    """Build h from control points and project to feasible set."""
    h = _interpolate_cp(cp)
    h = _project_to_feasible(h)
    return h


def _objective_from_cp(cp):
    """Objective: compute correlation value from control points."""
    h = _construct_from_cp(cp)
    return _correlation_value(h)


def _local_polish_cp(cp, max_iters=100, step_sizes=None):
    """
    Local search on control points. Much faster than full vector search.
    """
    best_cp = cp.copy()
    best_h = _construct_from_cp(best_cp)
    best_val = _correlation_value(best_h)
    n_cp = len(best_cp)
    
    if step_sizes is None:
        step_sizes = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    for it in range(max_iters):
        improved = False
        for i in range(n_cp):
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_cp = best_cp.copy()
                    new_cp[i] += sign * d
                    if new_cp[i] < 0 or new_cp[i] > 1:
                        continue
                    new_h = _construct_from_cp(new_cp)
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-14:
                        best_val = val
                        best_cp = new_cp
                        best_h = new_h
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_cp, best_h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over control points
    bounds = [(0.0, 1.0)] * N_CP
    
    def de_objective(cp):
        return _objective_from_cp(cp)
    
    de_deadline = start_time + 60.0
    
    try:
        def timed_de_objective(cp):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(cp)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
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
    
    # Phase 2: Multi-start SLSQP on control points
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points in control-point space
    starts = [best_h.copy()[:N_CP]]  # Approximate: use first N_CP values as initial cp
    
    # Actually, let's generate proper control point starts
    starts = []
    
    # From current best: find control points that best approximate
    # For simplicity, use evenly sampled values from best_h
    cp_from_best = np.interp(np.linspace(0, 1, N_CP), np.linspace(0, 1, N), best_h)
    starts.append(cp_from_best)
    
    # Random feasible starts
    for _ in range(10):
        cp_rand = rng.uniform(0, 1, N_CP)
        starts.append(cp_rand)
    
    # Structured: alternating high/low (adapted for N_CP)
    cp_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N_CP)])
    starts.append(cp_alt)
    
    # Structured: high frequency oscillation
    cp_osc = 0.5 + 0.5 * np.sin(np.pi * np.linspace(0, N_CP, N_CP))
    cp_osc = np.clip(cp_osc, 0, 1)
    starts.append(cp_osc)
    
    # Sine wave patterns
    for freq in [1, 2, 3, 4]:
        cp_sin = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.linspace(0, 1, N_CP))
        cp_sin = np.clip(cp_sin, 0, 1)
        starts.append(cp_sin)
    
    # Cosine wave patterns
    for freq in [1, 2, 3, 4]:
        cp_cos = 0.5 + 0.4 * np.cos(2 * np.pi * freq * np.linspace(0, 1, N_CP))
        cp_cos = np.clip(cp_cos, 0, 1)
        starts.append(cp_cos)
    
    # Linear ramps
    cp_ramp_up = np.linspace(0, 1, N_CP)
    cp_ramp_down = np.linspace(1, 0, N_CP)
    starts.append(cp_ramp_up)
    starts.append(cp_ramp_down)
    
    # Triangular wave
    t = np.linspace(0, 1, N_CP)
    cp_tri = 1 - np.abs(2 * t - 1)
    starts.append(cp_tri)
    
    # Double triangular (W shape)
    t = np.linspace(0, 1, N_CP)
    cp_w = np.abs(4 * t - 2)
    cp_w = np.clip(cp_w, 0, 1)
    starts.append(cp_w)
    
    slsqp_deadline = start_time + 105.0
    
    for idx, cp_start in enumerate(starts):
        if time.time() > slsqp_deadline - 8.0:
            break
        
        cp_start = np.clip(cp_start, 0, 1)
        
        # Optimize control points with SLSQP
        try:
            def cp_objective(cp):
                h = _construct_from_cp(cp)
                return _correlation_value(h)
            
            # Use the sum constraint via projection, so we optimize cp freely
            # and project after
            res = minimize(
                cp_objective,
                cp_start,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N_CP,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            cp_opt = np.clip(res.x, 0, 1)
            h_opt = _construct_from_cp(cp_opt)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Local search polish on the control points
    if time.time() < deadline - 5.0:
        try:
            # Convert current best_h back to control points for polishing
            cp_pos = np.linspace(0, 1, N_CP)
            sample_pos = np.linspace(0, 1, N)
            cp_init = np.interp(cp_pos, sample_pos, best_h)
            cp_polished, h_polished = _local_polish_cp(cp_init, max_iters=50)
            val_polished = _correlation_value(h_polished)
            if val_polished < best_val:
                best_val = val_polished
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
