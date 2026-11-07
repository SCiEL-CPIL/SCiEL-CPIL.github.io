"""Erdos minimum overlap - piecewise-linear parameterization approach.

Key idea: represent h as a smooth piecewise-linear function with 10 control
points, linearly interpolated to 100 samples. This low-dimensional smooth
parameterization lets the optimizer explore global shapes (bimodal, etc.)
far more effectively than 100 independent variables.
"""
N = 100
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Control points for piecewise-linear representation
N_CTRL = 10
# Positions of control points (evenly spaced, including endpoints)
_CTRL_POS = np.linspace(0, N - 1, N_CTRL)


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(30):
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


def _interp_ctrl_to_h(ctrl):
    """Interpolate control points to N samples using linear interpolation."""
    x = np.arange(N, dtype=float)
    h = np.interp(x, _CTRL_POS, ctrl)
    return h


def _construct_from_ctrl(ctrl):
    """Build h from control points, project to feasible set."""
    h = _interp_ctrl_to_h(ctrl)
    return _project_to_feasible(h)


def _objective_from_ctrl(ctrl):
    """Objective: compute correlation value from control points."""
    h = _construct_from_ctrl(ctrl)
    return _correlation_value(h)


def _smooth_max_corr(h, lam=200.0):
    """Soft-max of correlation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    return m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam


def _slsqp_on_ctrl(ctrl, maxiter=200):
    """Optimize control points with SLSQP using smooth surrogate."""
    def obj(c):
        h = _interp_ctrl_to_h(c)
        return _smooth_max_corr(h, lam=300.0)
    
    def sum_constraint(c):
        h = _interp_ctrl_to_h(c)
        return np.sum(h) - N / 2.0
    
    bounds = [(0.0, 1.0)] * N_CTRL
    
    try:
        res = minimize(
            obj,
            ctrl,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'eq', 'fun': sum_constraint}],
            options={'maxiter': maxiter, 'ftol': 1e-14}
        )
        h_opt = _project_to_feasible(_interp_ctrl_to_h(res.x))
        return h_opt, _correlation_value(h_opt)
    except Exception:
        h = _project_to_feasible(_interp_ctrl_to_h(ctrl))
        return h, _correlation_value(h)


def _epigraph_polish(h, max_iters=200, tol=1e-15):
    """
    Epigraph formulation on full h vector:
    minimize t subject to corr[k] <= t, sum(h) = N/2, 0 <= h <= 1.
    """
    init_t = _correlation_value(h)
    x0 = np.concatenate([h, [init_t]])
    
    def objective(x):
        return x[-1]
    
    def smooth_max_constraint(x):
        h_x = x[:N]
        t = x[-1]
        corr = np.correlate(h_x, 1 - h_x, mode="full")
        lam = 1000.0
        m = np.max(corr)
        if m > 1e15:
            return m - t
        soft_max = m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam
        return soft_max - t
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N / 2.0},
        {'type': 'ineq', 'fun': lambda x: -(smooth_max_constraint(x))}
    ]
    
    bounds = [(0.0, 1.0)] * N + [(-1.0, 10.0)]
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': tol}
        )
        h_opt = _project_to_feasible(res.x[:N])
        return h_opt, _correlation_value(h_opt)
    except Exception:
        return h, _correlation_value(h)


def _local_polish(h, max_iters=40, step_sizes=None):
    """
    Local search: pair-swap strategy to maintain sum constraint.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for d in step_sizes:
                for sign in [+1, -1]:
                    delta = sign * d
                    if best_h[i] + delta < -1e-12 or best_h[i] + delta > 1 + 1e-12:
                        continue
                    needed = -delta
                    best_j = -1
                    best_val_j = best_val
                    for j in range(n):
                        if j == i:
                            continue
                        if best_h[j] + needed < -1e-12 or best_h[j] + needed > 1 + 1e-12:
                            continue
                        new_h = best_h.copy()
                        new_h[i] += delta
                        new_h[j] += needed
                        val = _correlation_value(new_h)
                        if val < best_val_j - 1e-14:
                            best_val_j = val
                            best_j = j
                    if best_j >= 0:
                        best_h[i] += delta
                        best_h[best_j] += needed
                        best_val = best_val_j
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h


def _generate_ctrl_starts():
    """Generate diverse starting points in control-point space."""
    starts = []
    rng = np.random.default_rng(123)
    
    # Uniform
    starts.append(np.full(N_CTRL, 0.5))
    
    # Random
    for _ in range(10):
        starts.append(rng.uniform(0.1, 0.9, N_CTRL))
    
    # Bimodal: high at edges, low in middle
    starts.append(np.array([0.9, 0.85, 0.7, 0.4, 0.2, 0.2, 0.4, 0.7, 0.85, 0.9]))
    starts.append(np.array([0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.7, 0.8]))
    
    # Bimodal: low at edges, high in middle
    starts.append(np.array([0.2, 0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.5, 0.3, 0.2]))
    
    # Alternating high-low
    starts.append(np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2]))
    starts.append(np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]))
    
    # Linear ramps
    starts.append(np.linspace(0.1, 0.9, N_CTRL))
    starts.append(np.linspace(0.9, 0.1, N_CTRL))
    
    # Sinusoidal shapes
    for freq in [1, 2, 3]:
        amp = min(0.4, 0.5 / freq)
        ctrl = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N_CTRL) / N_CTRL)
        starts.append(np.clip(ctrl, 0.05, 0.95))
    
    # Step-like
    for split in range(1, N_CTRL):
        ctrl = np.zeros(N_CTRL)
        ctrl[:split] = 0.8
        ctrl[split:] = 0.2
        starts.append(ctrl)
        ctrl2 = np.zeros(N_CTRL)
        ctrl2[:split] = 0.2
        ctrl2[split:] = 0.8
        starts.append(ctrl2)
    
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over control points
    bounds = [(0.0, 1.0)] * N_CTRL
    de_deadline = start_time + 40.0
    
    try:
        def timed_de_obj(ctrl):
            if time.time() > de_deadline:
                raise TimeoutError()
            return _objective_from_ctrl(ctrl)
        
        result = differential_evolution(
            timed_de_obj,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.4, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(_interp_ctrl_to_h(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: SLSQP on control points from multiple starts
    slsqp_deadline = start_time + 100.0
    ctrl_starts = _generate_ctrl_starts()
    
    # Also add the best from DE as a start
    if best_val < 0.5:  # improved from seed
        ctrl_starts.insert(0, np.full(N_CTRL, 0.5))
    
    for idx, ctrl_start in enumerate(ctrl_starts):
        if time.time() > slsqp_deadline - 10.0:
            break
        
        ctrl_start = np.clip(ctrl_start, 0.0, 1.0)
        
        for lam in [100.0, 300.0]:
            if time.time() > slsqp_deadline - 5.0:
                break
            try:
                h_opt, val = _slsqp_on_ctrl(ctrl_start, maxiter=200)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Epigraph polish on full h
    if time.time() < deadline - 20.0:
        try:
            h_ep, val_ep = _epigraph_polish(best_h, max_iters=200, tol=1e-15)
            if val_ep < best_val:
                best_val = val_ep
                best_h = h_ep.copy()
        except Exception:
            pass
    
    # Phase 4: Local search polish
    if time.time() < deadline - 10.0:
        try:
            best_h = _local_polish(best_h, max_iters=40)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 5: Final epigraph refinement
    if time.time() < deadline - 5.0:
        try:
            h_final, val_final = _epigraph_polish(best_h, max_iters=300, tol=1e-16)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
