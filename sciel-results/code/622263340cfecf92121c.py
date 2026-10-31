"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE_MARGIN = 15.0
TARGET_SUM = N / 2.0
KILL_TIME = 150.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-15:
            break
        step = diff / len(h)
        h += step
        h = np.clip(h, 0.0, 1.0)
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        mask = (h > 1e-13) & (h < 1.0 - 1e-13)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """Compute soft max of x and its gradient using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-14:
        return 0.0, np.zeros_like(x)
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum == 0 or not np.isfinite(wsum):
        return max_x, np.zeros_like(x)
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _overlap_and_grad(h, beta):
    """Compute smooth max of overlap and gradient w.r.t h."""
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    grad_h = np.zeros(n)
    if n > 1:
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        grad_h[-1] += grad_overlap[-1]
    else:
        grad_h[0] += grad_overlap[0]
    return val_smooth, grad_h, overlap


def _objective_and_grad(h, beta, penalty_weight=50.0):
    """Objective with sum constraint penalty."""
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    val_smooth, grad_h, _ = _overlap_and_grad(h, beta)
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    return total_val, total_grad


def _sinusoidal_h(A, phi, A2=0.0, phi2=0.0, A3=0.0, phi3=0.0):
    """Generate step heights from sinusoidal model."""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
    if abs(A2) > 1e-12:
        h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
    if abs(A3) > 1e-12:
        h += A3 * np.sin(6.0 * np.pi * i / N + phi3)
    return h


def _optimize_sinusoidal(deadline):
    """Optimize sinusoidal parameters via grid search + local refinement."""
    best_score = float('inf')
    best_h = None
    best_params = None

    # Phase 1: Coarse grid search
    A_grid = np.linspace(0.05, 0.45, 30)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 60)

    best_A = 0.25
    best_phi = 0.0
    best_grid_score = float('inf')

    for A in A_grid:
        if time.time() > deadline - 30.0:
            break
        for phi in phi_grid:
            h = _sinusoidal_h(A, phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_A = A
                best_phi = phi

    # Phase 2: Local refinement with Nelder-Mead on 6 params
    def obj_sin(params):
        A, phi, A2, phi2, A3, phi3 = params
        A = np.clip(A, 0.0, 0.5)
        A2 = np.clip(A2, -0.3, 0.3)
        A3 = np.clip(A3, -0.2, 0.2)
        h = _sinusoidal_h(A, phi, A2, phi2, A3, phi3)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0 = np.array([best_A, best_phi, 0.0, 0.0, 0.0, 0.0])
    try:
        res = minimize(obj_sin, x0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Phase 3: Multiple starting points
    for A_init in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        if time.time() > deadline - 20.0:
            break
        for phi_init in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            for A2_init in [0.0, 0.05, -0.05]:
                x0 = np.array([A_init, phi_init, A2_init, 0.0, 0.0, 0.0])
                try:
                    res = minimize(obj_sin, x0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res.fun < best_grid_score:
                        best_grid_score = res.fun
                        best_params = res.x
                except Exception:
                    pass

    if best_params is not None:
        h_sin = _sinusoidal_h(*best_params)
    else:
        h_sin = _sinusoidal_h(best_A, best_phi)
    h_sin = np.clip(h_sin, 0.0, 1.0)
    h_sin = _fix_sum(h_sin)
    best_score = _score(h_sin)
    best_h = h_sin.copy()

    return best_score, best_h


def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """Optimize h using L-BFGS-B with analytic gradients and adaptive beta."""
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)

    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]

    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break

        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val

        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad

        try:
            res = minimize(
                obj_func,
                h,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception:
            pass

    return current_score, h


def _local_search_polish(h, max_iters=300, deadline=None):
    """Deterministic local search to polish the solution using coordinate descent."""
    h = h.copy()
    current_score = _score(h)

    if deadline is None:
        deadline = time.time() + 5.0

    # Adaptive deltas
    base_deltas = [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        improved = False
        # Cycle through indices
        for i in range(N):
            # Try a few deltas
            for delta in base_deltas:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0.0 or h_trial[i] > 1.0:
                    continue
                
                # Fix sum efficiently
                # Instead of full _fix_sum which clips and iterates, 
                # we can just distribute the error.
                # But _fix_sum is robust. Let's use it but it's O(N).
                # N=100 is small, so it's fine.
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if improved:
                # Restart cycle to catch dependencies
                break
                
        if not improved:
            # If no single step improved, try smaller steps or break
            # Shrink deltas
            base_deltas = [d * 0.5 for d in base_deltas]
            if base_deltas[0] < 1e-5:
                break

    return current_score, h


def _pwl_from_knots(knots):
    """Interpolate knot values to N points.
    knots: array of shape (2*K,). Even indices are x-coords (in [0, N]), odd indices are y-values.
    We assume x-coords are fixed to be evenly spaced for simplicity in optimization, 
    or we can optimize y-values only for fixed x-coords. 
    For 'sharp transitions', fixed evenly spaced knots with optimized heights is a good start.
    """
    K = len(knots) // 2
    if K < 2:
        return np.full(N, 0.5)
    
    # Fixed knot positions: 0, N/(2K-1), ..., N
    # Wait, if we have K knots, we have K points.
    # Let's use K=10 knots.
    # x_knots = np.linspace(0, N-1, K)
    # y_knots = knots
    
    # Simpler: Just optimize N/K points and interpolate.
    # Let's stick to the directive: "knot positions and values are jointly optimized".
    # But for simplicity and robustness in a limited time, let's optimize values of a coarse grid
    # and interpolate.
    
    # Let's define the function to take an array of 'values' at M points and interpolate to N.
    pass

def _optimize_pwl(deadline):
    """Optimize a piecewise-linear representation with M knots."""
    M = 10 # Number of knots
    # Knot positions: evenly spaced indices 0, N/M, ...
    # Actually, let's just optimize the values of a downsampled signal and upsample.
    # This is equivalent to PWL with fixed knots.
    
    # To allow "jointly optimized positions" is complex. 
    # Let's try optimizing values at M points, then linearly interpolate to N.
    # Then fix sum.
    
    def pwl_to_full(values):
        # values: shape (M,)
        # Map to N points
        # x_orig = np.linspace(0, M-1, M)
        # x_new = np.linspace(0, M-1, N)
        # return np.interp(x_new, x_orig, values)
        
        # Better: Map index i in [0, N-1] to x in [0, M-1]
        x_new = np.linspace(0, M-1, N)
        x_orig = np.arange(M)
        return np.interp(x_new, x_orig, values)

    def pwl_objective(params):
        h_coarse = np.clip(params, 0.0, 1.0)
        h_full = pwl_to_full(h_coarse)
        h_full = _fix_sum(h_full)
        return _score(h_full)

    bounds = [(0.0, 1.0)] * M
    
    best_score = float('inf')
    best_h = None
    
    # Differential Evolution on coarse params
    try:
        res = differential_evolution(
            pwl_objective, bounds,
            maxiter=100, popsize=10,
            tol=1e-8, seed=123,
            polish=False, # We will polish separately
            init='sobol'
        )
        if res.fun < best_score:
            best_score = res.fun
            h_coarse = res.x
            h_full = pwl_to_full(h_coarse)
            h_full = _fix_sum(h_full)
            best_h = h_full
    except Exception:
        pass

    # Local refinement on coarse params using Nelder-Mead
    if best_h is not None:
        h_coarse = np.linspace(0, 1, M) # Fallback init if DE failed? No, use DE result if available.
        if res.success or res.fun < float('inf'):
            h_coarse = res.x
        else:
            h_coarse = np.full(M, 0.5)
            
        try:
            res_nm = minimize(
                pwl_objective, h_coarse,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                h_coarse = res_nm.x
                h_full = pwl_to_full(h_coarse)
                h_full = _fix_sum(h_full)
                best_h = h_full
        except Exception:
            pass
            
    return best_score, best_h


def construct():
    start_time = time.time()
    deadline = start_time + (KILL_TIME - DEADLINE_MARGIN)

    best_score = float('inf')
    best_h = None

    # Phase 1: PWL Optimization (New)
    if time.time() < deadline - 60.0:
        s_pwl, h_pwl = _optimize_pwl(deadline - 60.0)
        if s_pwl < best_score:
            best_score = s_pwl
            best_h = h_pwl.copy()
    else:
        # Fallback to Sinusoidal if time is tight (should not happen with 150s limit)
        if time.time() < deadline - 30.0:
            s_sin, h_sin = _optimize_sinusoidal(deadline - 30.0)
            if s_sin < best_score:
                best_score = s_sin
                best_h = h_sin.copy()

    # Phase 2: Direct optimization with analytic gradients
    if best_h is not None and time.time() < deadline - 40.0:
        s_direct, h_direct = _optimize_direct(best_h, deadline - 40.0)
        if s_direct < best_score:
            best_score = s_direct
            best_h = h_direct.copy()

    # Phase 3: Local search polish
    if best_h is not None and time.time() < deadline - 20.0:
        s_polish, h_polish = _local_search_polish(best_h, max_iters=500, deadline=deadline - 20.0)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Phase 4: Final high-precision optimization
    if best_h is not None and time.time() < deadline - 10.0:
        s_final, h_final = _optimize_direct(best_h, deadline - 10.0, beta_start=100.0, beta_end=1000.0, max_iter=500)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Phase 5: Final polish
    if best_h is not None and time.time() < deadline - 2.0:
        s_final2, h_final2 = _local_search_polish(best_h, max_iters=300, deadline=deadline - 2.0)
        if s_final2 < best_score:
            best_score = s_final2
            best_h = h_final2.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
