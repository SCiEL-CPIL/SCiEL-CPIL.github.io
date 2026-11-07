"""Model-and-optimize approach for Erdos minimum overlap.

Key insight: the max correlation value is driven by the peaks of the
correlation function. We can parameterize h as a smooth function of a
few basis coefficients, then directly optimize those parameters.
"""
N = 100
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import sys

DEADLINE_OFFSET = 145.0  # Safety margin for 150s limit


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _softmax_correlation_value(h, lam=50.0):
    """Smooth surrogate for max correlation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m / N * 2
    # log-sum-exp for numerical stability
    shifted = corr - m
    log_sum_exp = m + np.log(np.sum(np.exp(shifted)) / len(corr))
    return log_sum_exp / N * 2


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(20):
        if abs(diff) < 1e-14:
            break
        # Only adjust elements that can absorb the correction
        if diff > 0:
            # Need to decrease sum: reduce elements that are > 0
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum: increase elements that are < 1
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)  # diff is negative, so this adds
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _construct_from_basis(coeffs):
    """
    Build h from a basis expansion:
    h[i] = 0.5 + sum_k coeffs[k] * basis_k(i)
    
    Basis functions are chosen to capture the key features:
    - Constant (handled by offset)
    - Low-frequency sinusoids
    - Linear ramp
    - Quadratic
    """
    n = len(coeffs)
    i = np.arange(N, dtype=float)
    h = np.full(N, 0.5)
    
    # Up to 12 basis functions
    if n > 0:
        h += coeffs[0] * (i / N - 0.5)  # Linear ramp
    if n > 1:
        h += coeffs[1] * ((i / N - 0.5) ** 2 - 1/12)  # Quadratic
    if n > 2:
        h += coeffs[2] * np.sin(2 * np.pi * i / N)  # freq 1
    if n > 3:
        h += coeffs[3] * np.cos(2 * np.pi * i / N)
    if n > 4:
        h += coeffs[4] * np.sin(4 * np.pi * i / N)  # freq 2
    if n > 5:
        h += coeffs[5] * np.cos(4 * np.pi * i / N)
    if n > 6:
        h += coeffs[6] * np.sin(6 * np.pi * i / N)  # freq 3
    if n > 7:
        h += coeffs[7] * np.cos(6 * np.pi * i / N)
    if n > 8:
        h += coeffs[8] * np.sin(10 * np.pi * i / N)  # freq 5
    if n > 9:
        h += coeffs[9] * np.cos(10 * np.pi * i / N)
    if n > 10:
        h += coeffs[10] * np.sin(14 * np.pi * i / N)  # freq 7
    if n > 11:
        h += coeffs[11] * np.cos(14 * np.pi * i / N)
    
    return h


def _objective_from_coeffs(coeffs):
    """Objective: compute correlation value from basis coefficients."""
    h = _construct_from_basis(coeffs)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _local_polish(h, max_iters=30, step_sizes=None):
    """
    Local search: try small adjustments to individual elements.
    More aggressive than the original.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    if step_sizes is None:
        step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    for it in range(max_iters):
        improved = False
        # Try adjusting each element by each step size
        for i in range(n):
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0 or new_h[i] > 1:
                        continue
                    # Adjust another element to maintain sum
                    # Find the element that can best absorb the change
                    needed = -sign * d
                    candidates = [(j, best_h[j] + needed) for j in range(n) if j != i]
                    valid = [(j, val) for j, val in candidates if 0 <= val <= 1]
                    if not valid:
                        continue
                    # Pick the one that minimizes the objective change (heuristic: pick j farthest from i)
                    j = valid[0][0]  # Simple: just pick first valid
                    new_h[j] = best_h[j] + needed
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-13:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h

def _high_precision_polish(h, time_budget=20.0):
    """
    High-precision polish using SLSQP with tighter tolerances and more iterations.
    """
    start_time = time.time()
    current_h = h.copy()
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    # Try multiple lambda values for the soft surrogate, increasing precision
    for lam in [100.0, 200.0, 500.0, 1000.0]:
        if time.time() - start_time > time_budget - 5.0:
            break
        
        def smooth_obj(x, lam=lam):
            h_local = x
            corr = np.correlate(h_local, 1 - h_local, mode="full")
            m = np.max(corr)
            if m > 1e15:
                return m
            shifted = corr - m
            log_sum_exp = m + np.log(np.sum(np.exp(shifted)) / len(corr))
            return log_sum_exp
        
        try:
            res = minimize(
                smooth_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
            current_h = h_opt
        except Exception:
            pass
    
    # Final exact optimization with very tight tolerance
    if time.time() - start_time < time_budget - 2.0:
        try:
            def exact_obj(x):
                h_local = x
                corr = np.correlate(h_local, 1 - h_local, mode="full")
                return np.max(corr)
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 1000, 'ftol': 1e-18}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over basis coefficients
    # Use a moderate number of basis functions
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    # Run differential evolution with time budget
    de_deadline = start_time + 45.0
    
    try:
        # Custom wrapper that checks time
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(coeffs)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-7,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(_construct_from_basis(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Multi-start SLSQP from good starting points
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    starts = [best_h.copy()]
    
    # Random feasible starts
    for _ in range(6):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)
    
    # Structured: alternating 0s and 1s
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    # Structured: blocks
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
    
    # Sine waves at various frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        amp = min(0.4, 0.5 / freq)
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))
    
    # Sawtooth / triangular
    for period in [10, 20, 25]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))
    
    # Optimize from each start with SLSQP
    slsqp_deadline = start_time + 100.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 15.0:
            break
        
        h_start = _project_to_feasible(h_start)
        
        # Smooth objective with increasing lambda
        current_h = h_start
        for lam in [30.0, 150.0, 500.0]:
            if time.time() > slsqp_deadline - 10.0:
                break
            
            def smooth_obj(x, lam=lam):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                m = np.max(corr)
                if m > 1e15:
                    return m
                shifted = corr - m
                log_sum_exp = m + np.log(np.sum(np.exp(shifted)) / len(corr))
                return log_sum_exp
            
            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 300, 'ftol': 1e-15}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Final exact optimization
        if time.time() > slsqp_deadline - 5.0:
            break
        
        try:
            def exact_obj(x):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                return np.max(corr)
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    # Phase 3: Local search polish
    if time.time() < deadline - 25.0:
        try:
            best_h = _local_polish(best_h, max_iters=30)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 4: High-precision polish
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 10.0:
        try:
            best_h = _high_precision_polish(best_h, time_budget=min(remaining_time, 30.0))
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
