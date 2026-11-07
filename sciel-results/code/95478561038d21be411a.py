"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _correlation_value(h):
    """Compute the evaluator's value: max(np.correlate(h, 1-h, mode='full')) / N * 2"""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _objective(x):
    """Objective: minimize max correlation / N * 2. We minimize the max correlation directly."""
    h = x
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)


def _objective_smooth(x, lam=50.0):
    """Smooth approximation of max using log-sum-exp."""
    h = x
    corr = np.correlate(h, 1 - h, mode="full")
    # Scale for numerical stability
    m = np.max(corr)
    if m > 1e15:
        return m
    return m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam


def _constraints(h):
    """Constraints: sum(h) = N/2, 0 <= h[i] <= 1"""
    return [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0},
    ]


def _bounds():
    return [(0.0, 1.0) for _ in range(N)]


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    if abs(diff) < 1e-12:
        return h
    # Distribute the correction
    h = h - diff / N
    h = np.clip(h, 0.0, 1.0)
    # Second pass
    diff = np.sum(h) - N / 2.0
    h = h - diff / N
    h = np.clip(h, 0.0, 1.0)
    return h


def _local_search_polish(h, max_iters=10):
    """
    Simple local search: try moving delta from i to j.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    deltas = [0.01, 0.05, 0.1]
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j: continue
                for d in deltas:
                    # Move d from i to j
                    if best_h[i] < d: continue
                    if best_h[j] > 1 - d: continue
                    new_h = best_h.copy()
                    new_h[i] -= d
                    new_h[j] += d
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                if improved: break
            if improved: break
        if not improved: break
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial feasible point
    h0 = np.full(N, 0.5)
    
    best_h = h0.copy()
    best_val = _correlation_value(best_h)
    
    # Try multiple initializations and optimization passes
    rng = np.random.default_rng(42)
    
    # Different starting points
    starts = [h0.copy()]
    
    # Random feasible starts
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)
    
    # Structured starts: alternating patterns
    for period in [2, 3, 4, 5]:
        h_pat = np.zeros(N)
        for i in range(N):
            h_pat[i] = 1.0 if (i % period) < period // 2 else 0.0
        h_pat = _project_to_feasible(h_pat)
        starts.append(h_pat)
    
    # Ramp patterns
    for slope in [0.02, 0.05, 0.1]:
        h_ramp = np.linspace(0.5 - slope * N / 2, 0.5 + slope * N / 2, N)
        h_ramp = _project_to_feasible(h_ramp)
        starts.append(h_ramp)
    
    # Sine patterns
    for freq in [1, 2, 3, 5, 7]:
        h_sin = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        h_sin = _project_to_feasible(h_sin)
        starts.append(h_sin)
    
    # Optimize from each start
    for idx, h_start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        
        h_start = _project_to_feasible(h_start)
        
        # Sequence of smooth optimizations with increasing lambda
        lambdas = [10.0, 50.0, 200.0, 1000.0]
        
        current_h = h_start
        for lam in lambdas:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    lambda x: _objective_smooth(x, lam=lam),
                    current_h,
                    method='SLSQP',
                    bounds=_bounds(),
                    constraints=_constraints(current_h),
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Final exact objective optimization
        if time.time() > deadline - 5.0:
            break
        
        try:
            res_exact = minimize(
                _objective,
                current_h,
                method='SLSQP',
                bounds=_bounds(),
                constraints=_constraints(current_h),
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            h_opt_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_opt_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_opt_exact.copy()
        except Exception:
            pass

    # Final local search polish
    if time.time() < deadline - 5.0:
        try:
            best_h = _local_search_polish(best_h, max_iters=5)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # Final projection to ensure feasibility
    best_h = _project_to_feasible(best_h)
    
    # Verify
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
# EVOLVE-BLOCK-END
