"""Model-and-optimize approach for Erdos minimum overlap.

Key insight: the max correlation value is driven by the peaks of the
correlation function. We can parameterize h as a smooth function of a
few basis coefficients, then directly optimize those parameters.
"""
N = 100
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


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


def _local_polish(h, max_iters=50, step_sizes=None):
    """
    Local search: try small adjustments to individual elements.
    Uses a smarter pair-swap strategy to maintain the sum constraint.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    for it in range(max_iters):
        improved = False
        # Try adjusting pairs of elements
        for i in range(n):
            for d in step_sizes:
                for sign in [+1, -1]:
                    delta = sign * d
                    if best_h[i] + delta < -1e-12 or best_h[i] + delta > 1 + 1e-12:
                        continue
                    # Find best partner j to compensate
                    needed = -delta
                    # Try all valid j and pick the one that gives best value
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


def _epigraph_polish(h, max_iters=200, tol=1e-15):
    """
    Epigraph formulation: minimize t subject to corr[k] <= t for all k,
    sum(h) = N/2, 0 <= h <= 1.
    Variables: [h_0, ..., h_{N-1}, t]
    """
    init_t = _correlation_value(h)
    x0 = np.concatenate([h, [init_t]])
    
    def objective(x):
        return x[-1]  # minimize t
    
    def smooth_max_constraint(x):
        h_x = x[:N]
        t = x[-1]
        corr = np.correlate(h_x, 1 - h_x, mode="full")
        # Soft max with large lambda
        lam = 1000.0
        m = np.max(corr)
        if m > 1e15:
            return m - t
        soft_max = m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam
        return soft_max - t  # should be <= 0
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N / 2.0},
        {'type': 'ineq', 'fun': lambda x: -(smooth_max_constraint(x))}
    ]
    
    bounds = [(0.0, 1.0)] * N + [(-1.0, 10.0)]  # t can be negative
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iters, 'ftol': tol}
        )
        h_opt = _project_to_feasible(res.x[:N])
        return h_opt, _correlation_value(h_opt)
    except Exception:
        return h, _correlation_value(h)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over basis coefficients
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    de_deadline = start_time + 45.0
    
    try:
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(coeffs)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-8,
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
    
    # Phase 2: Multi-start SLSQP with soft-max surrogate
    rng = np.random.default_rng(123)
    
    starts = [best_h.copy()]
    
    for _ in range(6):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)
    
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
    
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        amp = min(0.4, 0.5 / freq)
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))
    
    for period in [10, 20, 25]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))
    
    slsqp_deadline = start_time + 105.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 15.0:
            break
        
        h_start = _project_to_feasible(h_start)
        
        current_h = h_start
        for lam in [50.0, 200.0, 1000.0]:
            if time.time() > slsqp_deadline - 10.0:
                break
            
            def smooth_obj(x, lam=lam):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                m = np.max(corr)
                if m > 1e15:
                    return m
                return m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam
            
            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Epigraph polish
        if time.time() > slsqp_deadline - 5.0:
            break
        
        try:
            h_ep, val_ep = _epigraph_polish(current_h, max_iters=100, tol=1e-15)
            if val_ep < best_val:
                best_val = val_ep
                best_h = h_ep.copy()
        except Exception:
            pass
    
    # Phase 3: Local search polish
    if time.time() < deadline - 10.0:
        try:
            best_h = _local_polish(best_h, max_iters=40)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 4: Final epigraph refinement
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
