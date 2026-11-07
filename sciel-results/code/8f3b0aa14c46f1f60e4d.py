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


def _local_polish(h, max_iters=50, step_sizes=None, time_limit=30.0):
    """
    Fast local search: coordinate descent with sum constraint.
    Uses efficient candidate selection and time limit.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    start_time = time.time()
    
    if step_sizes is None:
        step_sizes = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try adjusting each element by each step size
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_val_i = best_h[i] + sign * d
                    if new_val_i < 0 or new_val_i > 1:
                        continue
                    # Need to adjust another element to maintain sum
                    needed = -sign * d
                    # Find best j: prefer elements far from i (less correlation impact)
                    best_j = -1
                    best_j_val = None
                    # Check a subset of candidates for speed
                    candidates = []
                    for j in range(n):
                        if j == i:
                            continue
                        new_val_j = best_h[j] + needed
                        if 0 <= new_val_j <= 1:
                            # Heuristic: distance from i
                            dist = min(abs(j - i), n - abs(j - i))
                            candidates.append((dist, j, new_val_j))
                    if not candidates:
                        continue
                    # Sort by distance (farthest first) but check a few close ones too
                    candidates.sort(key=lambda x: -x[0])
                    # Check up to 3 candidates
                    for dist, j, new_val_j in candidates[:3]:
                        new_h = best_h.copy()
                        new_h[i] = new_val_i
                        new_h[j] = new_val_j
                        val = _correlation_value(new_h)
                        if val < best_val - 1e-14:
                            best_val = val
                            best_h = new_h
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over basis coefficients
    # Use a moderate number of basis functions
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    # Run differential evolution with time budget
    de_deadline = start_time + 40.0
    
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
    for _ in range(8):
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
        for lam in [50.0, 200.0]:
            if time.time() > slsqp_deadline - 10.0:
                break
            
            def smooth_obj(x, lam=lam):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                m = np.max(corr)
                if m > 1e15:
                    return m
                # Use log-sum-exp for numerical stability
                shifted = (corr - m) * lam
                max_shifted = np.max(shifted)
                if max_shifted > 700:  # Avoid overflow
                    return m + max_shifted / lam
                return m + np.log(np.sum(np.exp(shifted)) / len(corr)) / lam
            
            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 150, 'ftol': 1e-12}
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
        if time.time() > slsqp_deadline - 8.0:
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
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    # Phase 3: Local search polish
    if time.time() < deadline - 10.0:
        try:
            best_h = _local_polish(best_h, max_iters=50, time_limit=25.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 4: Final high-precision SLSQP polish
    if time.time() < deadline - 5.0:
        try:
            def exact_obj(x):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                return np.max(corr)
            
            res_final = minimize(
                exact_obj,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_final = _project_to_feasible(res_final.x)
            val_final = _correlation_value(h_final)
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
