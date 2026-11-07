import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
PHI = (1 + np.sqrt(5)) / 2.0  # golden ratio

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _raw_max_corr(h):
    """Compute just the max correlation value (for optimization)."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(200):
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

def _smooth_max_corr(h, lam=50.0):
    """Smooth approximation of max(corr) using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    return m + np.log(np.sum(np.exp((corr - m) * lam))) / lam

def _golden_section_1d(f, a, b, tol=1e-10, max_iter=100):
    """Golden section search for 1D minimization."""
    invphi = 2.0 / (1.0 + np.sqrt(5.0))
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = f(d)
    return (a + b) / 2.0

def _pair_line_search(h, i, j, tol=1e-10):
    """
    Optimize along the 1D line h[i] += d, h[j] -= d (preserving sum).
    Returns updated h.
    """
    hi = h[i]
    hj = h[j]
    
    # Feasible range for d:
    # 0 <= hi + d <= 1  =>  -hi <= d <= 1 - hi
    # 0 <= hj - d <= 1  =>  hj - 1 <= d <= hj
    d_min = max(-hi, hj - 1.0)
    d_max = min(1.0 - hi, hj)
    
    if d_max - d_min < 1e-14:
        return h
    
    def obj(d):
        new_h = h.copy()
        new_h[i] = hi + d
        new_h[j] = hj - d
        return _raw_max_corr(new_h)
    
    # Golden section search on [d_min, d_max]
    d_opt = _golden_section_1d(obj, d_min, d_max, tol=tol)
    
    new_h = h.copy()
    new_h[i] = hi + d_opt
    new_h[j] = hj - d_opt
    new_h = np.clip(new_h, 0.0, 1.0)
    return new_h

def _local_search(h, max_iters=5000, time_limit=None, n_pairs_per_iter=10):
    """
    Local search using golden-section pair line searches.
    Each iteration picks random pairs and optimizes along that 1D direction.
    """
    best_h = h.copy()
    best_val = _raw_max_corr(best_h)
    
    start_time = time.time()
    if time_limit is None:
        time_limit = start_time + 30.0
    
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        improved = False
        
        for _ in range(n_pairs_per_iter):
            if time.time() > time_limit:
                break
            
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            new_h = _pair_line_search(best_h, i, j)
            new_val = _raw_max_corr(new_h)
            
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_h = new_h
                improved = True
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 200:
                break
    
    return best_h

def _incommensurate_pattern(freqs, phases, n_harmonics):
    """
    Generate a pattern using incommensurate (golden-ratio related) frequencies.
    This creates aperiodic patterns that spread correlation energy.
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(n_harmonics):
        freq = freqs[k]
        phase = phases[k]
        h += np.cos(2 * np.pi * freq * i / N + phase)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential Evolution on Fourier Coefficients with incommensurate frequencies
    # Use golden-ratio related frequencies for aperiodic patterns
    n_harm = 20
    n_params = 2 * n_harm
    i = np.arange(N)
    
    # Incommensurate frequencies based on golden ratio
    base_freqs = []
    for k in range(1, n_harm + 1):
        # Use golden-ratio scaled frequencies to avoid periodicity
        freq = k / PHI  # incommensurate with N
        base_freqs.append(freq)
    
    cols = []
    for k in range(n_harm):
        freq = base_freqs[k]
        cols.append(np.cos(2 * np.pi * freq * i / N))
        cols.append(np.sin(2 * np.pi * freq * i / N))
    basis = np.column_stack(cols)
    
    def fourier_objective(params):
        h = np.full(N, 0.5)
        h += params @ basis
        h = _project_to_feasible(h)
        return _raw_max_corr(h)
    
    bounds = [(-0.5, 0.5)] * n_params
    
    de_deadline = start_time + 40.0
    
    def timed_fourier_obj(params):
        if time.time() > de_deadline:
            raise TimeoutError()
        return fourier_objective(params)
    
    try:
        result = differential_evolution(
            timed_fourier_obj,
            bounds=bounds,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            init='sobol'
        )
        if result.x is not None:
            h_de = 0.5 + result.x @ basis
            h_de = _project_to_feasible(h_de)
            val_de = _raw_max_corr(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Local Search with golden-section pair line searches
    ls_deadline = deadline - 25.0
    if time.time() < ls_deadline:
        best_h = _local_search(best_h, time_limit=ls_deadline, n_pairs_per_iter=15)
        best_val = _raw_max_corr(best_h)
    
    # Phase 3: SLSQP with smooth objective
    if time.time() < deadline - 10.0:
        try:
            for lam in [50.0, 100.0, 200.0]:
                if time.time() > deadline - 5.0:
                    break
                
                def smooth_obj(x, lam=lam):
                    return _smooth_max_corr(x, lam=lam)
                
                def constraint(x):
                    return np.sum(x) - N / 2.0
                
                res = minimize(
                    smooth_obj,
                    best_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': constraint}],
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res.success or res.fun < best_val:
                    h_opt = _project_to_feasible(res.x)
                    val_opt = _raw_max_corr(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision local search with more iterations
    if time.time() < deadline - 1.0:
        best_h = _local_search(best_h, max_iters=10000, time_limit=deadline - 0.5, n_pairs_per_iter=20)
        best_val = _raw_max_corr(best_h)
        
    best_h = _project_to_feasible(best_h)
    
    return best_h
