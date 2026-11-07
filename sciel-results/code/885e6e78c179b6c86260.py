"""Erdos minimum overlap: binary-vertex search + continuous polish.

The correlation objective is multi-linear in h, so its minimum over the
feasible polytope lies on a vertex (all h[i] in {0,1}). We exploit this:
  Phase 1: Simulated annealing over binary assignments (exactly N/2 ones)
  Phase 2: Continuous polish of the ~N/2 boundary points via SLSQP
  Phase 3: High-precision final polish
"""
N = 100
import time
import numpy as np
from scipy.optimize import minimize


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _correlation_raw(h):
    """Raw max correlation (no normalization)."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)


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


def _binary_value(h_bin):
    """Compute max correlation for a binary vector (h in {0,1}).
    
    For binary h: h[i]*(1-h[j]) = h[i]*h[j] (since 1-h[j] = h[j] for binary)
    So corr[k] = sum_i h[i] * h[i-k] (circular-ish, but np.correlate is linear)
    """
    corr = np.correlate(h_bin.astype(float), h_bin.astype(float), mode="full")
    return np.max(corr)


def _sa_binary_search(start_time, time_budget, seed=42):
    """Simulated annealing over binary assignments with exactly N/2 ones."""
    rng = np.random.default_rng(seed)
    
    # Initial: alternating pattern
    h_bin = np.zeros(N, dtype=int)
    h_bin[::2] = 1  # 50 ones at even positions
    
    best_h = h_bin.copy()
    best_val = _binary_value(best_h)
    current_val = best_val
    
    # SA schedule
    T_start = 3.0
    T_end = 0.01
    n_steps = int(time_budget * 1000)  # ~1000 steps per second
    # Use geometric cooling
    if n_steps < 100:
        n_steps = 100
    cooling = (T_end / T_start) ** (1.0 / n_steps)
    
    start = time.time()
    
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        
        T = T_start * (cooling ** step)
        
        # Pick a random 1 to flip to 0
        ones = np.where(h_bin == 1)[0]
        zeros = np.where(h_bin == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        
        # Flip
        h_bin[i] = 0
        h_bin[j] = 1
        
        new_val = _binary_value(h_bin)
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h_bin.copy()
        else:
            # Revert
            h_bin[i] = 1
            h_bin[j] = 0
    
    return best_h, best_val


def _multi_start_sa(start_time, time_budget, n_restarts=5):
    """Run SA from multiple diverse starting points."""
    best_h = None
    best_val = float('inf')
    
    seeds = [42, 123, 456, 789, 1024, 2048]
    per_restart = time_budget / n_restarts
    
    for r in range(n_restarts):
        if time.time() - start_time > time_budget - 5:
            break
        
        seed = seeds[r % len(seeds)]
        h, val = _sa_binary_search(start_time, per_restart, seed=seed)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def _continuous_polish(h_bin, start_time, time_limit=40.0):
    """Polish the continuous values of the binary solution.
    
    The binary solution has N/2 ones and N/2 zeros. We keep the binary
    structure but allow the "1" positions to take values in [0.3, 1.0]
    and "0" positions in [0.0, 0.7] to satisfy the sum constraint while
    reducing the correlation.
    """
    h = h_bin.astype(float)
    
    # Which positions are "high" (were 1) and "low" (were 0)
    high_mask = h_bin == 1
    low_mask = h_bin == 0
    n_high = np.sum(high_mask)
    n_low = np.sum(low_mask)
    
    # Bounds: high positions in [0.3, 1.0], low in [0.0, 0.7]
    bounds = []
    for i in range(N):
        if high_mask[i]:
            bounds.append((0.3, 1.0))
        else:
            bounds.append((0.0, 0.7))
    
    def obj(x):
        corr = np.correlate(x, 1 - x, mode="full")
        return np.max(corr)
    
    def obj_grad(x):
        # Numerical gradient
        g = np.zeros(N)
        eps = 1e-6
        base = obj(x)
        for i in range(N):
            x[i] += eps
            g[i] = (obj(x) - base) / eps
            x[i] -= eps
        return g
    
    # Use SLSQP with the sum constraint
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    
    try:
        res = minimize(
            obj,
            h,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        h_opt = _project_to_feasible(res.x)
        val = _correlation_value(h_opt)
        return h_opt, val
    except Exception:
        return h, _correlation_value(h)


def _final_polish(h, start_time, time_limit=20.0):
    """Final high-precision SLSQP polish."""
    def obj(x):
        corr = np.correlate(x, 1 - x, mode="full")
        return np.max(corr)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    
    try:
        res = minimize(
            obj,
            h,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * N,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        h_opt = _project_to_feasible(res.x)
        val = _correlation_value(h_opt)
        return h_opt, val
    except Exception:
        return h, _correlation_value(h)


def _local_polish_binary(h_bin, start_time, time_limit=10.0):
    """Local search on binary assignments: try single swaps and accept if better."""
    best_h = h_bin.copy()
    best_val = _binary_value(best_h)
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        ones = np.where(best_h == 1)[0]
        zeros = np.where(best_h == 0)[0]
        
        for i in ones:
            if time.time() - start > time_limit:
                break
            for j in zeros:
                if time.time() - start > time_limit:
                    break
                best_h[i] = 0
                best_h[j] = 1
                val = _binary_value(best_h)
                if val < best_val:
                    best_val = val
                    improved = True
                else:
                    best_h[i] = 1
                    best_h[j] = 0
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Multi-restart simulated annealing over binary assignments
    sa_deadline = start_time + 60.0
    try:
        h_bin, val_bin = _multi_start_sa(start_time, 55.0, n_restarts=5)
        if h_bin is not None:
            # Convert binary to continuous (exact binary)
            h_cont = h_bin.astype(float)
            val_cont = _correlation_value(h_cont)
            if val_cont < best_val:
                best_val = val_cont
                best_h = h_cont.copy()
    except Exception:
        pass
    
    # Phase 2: Local binary polish
    if time.time() < deadline - 50.0:
        try:
            h_bin_pol, val_pol = _local_polish_binary(
                h_bin if h_bin is not None else np.array([1 if i%2==0 else 0 for i in range(N)]),
                start_time, 10.0
            )
            h_cont = h_bin_pol.astype(float)
            val_cont = _correlation_value(h_cont)
            if val_cont < best_val:
                best_val = val_cont
                best_h = h_cont.copy()
        except Exception:
            pass
    
    # Phase 3: Continuous polish of the binary solution
    if time.time() < deadline - 30.0:
        try:
            h_pol, val_pol = _continuous_polish(
                best_h, start_time, 30.0
            )
            if val_pol < best_val:
                best_val = val_pol
                best_h = h_pol.copy()
        except Exception:
            pass
    
    # Phase 4: Multi-start SLSQP from the binary solution and variations
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(999)
        
        # Start from current best
        starts = [best_h.copy()]
        
        # Perturbed versions of the binary solution
        for _ in range(4):
            h_pert = best_h.copy()
            # Randomly adjust a few values
            for _ in range(5):
                idx = rng.integers(N)
                h_pert[idx] = rng.uniform(0, 1)
            h_pert = _project_to_feasible(h_pert)
            starts.append(h_pert)
        
        # SLSQP from each
        for h_start in starts:
            if time.time() > deadline - 10.0:
                break
            
            def obj(x):
                corr = np.correlate(x, 1 - x, mode="full")
                return np.max(corr)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
            
            try:
                res = minimize(
                    obj,
                    h_start,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-13}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            h_final, val_final = _final_polish(best_h, start_time, 10.0)
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
