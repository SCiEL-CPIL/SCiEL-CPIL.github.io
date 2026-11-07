import numpy as np
from scipy.optimize import minimize
import time

N = 100
HALF = N / 2.0  # 50

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    if np.any(h < -1e-12) or np.any(h > 1.0 + 1e-12):
        return np.inf
    if abs(np.sum(h) - HALF) > 1e-9 * N:
        return np.inf
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr)) / N * 2.0

def _corr_max(h):
    """Fast computation of max correlation (no validity checks)."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _fix_sum(h):
    """Adjust h to have exact sum = HALF, keeping in [0,1]."""
    h = np.asarray(h, dtype=np.float64)
    diff = HALF - np.sum(h)
    if abs(diff) < 1e-15:
        return h
    # Distribute correction evenly, then clip and iterate
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    # Second pass for residual
    diff = HALF - np.sum(h)
    if abs(diff) > 1e-14:
        # Adjust only elements that can absorb the change
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if mask.any():
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
            diff = HALF - np.sum(h)
            if abs(diff) > 1e-14:
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
    return h

def _sa_move(h, rng, move_type=None):
    """
    Perform a sum-preserving move on h.
    move_type: 0=swap, 1=paired perturb, 2=block shift, 3=None (return h)
    Returns new h (may be same object if no move).
    """
    if move_type is None:
        move_type = rng.integers(0, 3)
    
    if move_type == 0:
        # Swap two elements
        i, j = rng.integers(0, N, size=2)
        if i == j:
            return h
        h_new = h.copy()
        h_new[i], h_new[j] = h[j], h[i]
        return h_new
    elif move_type == 1:
        # Paired perturbation: add delta to one, subtract from another
        i, j = rng.integers(0, N, size=2)
        if i == j:
            return h
        delta = rng.uniform(-0.3, 0.3)
        h_new = h.copy()
        h_new[i] += delta
        h_new[j] -= delta
        h_new = np.clip(h_new, 0.0, 1.0)
        # Re-fix sum after clipping
        h_new = _fix_sum(h_new)
        return h_new
    else:
        # Block shift: shift a contiguous block by one position
        length = rng.integers(2, 11)
        start = rng.integers(0, N - length)
        h_new = h.copy()
        # Shift right by 1 within the block
        h_new[start:start+length-1] = h_new[start+1:start+length]
        h_new[start+length-1] = h_new[start]
        return h_new

def _simulated_annealing(h0, rng, t_start, deadline, max_iters):
    """Run SA with sum-preserving moves."""
    h = h0.copy()
    h = _fix_sum(h)
    best_h = h.copy()
    best_val = _corr_max(h)
    
    T_start = 0.5
    T_end = 0.001
    # Geometric cooling
    alpha = (T_end / T_start) ** (1.0 / max(max_iters, 1))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        T = T_start * (alpha ** it)
        
        h_new = _sa_move(h, rng)
        val_new = _corr_max(h_new)
        delta = val_new - best_val
        
        if delta < 0:
            # Better: always accept
            h = h_new
            best_h = h_new.copy()
            best_val = val_new
        else:
            # Worse: accept with probability exp(-delta/T)
            if T > 0 and delta < 100 * T:
                if rng.random() < np.exp(-delta / T):
                    h = h_new
            # Otherwise reject, keep h
    
    return best_h, best_val

def _slsqp_polish(h0, deadline):
    """Polish with SLSQP."""
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}]
    
    def objective(x):
        h = np.clip(x, 0.0, 1.0)
        return _corr_max(h)
    
    try:
        res = minimize(
            objective, h0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-14}
        )
        h_candidate = np.clip(res.x, 0.0, 1.0)
        h_candidate = _fix_sum(h_candidate)
        val = _value(h_candidate)
        if np.isfinite(val):
            return h_candidate, val
    except Exception:
        pass
    return None, np.inf

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)  # seed value
    
    # Generate diverse initializations
    inits = []
    
    # 1. Seed
    inits.append(np.full(N, 0.5))
    
    # 2. Random
    for _ in range(5):
        hr = rng.uniform(0, 1, N)
        inits.append(_fix_sum(hr))
    
    # 3. Sine waves with various frequencies/phases
    for freq in [1, 2, 3, 5, 8]:
        for phase in [0, np.pi/2, np.pi]:
            hs = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            inits.append(_fix_sum(np.clip(hs, 0, 1)))
    
    # 4. Step functions
    for split in [25, 33, 50, 67, 75]:
        hstep = np.zeros(N)
        hstep[:split] = 1.0
        hstep[split:] = 0.0
        # Scale to have sum = 50
        hstep = hstep * (50.0 / split) if split > 0 else hstep
        inits.append(_fix_sum(np.clip(hstep, 0, 1)))
    
    # 5. Gradients
    for slope in [0.01, 0.02, 0.05, -0.01, -0.02]:
        hg = 0.5 + slope * (np.arange(N) - N/2)
        inits.append(_fix_sum(np.clip(hg, 0, 1)))
    
    # 6. Alternating patterns
    for period in [2, 3, 4, 5]:
        halt = np.zeros(N)
        for i in range(N):
            halt[i] = 1.0 if (i % period) < period // 2 else 0.0
        inits.append(_fix_sum(halt))
    
    # 7. Cosine windows
    for width in [20, 40, 60]:
        hc = np.zeros(N)
        center = N // 2
        for i in range(N):
            d = abs(i - center)
            if d < width:
                hc[i] = 0.5 + 0.4 * np.cos(np.pi * d / width)
        inits.append(_fix_sum(np.clip(hc, 0, 1)))
    
    # Run SA on each initialization
    sa_iters_per_init = 3000
    
    for idx, h_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        h_sa, val_sa = _simulated_annealing(
            h_init, rng, t_start, deadline, sa_iters_per_init
        )
        
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa.copy()
    
    # Final SLSQP polish on best result
    if time.time() < deadline - 10:
        h_polished, val_polished = _slsqp_polish(best_h, deadline)
        if h_polished is not None and val_polished < best_val:
            best_h = h_polished
            best_val = val_polished
    
    # One more SA round on the polished result if time permits
    if time.time() < deadline - 5:
        h_sa2, val_sa2 = _simulated_annealing(
            best_h, rng, t_start, deadline, 2000
        )
        if val_sa2 < best_val:
            best_h = h_sa2.copy()
            best_val = val_sa2
    
    # Final SLSQP polish
    if time.time() < deadline - 2:
        h_polished2, val_polished2 = _slsqp_polish(best_h, deadline)
        if h_polished2 is not None and val_polished2 < best_val:
            best_h = h_polished2
            best_val = val_polished2
    
    # Ensure validity
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)
    
    # Final validity check
    final_val = _value(best_h)
    if not np.isfinite(final_val):
        # Fallback to seed
        best_h = np.full(N, 0.5)
    
    return best_h.tolist()
