"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(h):
    """Compute the ratio: sum(h^2) / (sum(h) * max(autoconv))"""
    h = np.asarray(h)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution: c_k = sum_i h_i * h_{i+k}
    # For discrete signal of length N, full autoconvolution has length 2N-1
    c = np.correlate(h, h, mode='full')
    max_c = np.max(np.abs(c))
    if max_c < 1e-12:
        return 0.0
    return l2_sq / (l1 * max_c)

def _neg_score(h):
    return -_score(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Strategy 1: Single spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    val = _score(h_spike)
    if val > best_val:
        best_val = val
        best_h = h_spike.copy()
    
    # Strategy 2: Try various spike positions and amplitudes
    for pos in range(N):
        for amp in [0.5, 1.0, 1.5, 2.0]:
            h = np.zeros(N)
            h[pos] = amp
            val = _score(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
    
    # Strategy 3: Local optimization from spike initializations
    def _local_opt(h0):
        h0 = np.asarray(h0, dtype=float)
        res = minimize(_neg_score, h0, method='L-BFGS-B', 
                      bounds=[(0, None)] * N,
                      options={'maxiter': 500, 'ftol': 1e-10})
        return res.x, -res.fun
    
    # Optimize from single spike
    h0 = np.zeros(N)
    h0[N//2] = 1.0
    try:
        h_opt, val_opt = _local_opt(h0)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return list(best_h)
    
    # Strategy 4: Try two-spike configurations
    for d in range(1, N):
        for pos in range(N - d):
            for a1 in [0.5, 1.0]:
                for a2 in [0.5, 1.0]:
                    h = np.zeros(N)
                    h[pos] = a1
                    h[pos + d] = a2
                    val = _score(h)
                    if val > best_val:
                        best_val = val
                        best_h = h.copy()
    
    if time.time() > deadline:
        return list(best_h)
    
    # Strategy 5: Local optimization from two-spike configs
    for d in [5, 10, 20, 25]:
        for pos in [0, N//2 - d//2]:
            if pos < 0 or pos + d >= N:
                continue
            h0 = np.zeros(N)
            h0[pos] = 1.0
            h0[pos + d] = 1.0
            try:
                h_opt, val_opt = _local_opt(h0)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    if time.time() > deadline:
        return list(best_h)
    
    # Strategy 6: Gaussian-like shapes
    for center in [N//4, N//2, 3*N//4]:
        for sigma in [1.0, 2.0, 3.0, 5.0]:
            x = np.arange(N)
            h0 = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            h0 = h0 / np.max(h0)  # normalize peak to 1
            try:
                h_opt, val_opt = _local_opt(h0)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    if time.time() > deadline:
        return list(best_h)
    
    # Strategy 7: Random restarts with local optimization
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        h0 = rng.random(N) * 0.5
        # Add a spike
        h0[rng.integers(N)] += 1.0
        try:
            h_opt, val_opt = _local_opt(h0)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final local polish
    try:
        h_opt, val_opt = _local_opt(best_h)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    except Exception:
        pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
