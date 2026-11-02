import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    # h is array of length N
    # overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    # = N/2 - autocorr[k]
    # Compute all autocorrelations via FFT
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # This is the circular autocorrelation
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i] * h[(i+k) % N] (circular)
    # Actually, need to be careful: ifft(|F|^2) gives sum_i h[i] * h[(i-j) % N]
    # Let me just compute directly for safety, or verify
    # overlap_k = N/2 - autocorr[k]
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def _overlap_max_direct(h):
    """Direct computation for verification."""
    w = 2.0 / N
    max_ov = 0.0
    for k in range(N):
        ov = 0.0
        for i in range(N):
            ov += h[i] * (1.0 - h[(i + k) % N])
        if ov > max_ov:
            max_ov = ov
    return w * max_ov

def construct():
    deadline = time.time() + 110  # safety margin
    
    # Initial guess: all 0.5
    h0 = np.full(N, 0.5)
    
    # Try a few different initializations and optimizers
    best_h = h0.copy()
    best_val = _overlap_max(h0)
    
    # Try structured initializations
    inits = [
        np.full(N, 0.5),
        np.tile([0.0, 1.0], N // 2).astype(float),
        np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False)) * 0.3 + 0.5,
    ]
    
    # Ensure sum = N/2 for each init
    for i, h_init in enumerate(inits):
        if time.time() > deadline:
            break
        # Project to sum = N/2
        h_init = np.clip(h_init, 0, 1)
        diff = N / 2.0 - np.sum(h_init)
        h_init = h_init + diff / N
        h_init = np.clip(h_init, 0, 1)
        # Re-adjust sum
        for _ in range(10):
            diff = N / 2.0 - np.sum(h_init)
            if abs(diff) < 1e-10:
                break
            h_init = h_init + diff / N
            h_init = np.clip(h_init, 0, 1)
        
        val = _overlap_max(h_init)
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
        
        # Try SLSQP optimization
        try:
            res = minimize(
                _overlap_max, h_init, method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if time.time() > deadline:
                break
            if res.success or res.fun < best_val:
                h_opt = np.clip(res.x, 0, 1)
                # Enforce sum constraint
                diff = N / 2.0 - np.sum(h_opt)
                h_opt = h_opt + diff / N
                h_opt = np.clip(h_opt, 0, 1)
                val = _overlap_max(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check
    if time.time() > deadline - 5:
        return best_h
    
    # One more refinement
    try:
        res = minimize(
            _overlap_max, best_h, method='SLSQP',
            bounds=[(0, 1)] * N,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if res.fun < best_val:
            h_opt = np.clip(res.x, 0, 1)
            diff = N / 2.0 - np.sum(h_opt)
            h_opt = h_opt + diff / N
            h_opt = np.clip(h_opt, 0, 1)
            best_h = h_opt
    except Exception:
        pass
    
    return best_h
