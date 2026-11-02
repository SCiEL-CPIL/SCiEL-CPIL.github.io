import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """Compute max_k S_k / S_0 where S_k is autocorrelation."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        h = np.clip(h, 0, None)
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    S = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    S0 = np.dot(h, h)
    if S0 < 1e-30:
        return np.inf
    return np.max(S) / S0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Strategy: Use superposition of narrow rectangular pulses
    # Optimize pulse positions and amplitudes using Nelder-Mead
    
    def build_h_from_params(params):
        """Build step function from pulse parameters.
        params: [pos1, width1, amp1, pos2, width2, amp2, ...]
        """
        h = np.zeros(N)
        n_pulses = len(params) // 3
        for i in range(n_pulses):
            pos = params[3*i]
            width = params[3*i+1]
            amp = params[3*i+2]
            # Map position from [0,1] to [0, N)
            center_idx = int(pos * N)
            half_w = int(width * N / 2)
            start_idx = max(0, center_idx - half_w)
            end_idx = min(N, center_idx + half_w + 1)
            h[start_idx:end_idx] = amp
        return h
    
    def objective(params):
        h = build_h_from_params(params)
        return compute_score(h)
    
    # Start with 3 pulses
    n_pulses = 3
    # Initial guess: evenly spaced pulses with moderate width
    x0 = []
    for i in range(n_pulses):
        pos = (i + 0.5) / n_pulses
        width = 0.1  # 10% of interval
        amp = 1.0
        x0.extend([pos, width, amp])
    
    best_h = build_h_from_params(x0)
    best_score = compute_score(best_h)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            objective, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        if result.fun < best_score:
            best_h = build_h_from_params(result.x)
            best_score = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return list(best_h)
    
    # Try with different numbers of pulses
    for n_p in [2, 4, 5]:
        if time.time() > deadline - 10:
            break
        x0 = []
        for i in range(n_p):
            pos = (i + 0.5) / n_p
            width = 0.08
            amp = 1.0
            x0.extend([pos, width, amp])
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if result.fun < best_score:
                best_h = build_h_from_params(result.x)
                best_score = result.fun
        except Exception:
            pass
    
    # Also try uniform as baseline
    h_uniform = np.ones(N) / N
    score_uniform = compute_score(h_uniform)
    if score_uniform < best_score:
        best_h = h_uniform
        best_score = score_uniform
    
    return list(best_h)

construct()
