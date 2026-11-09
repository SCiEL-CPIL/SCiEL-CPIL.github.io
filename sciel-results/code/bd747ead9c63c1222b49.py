import numpy as np
import time
import scipy.optimize as opt

N = 600
DELTA = 1.0 / N

def compute_ratio(h):
    """Compute max(f*f) / (int f)^2 for step function with heights h."""
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e15
    v = h / s  # normalize so sum(v) = 1
    # Compute autoconvolution of v
    # conv[k] = sum_i v[i] v[i+k]
    # Using FFT for speed
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(v, nfft)
    conv = np.fft.irfft(F * F, nfft)[:2*N-1]
    # The value of the continuous f*f max is approx max(conv) / DELTA
    # But the score definition in the prompt implies:
    # Score = max(f*f) / (int f)^2
    # With int f = 1, Score = max(f*f)
    # f*f(t) approx conv[k] * DELTA?
    # Let's stick to the code's logic:
    # max_val = max(conv) * DELTA
    # But wait, the previous analysis showed Score = N * max(conv(v))
    # Let's verify.
    # v = h/s. s = sum(h)*DELTA.
    # If h=1, s=N*DELTA=1. v=1.
    # conv(v) max = N.
    # Score = N * DELTA = 1.
    # My formula: N * max(conv(v)) = N * N = N^2. Incorrect.
    # Let's re-derive.
    # h_norm = h/s.
    # conv(h_norm) max = M.
    # Score = M * DELTA.
    # h_norm = v / DELTA.
    # conv(v/DELTA) = conv(v) / DELTA^2.
    # M = max(conv(v)) / DELTA^2.
    # Score = max(conv(v)) / DELTA.
    # Since DELTA = 1/N, Score = N * max(conv(v)).
    # For uniform v (sum 1), v_i = 1/N.
    # conv(v)_0 = sum (1/N)^2 = N * 1/N^2 = 1/N.
    # Score = N * (1/N) = 1.
    # Correct.
    return N * np.max(conv)

def smooth_max(x, alpha=100):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp((x - x_max) / alpha))) / alpha

def objective(v_flat, alpha=100):
    v = np.maximum(v_flat, 0)
    s = np.sum(v)
    if s < 1e-15:
        return 1e15
    v = v / s
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(v, nfft)
    conv = np.fft.irfft(F * F, nfft)[:2*N-1]
    return smooth_max(conv, alpha)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Initial candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Cosine window
    x = np.linspace(0, np.pi, N)
    candidates.append(np.cos(x)**2 / np.sum(np.cos(x)**2))
    
    # Two bumps
    centers = np.array([N*0.25, N*0.75])
    sigma = N * 0.1
    h = np.zeros(N)
    for c in centers:
        h += np.exp(-((np.arange(N) - c)**2) / (2*sigma**2))
    candidates.append(h / np.sum(h))
    
    best_v = None
    best_score = 1e15
    
    for v0 in candidates:
        v0 = np.maximum(v0, 0)
        v0 = v0 / np.sum(v0)
        
        # Optimize
        try:
            res = opt.minimize(objective, v0, method='L-BFGS-B', 
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
            v_opt = res.x
            v_opt = np.maximum(v_opt, 0)
            if np.sum(v_opt) < 1e-15:
                continue
            v_opt = v_opt / np.sum(v_opt)
            
            score = compute_ratio(v_opt)
            if score < best_score:
                best_score = score
                best_v = v_opt
        except Exception:
            pass
        
        if time.time() - start_time > deadline - 10:
            break
            
    if best_v is None:
        best_v = np.ones(N) / N
        best_score = compute_ratio(best_v)
        
    # Final refinement: Coordinate descent on exact max
    v = best_v.copy()
    current_score = best_score
    
    for iter in range(200):
        if time.time() - start_time > deadline - 5:
            break
        improved = False
        # Random subset of indices for speed
        indices = np.random.choice(N, size=min(50, N), replace=False)
        for i in indices:
            old_val = v[i]
            # Try scaling
            for frac in [0.5, 0.8, 1.2, 1.5, 0.0, 2.0]:
                new_val = old_val * frac
                v_test = v.copy()
                v_test[i] = new_val
                v_test = np.maximum(v_test, 0)
                s = np.sum(v_test)
                if s < 1e-15:
                    continue
                v_test = v_test / s
                score = compute_ratio(v_test)
                if score < current_score - 1e-10:
                    current_score = score
                    v = v_test
                    improved = True
                    break
        if not improved:
            break
            
    # Convert back to heights h for the interface
    # The interface expects h such that the evaluator normalizes or uses as is?
    # "construct() -> sequence of nonnegative step heights"
    # The evaluator likely computes int f and max f*f from these heights.
    # Since the score is normalized, we can return any scaling.
    # Returning v (sum 1) is safe.
    return [float(x) for x in v]
