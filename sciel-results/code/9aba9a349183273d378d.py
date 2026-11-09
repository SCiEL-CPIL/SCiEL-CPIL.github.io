"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import sys

N = 50

# EVOLVE-BLOCK-START

def _autoconv(h):
    """Compute the autoconvolution of step heights h on equal intervals.
    
    For step functions on intervals of width w in [-1/4, 1/4], 
    the autoconvolution at lag k (k = -N+1..N-1) is:
    (h*h)(k) = sum_i h[i] * h[i-k] * (overlap length)
    
    For unit-width steps (w=1 for simplicity, since ratio is scale-invariant),
    the overlap at lag k is (N - |k|) if we think of it as discrete.
    
    Actually, for step functions: if h_i is the height on interval [i*w, (i+1)*w),
    then (f*f)(t) = integral f(t-s) f(s) ds.
    At t = k*w, the value is w * sum_i h_i * h_{i-k} * (1 - |k|/N_effective)... 
    
    Let me be more careful. The standard discrete autoconvolution for step functions:
    If f(x) = h_i for x in [i, i+1), i=0..N-1, then
    (f*f)(t) = integral_0^N f(t-s) f(s) ds
    
    For t = k (integer), this equals sum over pairs where the overlap of 
    [s, s+1) and [t-s, t-s+1) contributes. The key result is:
    (f*f)(k) = sum_i h_i * h_{i-k} for integer k, but with the overlap factor.
    
    Actually for unit-width steps, (f*f)(k) at integer k is exactly sum_i h_i h_{i-k}.
    But between integers, it's piecewise linear. The supremum occurs at integer points 
    for non-negative h.
    
    So sup|f*f| = max_k |sum_i h_i h_{i-k}| for k in range(-N+1, N).
    For non-negative h, this is max_k sum_i h_i h_{i-k}.
    """
    # Full autoconvolution
    conv = np.convolve(h, h, mode='full')
    return conv

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * sup|f*f|)
    
    Wait, re-reading: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    So it's ||h||_2^2 / (||h||_1 * ||f*f||_inf)
    
    For the uniform case: h = [1]*N
    ||h||_2^2 = N
    ||h||_1 = N
    f*f at 0 = N, and sup is N (at k=0)
    Ratio = N / (N * N) = 1/N = 1/50 = 0.02
    
    Hmm, but the seed scores 1.0 and the task says "score is the lower bound on C2"
    and "search score is value/seed". So the actual value for uniform is some baseline.
    
    Let me reconsider. Maybe it's ||h||_2^2 / (||h||_1 * sup|f*f|) but normalized differently.
    Or perhaps the "value" is computed differently.
    
    Actually, looking at inspiration 1 scoring 1.1298, the uniform scores 1.0.
    Let me just compute the ratio as stated and see what makes sense.
    
    For h = [1]*N:
    ||h||_2^2 = sum h_i^2 = N = 50
    ||h||_1 = sum h_i = 50
    sup|f*f| = max of autoconv = N = 50 (at lag 0)
    Ratio = 50 / (50 * 50) = 1/50 = 0.02
    
    If the seed value is 0.02 and score = value/seed_value, then uniform gives 1.0.
    So we want to maximize this ratio.
    """
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv = _autoconv(h)
    sup_conv = np.max(np.abs(conv))
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _neg_ratio(h):
    return -_score(h)

def _elapsed(start):
    return time.time() - start

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Multi-start L-BFGS-B optimization
    # Objective: maximize ||h||_2^2 / (||h||_1 * sup|f*f|)
    # This is scale-invariant, so we can fix ||h||_2 = 1 and maximize 1/(||h||_1 * sup|f*f|)
    # Or equivalently minimize ||h||_1 * sup|f*f| subject to ||h||_2 = 1, h >= 0
    
    def _obj_constrained(x):
        """Minimize ||h||_1 * sup|f*f| subject to ||h||_2 = 1 (handled by penalty or normalization)"""
        h = np.abs(x)  # enforce non-negativity
        norm = np.linalg.norm(h)
        if norm < 1e-15:
            return 1e10
        h_norm = h / norm
        l1 = np.sum(h_norm)
        conv = _autoconv(h_norm)
        sup_conv = np.max(np.abs(conv))
        return l1 * sup_conv
    
    def _obj_penalized(x):
        """Penalized version: minimize ||h||_1 * sup|f*f| + penalty*(||h||_2 - 1)^2"""
        h = np.maximum(x, 0)
        l2 = np.linalg.norm(h)
        if l2 < 1e-15:
            return 1e10
        h_norm = h / l2
        l1 = np.sum(h_norm)
        conv = _autoconv(h_norm)
        sup_conv = np.max(np.abs(conv))
        return l1 * sup_conv + 100.0 * (l2 - 1.0) ** 2
    
    # Diverse initializations
    inits = []
    # Uniform
    inits.append(np.ones(N))
    # Concentrated center
    for width in [5, 10, 15, 20]:
        h = np.zeros(N)
        center = N // 2
        start_idx = max(0, center - width // 2)
        end_idx = min(N, start_idx + width)
        h[start_idx:end_idx] = 1.0
        inits.append(h)
    # Sparse: few peaks
    for n_peaks in [2, 3, 5]:
        h = np.zeros(N)
        positions = np.linspace(5, N-5, n_peaks).astype(int)
        for p in positions:
            h[max(0, p-1):min(N, p+2)] = 1.0
        inits.append(h)
    # Multi-peak with gaps
    h = np.zeros(N)
    for i in range(0, N, 10):
        h[i:i+3] = 1.0
    inits.append(h)
    # Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        h = rng.random(N)
        h[h < 0.3] = 0
        inits.append(h)
    
    # Normalize initializations
    inits = [h / (np.linalg.norm(h) + 1e-15) if np.linalg.norm(h) > 1e-15 else h for h in inits]
    
    # L-BFGS-B optimization from each start
    for i, h0 in enumerate(inits):
        if _elapsed(start) > deadline - 5.0:
            break
        
        try:
            res = minimize(
                _obj_constrained,
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = np.maximum(res.x, 0)
            if np.linalg.norm(h_opt) > 1e-15:
                h_opt = h_opt / np.linalg.norm(h_opt)
                val = _score(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: more iterations on best
    if _elapsed(start) < deadline - 10.0:
        try:
            res = minimize(
                _obj_constrained,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            if np.linalg.norm(h_opt) > 1e-15:
                h_opt = h_opt / np.linalg.norm(h_opt)
                val = _score(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Return the heights (scale doesn't matter for the ratio, but keep them reasonable)
    return list(best_h)

# EVOLVE-BLOCK-END
