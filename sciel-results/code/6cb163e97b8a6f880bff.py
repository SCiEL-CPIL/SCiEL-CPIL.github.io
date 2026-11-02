"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 140  # 10s safety margin

def _autoconv_l1_bound(h):
    """Compute an upper bound related to the autoconvolution L1 norm.
    
    For a step function with heights h on N equal intervals over [-1/4, 1/4],
    the autoconvolution (f*f)(x) = integral f(x-t) f(t) dt.
    
    We approximate the L1 norm of |f*f| as the objective to minimize.
    """
    w = (0.5) / N  # width of each interval
    # f*f at the center of each output interval
    # The autoconvolution has support [-1/2, 1/2], discretized into 2N-1 points
    # For step functions, (f*f) is piecewise linear
    # Approximate: (f*f)[k] for k in 0..2N-2 represents the value at offset k*w from -1/2
    
    # More precisely, for interval i (height h[i]) and interval j (height h[j]),
    # their product contributes to the autoconvolution where the intervals overlap.
    # The overlap length for intervals i and j when shifted by lag k = j-i is:
    # max(0, w - |k|*w) = w * max(0, 1 - |k|)
    
    # (f*f) at lag k (k from -(N-1) to N-1):
    # R[k] = w * sum_{i} h[i] * h[i+k] * (overlap factor)
    # For step functions on equal intervals, the autoconvolution at discrete lags:
    # (f*f)[k] = w * sum_i h[i] * h[i+k] for k >= 0, with appropriate range
    
    # Actually for step functions, the autoconvolution is:
    # (f*f)(x) for x in [kw, (k+1)w] (where x is in [-1/2, 1/2])
    # Let's use the discrete autocorrelation scaled by w
    
    # Discrete autocorrelation: R[k] = sum_{i=0}^{N-1-|k|} h[i] * h[i+|k|]
    # The continuous autoconvolution L1 norm ≈ w * sum_k |R[k]| * (some factor)
    
    # For simplicity, use: bound = w * sum_{k=-(N-1)}^{N-1} |R[k]|
    # where R[k] = sum_i h[i]*h[i+k]
    
    R = np.correlate(h, h, mode='full')
    # R has length 2N-1, R[k+N-1] = sum_i h[i]*h[i+k] for k from -(N-1) to N-1
    return w * np.sum(np.abs(R))


def _try_pattern(pattern_func):
    """Try a sign pattern and return its bound value."""
    h = pattern_func()
    if len(h) != N:
        return float('inf')
    return _autoconv_l1_bound(h)


def construct():
    global _deadline
    
    candidates = []
    
    # Pattern 1: All ones (seed)
    candidates.append(np.ones(N))
    
    # Pattern 2: Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Pattern 3: Block pattern: first half +1, second half -1
    h3 = np.ones(N)
    h3[N//2:] = -1.0
    candidates.append(h3)
    
    # Pattern 4: Quarter blocks: +1, -1, +1, -1 in quarters
    h4 = np.ones(N)
    h4[N//4:2*N//4] = -1.0
    h4[3*N//4:] = -1.0  # Actually let's do + - + -
    h4 = np.array([1.0 if (i // (N//4)) % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(h4)
    
    # Pattern 5: Sine-based sign
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    candidates.append(np.sign(np.sin(2 * np.pi * 2 * x)))  # 2 cycles
    candidates[-1][np.abs(candidates[-1]) < 0.5] = 1.0  # replace zeros with 1
    
    # Pattern 6: Pseudo-random signs (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Pattern 7: Sine with 1 cycle
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h7 = np.sign(np.sin(2 * np.pi * x))
    h7[np.abs(h7) < 0.5] = 1.0
    candidates.append(h7)
    
    # Pattern 8: Sine with 3 cycles
    h8 = np.sign(np.sin(2 * np.pi * 3 * x))
    h8[np.abs(h8) < 0.5] = 1.0
    candidates.append(h8)
    
    # Pattern 9: Balanced random with fixed seed
    rng2 = np.random.default_rng(123)
    h9 = rng2.choice([-1.0, 1.0], size=N)
    # Ensure balance
    n_pos = np.sum(h9 > 0)
    if n_pos != N // 2:
        idx = np.where(h9 > 0)[0]
        if n_pos > N // 2:
            h9[idx[n_pos - N//2]] = -1.0
        else:
            idx_neg = np.where(h9 < 0)[0]
            h9[idx_neg[N//2 - n_pos]] = 1.0
    candidates.append(h9)
    
    # Pattern 10: m-sequence-like (use a simple LFSR-based pattern)
    # For N=400, use a repeating short m-sequence
    # m-sequence of length 31: 1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1
    # Convert to signs: 1->+1, 0->-1
    m31 = np.array([1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1])
    signs_m31 = np.where(m31 == 1, 1.0, -1.0)
    h10 = np.tile(signs_m31, N // 31 + 1)[:N]
    candidates.append(h10)
    
    # Evaluate all candidates
    best_h = candidates[0]
    best_score = float('inf')
    
    for h in candidates:
        if time.time() > _deadline:
            break
        try:
            s = _autoconv_l1_bound(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
