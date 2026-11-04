"""
Strategy: Zadoff-Chu based signal design with optimized quadratic phase parameter.

Rationale:
1. Zadoff-Chu sequences have ideal auto-correlation properties (constant amplitude,
   zero auto-correlation for non-zero shifts) in the complex domain.
2. Taking the real part of a complex Zadoff-Chu-like signal with a tuned quadratic 
   chirp phase can yield a real-valued signal with very low auto-convolution peaks.
3. The key insight: for a complex signal z[n] = exp(i * (a*n^2 + b*n)), the real 
   part is cos(a*n^2 + b*n). The auto-convolution of cos(a*n^2) has a structure 
   that can be analyzed: the peak at lag 0 is N, but off-peak values are bounded 
   by ~sqrt(N) for appropriate 'a'.
4. By optimizing the quadratic phase coefficient 'a' (and possibly linear 'b'), 
   we can minimize the max |autoconv| / integral^2.
5. This is a 1-2 dimensional search problem, much easier than 400D.
6. We also try M-sequences (maximal length sequences) which have similar properties.
7. Finally, we do a local search polish on the best candidate.

The critical difference from the current program: instead of using differential 
evolution on 16 parameters to synthesize a multi-sine signal, we directly exploit 
the algebraic structure of Zadoff-Chu / chirp signals which have known optimal 
auto-correlation properties, reducing the search to 1-2 scalar parameters.
"""

import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _zadoff_chu_signal(a, b, N=N):
    """
    Generate a real-valued signal from the real part of a complex chirp:
    x[n] = cos(a * n^2 + b * n)
    
    Parameters:
    - a: quadratic phase coefficient
    - b: linear phase coefficient
    """
    n = np.arange(N)
    return np.cos(a * n * n + b * n)

def _m_sequence_signal(m, c, N=N):
    """
    Generate an M-sequence (maximal length LFSR sequence) mapped to [-1, 1].
    
    Parameters:
    - m: LFSR feedback polynomial (as integer bitmask)
    - c: initial state (as integer bitmask)
    """
    # Use a simple LFSR to generate a pseudo-random binary sequence
    # For N=400, we need a sequence of length at least 400
    # A 9-bit LFSR gives 511 bits, which is enough
    
    # Try a few known good polynomials
    # Polynomial 0x119 (x^9 + x^5 + 1) gives maximal length 511
    poly = m
    state = c
    seq = []
    for i in range(N):
        seq.append(state & 1)
        # Feedback
        bit = (state & 1) ^ ((state >> 8) & 1)  # For 9-bit LFSR
        state = (state >> 1) | (bit << 8)
    
    # Map 0/1 to -1/+1
    x = np.array(seq, dtype=float)
    x = 2.0 * x - 1.0  # Map {0,1} to {-1,+1}
    return x

def _try_zadoff_chu(deadline, rng):
    """
    Search for optimal Zadoff-Chu-like parameters (a, b).
    """
    best_val = float('inf')
    best_a, best_b = 0, 0
    
    # Grid search over 'a' (quadratic phase) and 'b' (linear phase)
    # 'a' should be small enough that the phase doesn't wrap too many times
    # but large enough to create good spreading
    
    # The auto-convolution of cos(a*n^2) has peaks at lags where 2*a*n*lag + a*lag^2 ≈ 0 mod 2*pi
    # For good spreading, we want a such that the "chirp rate" is optimal
    
    # Typical good values: a ~ pi / N to pi
    a_values = np.linspace(0.01, 2.0, 100)
    b_values = np.linspace(0, 2 * np.pi, 50)
    
    for a in a_values:
        if time.time() > deadline - 5:
            break
        for b in b_values:
            x = _zadoff_chu_signal(a, b, N)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_a, best_b = a, b
    
    # Local refinement around best
    if best_val < float('inf'):
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            a_new = best_a + rng.normal(0, 0.05)
            b_new = best_b + rng.normal(0, 0.5)
            if a_new < 0:
                a_new = abs(a_new)
            b_new = b_new % (2 * np.pi)
            x = _zadoff_chu_signal(a_new, b_new, N)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_a, best_b = a_new, b_new
    
    return _zadoff_chu_signal(best_a, best_b, N), best_val

def _try_m_sequences(deadline, rng):
    """
    Try several M-sequences and pick the best.
    """
    best_val = float('inf')
    best_x = None
    
    # Try several polynomial/state combinations
    polynomials = [0x119, 0x123, 0x131, 0x103, 0x113, 0x107]
    states = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    for poly in polynomials:
        if time.time() > deadline - 5:
            break
        for state in states:
            x = _m_sequence_signal(poly, state, N)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    
    return best_x, best_val

def _try_white_noise(deadline, rng):
    """
    Try random white noise signals.
    """
    best_val = float('inf')
    best_x = None
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        x = rng.normal(0, 1, N)
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    
    return best_x, best_val

def _polish_local(x, deadline, rng):
    """
    Local search to fine-tune the signal.
    """
    if x is None:
        return None, float('inf')
    
    best_x = x.copy()
    best_val = _metric(best_x)
    
    step_size = 0.5
    no_improve = 0
    
    for i in range(3000):
        if time.time() > deadline - 2:
            break
            
        if no_improve > 100:
            step_size *= 0.85
            if step_size < 0.01:
                break
            no_improve = 0
        
        # Pick a few indices to perturb
        k = rng.integers(1, 4)
        indices = rng.choice(N, size=k, replace=False)
        
        x_new = best_x.copy()
        for idx in indices:
            x_new[idx] += rng.normal(0, step_size)
        
        v_new = _metric(x_new)
        if v_new < best_val:
            best_val = v_new
            best_x = x_new
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 140.0  # Safety margin (evaluator kills at 150s)
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # 1. Try Zadoff-Chu based signals (primary strategy)
    x_zc, v_zc = _try_zadoff_chu(deadline - 30, rng)
    if v_zc < best_val:
        best_val = v_zc
        best_x = x_zc
    
    # 2. Try M-sequences
    x_ms, v_ms = _try_m_sequences(deadline - 20, rng)
    if v_ms < best_val:
        best_val = v_ms
        best_x = x_ms
    
    # 3. Try white noise
    x_wn, v_wn = _try_white_noise(deadline - 10, rng)
    if v_wn < best_val:
        best_val = v_wn
        best_x = x_wn
    
    # 4. Local polish on the best candidate
    if best_x is not None:
        best_x, best_val = _polish_local(best_x, deadline, rng)
    
    # 5. Fallback: all ones
    x_ones = np.ones(N)
    v_ones = _metric(x_ones)
    if v_ones < best_val:
        best_x = x_ones
        best_val = v_ones
    
    # Final safety check
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x

# EVOLVE-BLOCK-END
