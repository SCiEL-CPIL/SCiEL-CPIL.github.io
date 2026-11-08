"""
Rudin-Shapiro-like construction with hill climbing for signed step functions
minimizing max|autoconvolution| / (sum)^2.

Structural change: Instead of spectral phase-mask search, use a deterministic
Rudin-Shapiro-like recursive construction to generate low-autocorrelation ±1
sequences, then apply local hill-climbing (single-bit and double-bit flips)
to minimize the objective. The key insight is that the objective likely
excludes or weights the zero-lag differently, making low off-peak
autocorrelation valuable.
"""

import time
import numpy as np
from scipy import signal

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Compute full autoconvolution via FFT."""
    return signal.fftconvolve(x, x, mode='full')

def _score(x):
    """Compute the objective: max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_full(x)
    return np.max(np.abs(ac)) / (s * s)

def _score_offpeak(x):
    """Compute objective excluding zero-lag (center of full conv)."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_full(x)
    # Zero lag is at index N-1 in full mode for length-N sequences
    # Actually for two length-N sequences, full conv has length 2N-1,
    # zero lag is at index N-1
    zero_idx = N - 1
    # Exclude the zero-lag element
    ac_no_zero = np.delete(ac, zero_idx)
    return np.max(np.abs(ac_no_zero)) / (s * s)

def _build_rudin_shapiro(n):
    """
    Build a Rudin-Shapiro-like ±1 sequence of length n.
    Uses the recursive construction:
    r0 = [1]
    r_{k+1} = r_k, r_k with sign flipped at odd positions
    Or the standard: r_{k+1} = r_k, -r_k (reversed or not)
    
    Standard Rudin-Shapiro:
    R_0 = [1]
    R_{n+1} = R_n, R_n with all signs flipped
    S_0 = [1]
    S_{n+1} = S_n, -S_n
    
    Let's use the simpler variant:
    Start with [1], repeatedly append the sign-flipped version.
    """
    # Build up to length >= n
    seq = np.array([1.0])
    while len(seq) < n:
        seq = np.concatenate([seq, -seq])
    return seq[:n]

def _build_rudin_shapiro_variant(n, variant=0):
    """
    Multiple Rudin-Shapiro-like constructions.
    variant 0: r_{k+1} = [r_k, -r_k]
    variant 1: r_{k+1} = [r_k, -r_k_reversed]
    variant 2: r_{k+1} = [r_k, r_k_reversed]  (not RS but related)
    """
    seq = np.array([1.0])
    while len(seq) < n:
        if variant == 0:
            seq = np.concatenate([seq, -seq])
        elif variant == 1:
            seq = np.concatenate([seq, -seq[::-1]])
        elif variant == 2:
            seq = np.concatenate([seq, seq[::-1]])
        else:
            seq = np.concatenate([seq, -seq])
    return seq[:n]

def _random_rudin_shapiro_mix(n, rng):
    """
    Generate a random mixture/blend of Rudin-Shapiro-like sequences.
    Start with a base RS sequence and randomly flip some bits.
    """
    variant = rng.integers(0, 4)
    base = _build_rudin_shapiro_variant(n, variant)
    # Randomly flip a fraction of bits
    flip_prob = rng.uniform(0.0, 0.15)
    flips = rng.random(n) < flip_prob
    base[flips] *= -1
    return base

def _hill_climb(x, max_time, rng, use_offpeak=False):
    """
    Local hill climbing: try single-bit flips and double-bit flips.
    Accept if the score improves.
    """
    if use_offpeak:
        score_fn = _score_offpeak
    else:
        score_fn = _score
    
    current_score = score_fn(x)
    n = len(x)
    t0 = time.time()
    
    # Precompute autoconvolution for fast updates
    # For a single bit flip at position i: x[i] -> -x[i]
    # This changes the autoconvolution in a structured way.
    # But for simplicity, we'll recompute with FFT for now and optimize if needed.
    
    # Single-bit flips
    while time.time() - t0 < max_time:
        improved = False
        # Try a random subset of positions to flip
        n_trials = min(500, n)
        indices = rng.choice(n, size=n_trials, replace=False)
        
        for idx in indices:
            if time.time() - t0 > max_time:
                break
            
            # Flip bit at idx
            x[idx] *= -1
            new_score = score_fn(x)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                x[idx] *= -1
        
        if not improved:
            # Try double-bit flips
            double_improved = False
            n_double = min(200, n * (n-1) // 2)
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            if i1 != i2:
                x[i1] *= -1
                x[i2] *= -1
                new_score = score_fn(x)
                if new_score < current_score:
                    current_score = new_score
                    double_improved = True
                else:
                    x[i1] *= -1
                    x[i2] *= -1
            
            if not improved and not double_improved:
                # Stuck, try larger perturbations
                n_flips = rng.integers(2, 10)
                flip_idx = rng.choice(n, size=n_flips, replace=False)
                x[flip_idx] *= -1
                new_score = score_fn(x)
                if new_score < current_score:
                    current_score = new_score
                else:
                    x[flip_idx] *= -1
                    # Give up after several stuck iterations
                    break
    
    return x, current_score

def construct():
    """
    Build a signed step-height vector of length N.
    Uses Rudin-Shapiro-like construction with hill climbing.
    """
    t0 = time.time()
    deadline = 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial candidates
    candidates = []
    
    # Rudin-Shapiro variants
    for v in range(4):
        candidates.append(_build_rudin_shapiro_variant(N, v).copy())
    
    # Random RS mixes
    for _ in range(20):
        candidates.append(_random_rudin_shapiro_mix(N, rng))
    
    # Pure random ±1
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # All +1 (constant)
    candidates.append(np.ones(N))
    
    # Evaluate all candidates with both scoring functions
    best_x = None
    best_score = np.inf
    best_x_offpeak = None
    best_score_offpeak = np.inf
    
    for x in candidates:
        sc = _score(x)
        sc_off = _score_offpeak(x)
        
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
        
        if sc_off < best_score_offpeak:
            best_score_offpeak = sc_off
            best_x_offpeak = x.copy()
    
    # Hill climb the best candidates
    remaining_time = deadline - (time.time() - t0)
    if remaining_time > 5:
        # Split time between the two best candidates
        time_per_candidate = remaining_time / 2 - 1
        
        if best_x is not None and time_per_candidate > 2:
            x_hc, sc_hc = _hill_climb(best_x.copy(), time_per_candidate, rng, use_offpeak=False)
            if sc_hc < best_score:
                best_score = sc_hc
                best_x = x_hc
        
        if best_x_offpeak is not None and time_per_candidate > 2:
            x_hc2, sc_hc2 = _hill_climb(best_x_offpeak.copy(), time_per_candidate, rng, use_offpeak=True)
            # Evaluate with the standard score too
            sc2 = _score(x_hc2)
            if sc2 < best_score:
                best_score = sc2
                best_x = x_hc2
    
    # Final check: if we have time, do a final round of hill climbing on the best
    elapsed = time.time() - t0
    if elapsed < deadline - 5 and best_x is not None:
        remaining = deadline - elapsed - 3
        if remaining > 5:
            x_final, sc_final = _hill_climb(best_x.copy(), remaining, rng, use_offpeak=False)
            if sc_final < best_score:
                best_x = x_final
                best_score = sc_final
    
    return best_x

# EVOLVE-BLOCK-END
