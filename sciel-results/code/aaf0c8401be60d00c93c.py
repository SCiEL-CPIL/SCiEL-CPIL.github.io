"""Hybrid spectral initialization and structured local search for signed autocorrelation minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1) using FFT."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def _compute_score(x):
    """Compute the C3' metric: max |autoconvolution| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_full(x)
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _construct_signal_from_params(params):
    """
    Construct a binary signal from spectral parameters.
    params: array of shape (3*M,) -> [A1, f1, p1, A2, f2, p2, ...]
    """
    M = len(params) // 3
    if M == 0:
        return np.ones(N)
    
    t = np.arange(N)
    signal = np.zeros(N)
    
    for k in range(M):
        A = params[3*k]
        f = params[3*k + 1]
        p = params[3*k + 2]
        signal += A * np.sin(2 * np.pi * f * t + p)
        
    # Threshold to binary +/- 1
    x = np.where(signal >= 0, 1.0, -1.0)
    return x

def _spectral_init():
    """
    Generate a good initial signal using a spectral approach.
    We try to find a set of sinusoidal components that, when thresholded,
    produce a signal with low autoconvolution peaks.
    This is a quick heuristic initialization, not a full DE run.
    """
    best_x = None
    best_score = float('inf')
    rng = np.random.RandomState(42)
    
    # Try a few random spectral configurations
    for _ in range(20):
        M = 10
        params = np.zeros(3 * M)
        for k in range(M):
            params[3*k] = rng.uniform(0.5, 1.0)
            params[3*k+1] = rng.uniform(0.01, 0.5)
            params[3*k+2] = rng.uniform(0, 2*np.pi)
        
        x = _construct_signal_from_params(params)
        score = _compute_score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    return best_x, best_score

def _single_flip_refine(current, current_score, deadline):
    """
    Fast single-bit flip refinement pass.
    Uses efficient incremental updates where possible, but for simplicity
    and correctness, we recompute. Given N=400, FFT is fast.
    """
    n = len(current)
    improved = True
    passes = 0
    max_passes = 50 # Safety cap
    
    while improved and time.time() < deadline - 2 and passes < max_passes:
        improved = False
        passes += 1
        # Randomize order each pass to avoid systematic bias
        indices = np.random.permutation(n)
        
        for i in indices:
            if time.time() > deadline - 2:
                break
            
            old_val = current[i]
            current[i] = -old_val
            
            new_score = _compute_score(current)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
    return current, current_score

def _chunk_flip_search(current, current_score, deadline):
    """
    Structural change: chunk-based local search.
    Flips contiguous chunks of bits. This allows for more significant structural
    changes than single-bit flips, potentially escaping local minima.
    """
    n = len(current)
    no_improve_count = 0
    max_no_improve = 2000
    rng = np.random.RandomState(123)
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        # Choose chunk size
        r = rng.random()
        if r < 0.5:
            chunk_sz = 1
        elif r < 0.7:
            chunk_sz = rng.choice([2, 3, 4])
        elif r < 0.9:
            chunk_sz = rng.choice([5, 8, 12])
        else:
            chunk_sz = rng.choice([16, 24, 32])
            
        start = int(rng.integers(0, max(1, n - chunk_sz + 1)))
        end = start + chunk_sz
        
        # Perform the chunk flip
        old_vals = current[start:end].copy()
        current[start:end] = -old_vals
        
        new_score = _compute_score(current)
        
        # Accept or reject
        if new_score <= current_score:
            current_score = new_score
            no_improve_count = 0
        else:
            current[start:end] = old_vals
            no_improve_count += 1
            
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = float('inf')
    best_seq = None
    
    # 1. Spectral Initialization
    # Use the spectral parameterization idea to get a good starting point
    # that is already "flat-spectrum" in nature.
    x_init, score_init = _spectral_init()
    
    if x_init is not None and score_init < best_score:
        best_score = score_init
        best_seq = x_init.copy()
    else:
        # Fallback to random
        rng = np.random.RandomState(42)
        best_seq = rng.choice([-1, 1], N).astype(float)
        best_score = _compute_score(best_seq)
        
    current = best_seq.copy()
    current_score = best_score
    
    # 2. Local Refinement Phases
    # Alternate between single-bit flips and chunk flips.
    # Single-bit flips are good for fine-tuning.
    # Chunk flips are good for structural changes.
    
    # Phase 1: Single-bit refinement
    if time.time() < deadline - 10:
        current, current_score = _single_flip_refine(current, current_score, deadline)
        if current_score < best_score:
            best_score = current_score
            best_seq = current.copy()
            
    # Phase 2: Chunk flip search
    if time.time() < deadline - 10:
        current, current_score = _chunk_flip_search(current, current_score, deadline)
        if current_score < best_score:
            best_score = current_score
            best_seq = current.copy()
            
    # Phase 3: Final single-bit refinement to polish
    if time.time() < deadline - 2:
        current, current_score = _single_flip_refine(current, current_score, deadline)
        if current_score < best_score:
            best_score = current_score
            best_seq = current.copy()
            
    return best_seq
# EVOLVE-BLOCK-END
