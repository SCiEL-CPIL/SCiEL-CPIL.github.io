"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import signal

def _compute_c3(h):
    """Compute the upper bound C3 for the autoconvolution."""
    # Autoconvolution of the step function
    # h is the sequence of signed step heights
    # The autoconvolution is h * h (convolution)
    conv = signal.fftconvolve(h, h, mode='full')
    
    # The C3 bound relates to the peak of the autoconvolution
    # normalized by the L1 norm squared
    l1_norm = np.sum(np.abs(h))
    if l1_norm == 0:
        return np.inf
    
    # The upper bound on C3 is the max absolute value of the autoconvolution
    # divided by the L1 norm squared (or similar normalization)
    # For the third autocorrelation inequality, this is typically:
    # C3 = max |autocorrelation| / (L1 norm)^2
    peak = np.max(np.abs(conv))
    return peak / (l1_norm ** 2)

def _local_search(h, max_iters=50000):
    """Greedy local search by flipping individual signs."""
    best_h = h.copy()
    best_c3 = _compute_c3(best_h)
    n = len(best_h)
    
    for _ in range(max_iters):
        # Randomly pick an index to flip
        idx = np.random.randint(n)
        new_h = best_h.copy()
        new_h[idx] = -new_h[idx]
        new_c3 = _compute_c3(new_h)
        
        if new_c3 < best_c3:
            best_h = new_h
            best_c3 = new_c3
    
    return best_h, best_c3

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = None
    best_c3 = np.inf
    
    # Multiple restarts with diverse initializations
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Choose initialization strategy
        if restart % 3 == 0:
            # Random signs
            h = np.random.choice([-1.0, 1.0], size=N)
        elif restart % 3 == 1:
            # Quasi-periodic: based on irrational frequency
            x = np.linspace(0, 1, N, endpoint=False)
            freq = 0.6180339887498949  # Golden ratio conjugate
            h = np.sign(np.sin(2 * np.pi * freq * N * x + restart * 0.1))
            h[h == 0] = 1.0
        else:
            # Block structure with random block sizes
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(1, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        
        # Quick evaluation
        c3 = _compute_c3(h)
        
        if c3 < best_c3:
            best_h = h.copy()
            best_c3 = c3
        
        # Local search if we have time
        if time.time() < deadline - 5.0:
            h_opt, c3_opt = _local_search(h, max_iters=20000)
            if c3_opt < best_c3:
                best_h = h_opt
                best_c3 = c3_opt
    
    # Final polish with more iterations if time permits
    if best_h is not None and time.time() < deadline - 2.0:
        h_polished, c3_polished = _local_search(best_h, max_iters=50000)
        if c3_polished < best_c3:
            best_h = h_polished
            best_c3 = c3_polished
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
