"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _m_sequence(m):
    """
    Generate an m-sequence (maximum length LFSR sequence) of length 2^m - 1.
    Uses a primitive polynomial for the LFSR.
    """
    # For m=8, use primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x1d9 in some conventions)
    # Let's use a standard one. For m=8, tap positions 8,4,3,2,1 (1-indexed)
    # Actually, let's use a simpler approach with a known primitive polynomial.
    # Primitive polynomial for m=8: x^8 + x^5 + x^3 + x^2 + 1
    # Taps at positions 8, 5, 3, 2 (0-indexed from LSB, with feedback from position 8)
    
    # Let's use m=9 to get a longer sequence, then truncate.
    # 2^9 - 1 = 511 > 400
    m = 9
    length = (1 << m) - 1  # 511
    
    # Primitive polynomial for m=9: x^9 + x^5 + x^3 + x^2 + 1
    # Tap mask: bits at positions 9, 5, 3, 2, 0 (for the recurrence)
    # LFSR state is m bits. New bit = XOR of tapped bits.
    # Let's use a standard implementation.
    
    # For m=9, one primitive polynomial is x^9 + x^5 + x^3 + x^2 + 1
    # This means the recurrence is: s_{n+9} = s_{n+4} XOR s_{n+2} XOR s_{n+1} XOR s_n
    # Taps (0-indexed from current bit): 4, 2, 1, 0
    
    # Initialize state to all 1s (non-zero)
    state = (1 << m) - 1  # 511 for m=9
    
    seq = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # Output the LSB
        seq[i] = 1 - 2 * (state & 1)  # Map 0->1, 1->-1 for {-1, +1}
        
        # Compute new bit
        # Taps at positions 4, 2, 1, 0 (relative to current state)
        # new_bit = (state >> 4) ^ (state >> 2) ^ (state >> 1) ^ state
        # But we only care about bit 0 of each tap
        new_bit = ((state >> 4) ^ (state >> 2) ^ (state >> 1) ^ state) & 1
        
        # Shift right and insert new bit at MSB
        state = ((state >> 1) | (new_bit << (m - 1))) & ((1 << m) - 1)
    
    return seq

def _compute_metric(x):
    """
    Compute the upper bound on C3 (autocorrelation-based metric).
    Uses linear autocorrelation: max_{k=1}^{N-1} |R[k]|
    where R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Use FFT to compute linear autocorrelation
    size = 1
    while size < 2 * n:
        size <<= 1
    
    X = np.fft.rfft(x, n=size)
    R_full = np.fft.irfft(X * np.conj(X), n=size)
    
    # R_full[k] for k=0..n-1 gives the linear autocorrelation R[k]
    sidelobes = R_full[1:n]
    metric = np.max(np.abs(sidelobes))
    
    return metric

def _optimize_boundary(x, max_time=80.0):
    """
    Optimize the first and last 20 intervals to minimize spectral leakage.
    Uses coordinate descent on the boundary regions.
    """
    x = x.copy()
    current_metric = _compute_metric(x)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    n = len(x)
    boundary_size = 20
    
    # Indices to optimize: first 20 and last 20
    indices = list(range(boundary_size)) + list(range(n - boundary_size, n))
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try flipping x[i]
            old_val = x[i]
            x[i] = -old_val
            new_metric = _compute_metric(x)
            
            if new_metric < current_metric:
                current_metric = new_metric
                improved = True
            else:
                x[i] = old_val
    
    return x, current_metric

def _global_hill_climb(x, max_time=30.0):
    """
    Perform global hill-climbing on the metric by flipping individual signs.
    """
    x = x.copy()
    current_metric = _compute_metric(x)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    n = len(x)
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Try flipping x[i]
            old_val = x[i]
            x[i] = -old_val
            new_metric = _compute_metric(x)
            
            if new_metric < current_metric:
                current_metric = new_metric
                improved = True
            else:
                x[i] = old_val
    
    return x, current_metric

def construct():
    start_time = time.time()
    
    # Create initial m-sequence truncated to 400 samples
    m_seq = _m_sequence(9)
    x = m_seq[:N].copy()
    
    # Compute initial metric
    initial_metric = _compute_metric(x)
    
    # First, optimize the boundary regions (first and last 20 intervals)
    # This addresses the truncation artifacts
    x, boundary_metric = _optimize_boundary(x, max_time=80.0)
    
    # Then, do a global hill-climb with remaining time
    elapsed = time.time() - start_time
    remaining = max(0, 120 - elapsed - 5)  # Leave 5s margin
    
    if remaining > 5:
        x, final_metric = _global_hill_climb(x, max_time=remaining)
    else:
        final_metric = boundary_metric
    
    return x

# EVOLVE-BLOCK-END
