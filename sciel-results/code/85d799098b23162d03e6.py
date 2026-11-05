"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _value(h):
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / s ** 2)

def _quadratic_residue_sequence(p, offset=0):
    """Generate a Legendre symbol sequence of length p."""
    seq = np.zeros(p, dtype=np.float64)
    for i in range(p):
        x = (i + offset) % p
        if x == 0:
            seq[i] = 0
        else:
            # Legendre symbol: 1 if x is QR mod p, -1 otherwise
            seq[i] = 1 if pow(x, p-1, p) == 1 else -1
    return seq

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Try quadratic residue based sequences
    # Closest primes to 400: 397, 401
    for p in [397, 401]:
        if time.time() - start_time > deadline - 5:
            break
        seq = _quadratic_residue_sequence(p)
        # Take first N elements or wrap
        if p >= N:
            h_candidate = seq[:N].copy()
        else:
            # Pad or repeat
            h_candidate = np.tile(seq, N // p + 1)[:N].copy()
        
        val = _value(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()
    
    # Try random binary sequences with optimization
    rng = np.random.RandomState(42)
    for trial in range(200):
        if time.time() - start_time > deadline - 5:
            break
        h_candidate = rng.choice([-1.0, 1.0], size=N)
        val = _value(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()
    
    # Try structured sequences: 1 + a*cos
    for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        if time.time() - start_time > deadline - 5:
            break
        for freq in [1, 2, 3, 5, 10]:
            i = np.arange(N)
            h_candidate = 1.0 + a * np.cos(2 * np.pi * freq * i / N)
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate.copy()
    
    # Try linear ramp
    for slope in [0.1, 0.5, 1.0, 2.0]:
        if time.time() - start_time > deadline - 5:
            break
        i = np.arange(N)
        h_candidate = 1.0 + slope * (i / N - 0.5)
        val = _value(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()
    
    return best_h
# EVOLVE-BLOCK-END
