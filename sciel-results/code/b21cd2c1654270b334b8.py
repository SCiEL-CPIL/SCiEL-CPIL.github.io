"""
Optimized signed step function for third autocorrelation inequality.

The task requires choosing a signed step function h with N=400 equal intervals
on [-1/4, 1/4] to minimize an upper bound on C3. The metric is the peak
sidelobe of the autocorrelation normalized by N:
  R[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k] for k = 0, ..., N-1
  score = max_{k=1}^{N-1} |R[k]| / N

Strategy:
1. Use structured initializations (Thue-Morse, quadratic residues) to jump
   into low-autocorrelation basins.
2. Use a fast incremental delta-update for single-bit flips to speed up
   greedy local search and simulated annealing.
3. Simulated annealing with cooling to escape local minima.
4. Multiple restarts with time-bounded search.
"""

import numpy as np
import time
import math

N = 400

# EVOLVE-BLOCK-START

def _autocorrelation(h):
    """Compute the autocorrelation of h using FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    R = np.fft.irfft(H * H, n=size)[:n]
    return R

def _score(h):
    """Compute the score: peak sidelobe of autocorrelation normalized by N."""
    R = _autocorrelation(h)
    if len(R) > 1:
        peak_sidelobe = np.max(np.abs(R[1:]))
    else:
        peak_sidelobe = 0.0
    return peak_sidelobe / N

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count bits
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq

def _quadratic_residues(n, p):
    """Generate a sequence based on quadratic residues mod p, tiled to length n."""
    # For prime p, the quadratic residues give a low-autocorrelation sequence
    # We use p = next prime >= n
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    p = p
    while not is_prime(p):
        p += 1
    
    # Quadratic residues mod p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Map to ±1
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        seq[i] = 1.0 if i % p in residues else -1.0
    return seq

def _incremental_flip(h, R, idx):
    """
    Flip h[idx] and update R incrementally.
    Returns the new R (a copy is not needed if we modify in place, but we return for clarity).
    """
    # Delta R[k] = -2 * h[idx] * (h[idx+k] if idx+k < N else 0 + h[idx-k] if idx-k >= 0 else 0)
    # For k=0, the two terms are the same, so we must be careful.
    # Actually, for k=0: R[0] = sum h[j]^2 = N, invariant. So delta R[0] = 0.
    # For k > 0:
    #   if idx+k < N and idx-k >= 0: delta = -2*h[idx]*(h[idx+k]+h[idx-k])
    #   if idx+k < N and idx-k < 0: delta = -2*h[idx]*h[idx+k]
    #   if idx+k >= N and idx-k >= 0: delta = -2*h[idx]*h[idx-k]
    
    old_val = h[idx]
    h[idx] *= -1.0
    
    # Update R for k=1 to N-1
    for k in range(1, N):
        delta = 0.0
        if idx + k < N:
            delta += old_val * h[idx + k]  # old_val is the pre-flip value
        if idx - k >= 0:
            delta += old_val * h[idx - k]
        R[k] += -2.0 * delta
    
    return R

def _recompute_R(h):
    """Recompute R from scratch using FFT."""
    return _autocorrelation(h)

def construct():
    """
    Construct a signed step function with 400 equal intervals to minimize
    the autoconvolution upper bound (peak sidelobe / N).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Structured initializations
    initializers = []
    
    # Thue-Morse
    initializers.append(_thue_morse(N))
    
    # Quadratic residues with several primes
    for p in [401, 409, 421, 431, 439, 443, 449, 457, 461, 463]:
        initializers.append(_quadratic_residues(N, p))
    
    # Random sequences
    for _ in range(20):
        initializers.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Evaluate all initializers
    candidates = []
    for h0 in initializers:
        R0 = _autocorrelation(h0)
        s0 = np.max(np.abs(R0[1:])) / N
        candidates.append((s0, h0.copy(), R0.copy()))
    
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # Take top K candidates for local search
    K = 10
    
    for s0, h0, R0 in candidates[:K]:
        if time.time() > deadline - 15.0:
            break
        
        if s0 < best_score:
            best_score = s0
            best_h = h0.copy()
        
        # Simulated annealing on this candidate
        h = h0.copy()
        R = R0.copy()
        current_score = s0
        
        # Simulated annealing parameters
        T_start = 0.05
        T_end = 0.001
        cooling = 0.9995
        T = T_start
        
        while T > T_end and time.time() < deadline - 10.0:
            idx = np.random.randint(N)
            old_val = h[idx]
            
            # Compute delta R
            old_R = R.copy()
            _incremental_flip(h, R, idx)
            new_score = np.max(np.abs(R[1:])) / N
            
            delta_s = new_score - current_score
            
            if delta_s < 0 or np.random.random() < math.exp(-delta_s / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Revert
                h[idx] = old_val
                R[:] = old_R
            
            T *= cooling
        
        # Greedy local search polish
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5.0:
                    break
                old_val = h[idx]
                old_R = R.copy()
                _incremental_flip(h, R, idx)
                new_score = np.max(np.abs(R[1:])) / N
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                else:
                    h[idx] = old_val
                    R[:] = old_R
    
    if best_h is None:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
