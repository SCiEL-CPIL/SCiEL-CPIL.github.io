"""
Optimized signed step function for third autocorrelation inequality.

We seek a signed step function h with N=400 equal intervals on [-1/4, 1/4]
that minimizes an upper bound on C3 derived from the autoconvolution.

The metric is the peak sidelobe of the autocorrelation normalized by N:
  R[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k] for k = 0, ..., N-1
  score = max_{k=1}^{N-1} |R[k]| / N

Strategy:
1. Use structured initializations (random, alternating, block patterns).
2. Apply simulated annealing with multiple restarts to escape local minima.
3. Use FFT-based autocorrelation for efficiency.
4. Time-bounded search with safety margin before the 150s limit.
"""

import numpy as np
import time

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

def _simulated_annealing(h, start_temp, end_temp, cooling_rate, max_iter, deadline):
    """
    Run simulated annealing on the given sequence h.
    Returns (best_h, best_score).
    """
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    temp = start_temp
    no_improve = 0
    max_no_improve = 200
    
    while temp > end_temp and time.time() < deadline:
        # Pick a random position to flip
        idx = np.random.randint(N)
        current_h[idx] *= -1.0
        new_score = _score(current_h)
        
        delta = new_score - current_score
        
        if delta < 0:
            # Accept the move
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept with probability exp(-delta / temp)
            if np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_score = new_score
                no_improve += 1
            else:
                # Revert
                current_h[idx] *= -1.0
                no_improve += 1
        
        temp *= cooling_rate
        
        if no_improve >= max_no_improve:
            break
        
        if max_iter is not None and max_iter > 0:
            max_iter -= 1
            if max_iter <= 0:
                break
    
    return best_h, best_score

def construct():
    """
    Construct a signed step function with 400 equal intervals to minimize
    the autoconvolution upper bound (peak sidelobe / N).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Multiple diverse restarts with simulated annealing
    num_restarts = 40
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Diverse initialization strategies
        if restart % 4 == 0:
            # Random binary
            h = np.random.choice([-1.0, 1.0], size=N)
        elif restart % 4 == 1:
            # Alternating pattern with random perturbations
            h = np.ones(N)
            h[::2] = -1.0
            flip_mask = np.random.random(N) < 0.2
            h[flip_mask] *= -1.0
        elif restart % 4 == 2:
            # Block-based pattern
            block_size = N // 8
            h = np.zeros(N)
            for i in range(8):
                h[i*block_size:(i+1)*block_size] = 1.0 if i % 2 == 0 else -1.0
            flip_mask = np.random.random(N) < 0.1
            h[flip_mask] *= -1.0
        else:
            # Random with low frequency bias
            h = np.random.choice([-1.0, 1.0], size=N)
            # Add a low-frequency component
            freq = np.random.randint(1, N // 10)
            phase = np.random.uniform(0, 2 * np.pi)
            low_freq = np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
            h += 0.5 * low_freq
            h = np.sign(h)
            h[h == 0] = 1.0
        
        current_score = _score(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Run simulated annealing from this initialization
        sa_start_temp = 0.5
        sa_end_temp = 0.001
        sa_cooling_rate = 0.995
        
        sa_h, sa_score = _simulated_annealing(
            h, sa_start_temp, sa_end_temp, sa_cooling_rate,
            max_iter=5000, deadline=deadline - 10.0
        )
        
        if sa_score < best_score:
            best_score = sa_score
            best_h = sa_h.copy()
        
        # If we're doing well, do a second pass with tighter parameters
        if best_score < 0.05 and time.time() < deadline - 20.0:
            sa_start_temp2 = 0.2
            sa_end_temp2 = 0.0005
            sa_cooling_rate2 = 0.997
            sa_h2, sa_score2 = _simulated_annealing(
                best_h, sa_start_temp2, sa_end_temp2, sa_cooling_rate2,
                max_iter=3000, deadline=deadline - 10.0
            )
            if sa_score2 < best_score:
                best_score = sa_score2
                best_h = sa_h2.copy()
    
    if best_h is None:
        # Fallback to seed if no better found
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
