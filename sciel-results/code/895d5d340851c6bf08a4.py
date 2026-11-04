"""
Strategy: Combinatorial Block-Flip Search on Binary Sequences.

The current program uses continuous local search (gradient/random perturbations) on a float vector.
This program switches to a discrete search space: x_i in {+1, -1}.
This is structurally different because:
1. The representation is binary (integers), not continuous floats.
2. The optimization is combinatorial (block flips), not gradient-based.
3. It exploits the symmetry of the problem: for binary sequences, the integral S is an integer, 
   and the autoconvolution values are integers.

We use a "Block Flip" strategy:
- Divide the sequence into blocks.
- For each block, try all possible binary configurations (or a subset) that minimize the 
  contribution to the autoconvolution peaks.
- This is analogous to the "Chirp" or "m-sequence" design but optimized for the specific 
  max-abs metric.

Since N=400, we can do a multi-resolution search:
1. Start with a random binary sequence.
2. Use a "divide and conquer" approach: split into two halves, optimize the boundary, 
   then split into quarters, etc.
   Actually, a simpler and effective method for binary sequences is "Local Block Search":
   - Pick a block of length L (e.g., 10-20).
   - Try all 2^L configurations? No, too many.
   - Instead, use a greedy flip: for each bit in the block, flip it if it improves the metric.
   - Repeat for random blocks.

However, a more powerful structural change is to use **Simulated Annealing** on the binary 
space, which allows escaping local minima more effectively than simple hill-climbing.

Let's implement:
1. A fast metric calculator for binary sequences (using FFT).
2. Simulated Annealing with a cooling schedule.
3. Multi-start: Run several SA instances with different seeds and initial temperatures.
"""

import numpy as np
import time

N = 400

# Pre-compute FFT size for speed
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _metric_binary(x):
    """
    Calculate the metric for a binary sequence x (values +1/-1).
    Metric = max(|autoconv|) / (sum(x))^2
    """
    # Sum of x
    s = np.sum(x)
    if s == 0:
        return float('inf')
    
    # Autoconvolution via FFT
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    # Max absolute value
    max_abs = np.max(np.abs(conv))
    
    return max_abs / (s * s)

def _sa_optimize(x_init, deadline, rng, T_start=10.0, T_end=0.01, alpha=0.999):
    """
    Simulated Annealing on the binary sequence.
    """
    current_x = x_init.copy()
    current_val = _metric_binary(current_x)
    best_x = current_x.copy()
    best_val = current_val
    
    T = T_start
    n_iterations = 0
    
    while time.time() < deadline and T > T_end:
        n_iterations += 1
        
        # Propose a move: flip a random subset of bits
        # Flipping 1 bit is too slow to escape deep minima. Flipping too many is random.
        # Let's flip a small random block of bits.
        block_size = rng.integers(1, 10)
        start_idx = rng.integers(0, N - block_size + 1)
        
        x_new = current_x.copy()
        x_new[start_idx:start_idx + block_size] *= -1  # Flip signs
        
        new_val = _metric_binary(x_new)
        
        # Acceptance criterion
        delta = new_val - current_val
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_x = x_new
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
        
        # Cool down
        T *= alpha
        
        # If we are stuck for a long time, reheat slightly? 
        # For now, simple cooling.
        
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Multi-start Simulated Annealing
    # We have ~145 seconds. 
    # Each SA run takes some time. Let's estimate how many we can do.
    # A single SA run with ~10k-50k iterations might take 1-5 seconds.
    # Let's try to run as many independent SA instances as possible.
    
    num_starts = 0
    while time.time() < deadline - 5.0:
        # Initialize with a random binary sequence
        # Sometimes starting with a structured sequence (like alternating) is better?
        # Let's mix: 50% random, 50% alternating
        if rng.random() < 0.5:
            x_init = rng.choice([-1, 1], size=N)
        else:
            x_init = np.ones(N, dtype=int)
            x_init[::2] = -1  # Alternating
        
        # Run SA
        # Adjust T_start based on remaining time? 
        # Let's use a fixed schedule but break early if time is up.
        x_opt, val_opt = _sa_optimize(x_init, deadline, rng, T_start=5.0, T_end=0.001, alpha=0.9995)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
        
        num_starts += 1
        # Small sleep to avoid CPU spike? No, we want to use all time.
        
    # Fallback if no valid result (should not happen)
    if best_x is None:
        best_x = np.ones(N, dtype=int)
    
    return best_x
