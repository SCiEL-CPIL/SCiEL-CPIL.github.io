import numpy as np
import time

N = 400

def _get_cost(h):
    """Compute the upper bound on C3: max(|autocorr_sidelobe|)"""
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)[:N]
    if N > 1:
        return np.max(np.abs(C[1:]))
    return 0.0

def _fast_cost(h):
    """Optimized cost computation"""
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)[:N]
    return np.max(np.abs(C[1:]))

def _try_flip(h, idx):
    """Try flipping a single bit and return new cost"""
    old_val = h[idx]
    h[idx] = -old_val
    cost = _fast_cost(h)
    h[idx] = old_val
    return cost

def _try_block_flip(h, indices):
    """Try flipping multiple bits at once"""
    old_vals = h[indices].copy()
    h[indices] = -old_vals
    cost = _fast_cost(h)
    h[indices] = old_vals
    return cost

def construct():
    deadline = time.time() + 140.0  # Safety margin
    np.random.seed(42)
    
    # Initialize with alternating pattern (known to have low autocorrelation)
    h = np.ones(N)
    h[1::2] = -1.0
    
    current_cost = _fast_cost(h)
    best_h = h.copy()
    best_cost = current_cost
    
    # Simulated annealing parameters
    T_start = current_cost * 0.5
    T_end = current_cost * 0.001
    total_steps = 50000
    
    step = 0
    while step < total_steps and time.time() < deadline - 5.0:
        # Temperature schedule (exponential cooling)
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        # Choose move type: single flip or block flip
        if np.random.rand() < 0.7:
            # Single flip
            idx = np.random.randint(N)
            old_val = h[idx]
            h[idx] = -old_val
            new_cost = _fast_cost(h)
            delta = new_cost - current_cost
        else:
            # Block flip (2-4 bits)
            block_size = np.random.randint(2, 5)
            indices = np.random.choice(N, size=block_size, replace=False)
            old_vals = h[indices].copy()
            h[indices] = -old_vals
            new_cost = _fast_cost(h)
            delta = new_cost - current_cost
        
        # Accept or reject
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            # Revert
            if np.random.rand() < 0.7:
                h[idx] = old_val
            else:
                h[indices] = old_vals
        
        step += 1
    
    # Final local polish: greedy single-bit flips
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for _ in range(100):
            idx = np.random.randint(N)
            old_val = best_h[idx]
            best_h[idx] = -old_val
            new_cost = _fast_cost(best_h)
            if new_cost < best_cost:
                best_cost = new_cost
                improved = True
            else:
                best_h[idx] = old_val
    
    return best_h
