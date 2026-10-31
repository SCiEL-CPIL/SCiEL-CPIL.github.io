"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 50

def _is_golomb(ruler):
    """Check if the set of indices forms a Golomb ruler (unique differences)."""
    diffs = set()
    for i in range(len(ruler)):
        for j in range(i):
            d = ruler[i] - ruler[j]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _find_golomb_ruler(n, target_size):
    """Find a Golomb ruler of target_size in range(n)."""
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        # Try to add i
        new_ruler = ruler + [i]
        if _is_golomb(new_ruler):
            ruler = new_ruler
            if len(ruler) >= target_size:
                break
    return ruler

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # We want to maximize the ratio.
    # Theoretical max is ~100 (1/DELTA).
    # Uniform is ~2.
    # We construct a sparse Golomb ruler.
    
    best_h = None
    best_ratio = 0.0
    
    # Try different target sizes for the Golomb ruler
    # Larger size might be better if it maintains the Golomb property?
    # Actually, for Golomb, ratio is constant 1/DELTA regardless of size (if diffs unique).
    # So any valid Golomb ruler should give ~100.
    # Let's find the largest possible one to be safe, or just a valid one.
    
    # Greedy Golomb Ruler
    target_sizes = [8, 7, 6, 5]
    
    for ts in target_sizes:
        if time.time() > deadline:
            break
        ruler = _find_golomb_ruler(N, ts)
        if len(ruler) >= 4: # Minimum sanity check
            h = np.zeros(N)
            for idx in ruler:
                h[idx] = 1.0
            
            # Calculate ratio to verify
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            conv = np.convolve(h, h, mode='full')
            # The continuous max is approx max(conv) * DELTA?
            # No, the discrete conv value * DELTA is the height?
            # If h_i=1, conv[k] is count of pairs.
            # The continuous function peak is conv[k] * DELTA?
            # Actually, the convolution of two boxes of width DELTA and height 1 is a triangle of height DELTA.
            # So if conv[k]=1, peak is DELTA.
            # If conv[k]=k, peak is k*DELTA.
            # So max_val = max(conv) * DELTA.
            
            max_conv = np.max(conv)
            # DELTA is 1/(2*N) = 0.01
            DELTA = 1.0 / (2 * N)
            max_val = max_conv * DELTA
            
            if l1 > 0 and max_val > 0:
                ratio = l2_sq / (l1 * max_val)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h.copy()
    
    if best_h is None:
        # Fallback to single spike
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
