import numpy as np
import time

N = 100
DEADLINE = 140.0  # Safety margin for the 150s limit

def compute_max_overlap(h):
    """Compute max_k sum_i h[i] * (1 - h[(i+k)%N])"""
    b = 1.0 - h
    max_val = 0.0
    for k in range(N):
        s = np.sum(h * np.roll(b, -k))
        if s > max_val:
            max_val = s
    return max_val

def compute_min_autocorr(h):
    """Compute min_k sum_i h[i] * h[(i+k)%N]"""
    min_val = np.inf
    for k in range(N):
        s = np.sum(h * np.roll(h, -k))
        if s < min_val:
            min_val = s
    return min_val

def construct():
    start_time = time.time()
    
    # The overlap at shift k: O(k) = sum_i h[i]*(1-h[(i+k)%N]) = N/2 - sum_i h[i]*h[(i+k)%N]
    # Minimizing max_k O(k) = maximizing min_k autocorr(k) where autocorr(k) = sum_i h[i]*h[(i+k)%N]
    # For constant h=0.5: autocorr(k) = 25 for all k, so max O(k) = 50 - 25 = 25.
    
    # Strategy: Use multi-harmonic signals h[i] = 0.5 + sum_m a_m cos(2*pi*m*i/N + phi_m)
    # Optimize amplitudes and phases to maximize the minimum autocorrelation.
    
    # For a single harmonic h = 0.5 + A*cos(2*pi*i/N):
    # autocorr(k) = 25 + A^2/2 * cos(2*pi*k/N)
    # min autocorr = 25 - A^2/2, so max overlap = 50 - (25 - A^2/2) = 25 + A^2/2
    # This is WORSE than constant! The cosine increases the max overlap.
    
    # Key insight: We need the autocorrelation to be as FLAT as possible (constant 25).
    # Any deviation from constant autocorr increases the max overlap.
    # The constant sequence is actually optimal for minimizing max overlap!
    
    # Wait, let me reconsider. The problem says "minimize the maximum overlap".
    # For h=0.5: all overlaps = 25, max = 25.
    # Can we do better? We need some overlaps < 25 and none > 25.
    # But sum of all overlaps = sum_k sum_i h[i]*(1-h[(i+k)%N]) = sum_i h[i] * sum_k (1-h[(i+k)%N])
    # = sum_i h[i] * (N - sum_j h[j]) = sum_i h[i] * (N - N/2) = (N/2) * (N/2) = N^2/4 = 2500
    # Average overlap = 2500/N = 25.
    # So if max overlap < 25, then average < 25, contradiction.
    # Therefore max overlap >= 25 always, and constant 0.5 achieves exactly 25.
    
    # The constant sequence IS optimal! Score is 25.
    # The seed program's cosine scan would find that A=0 (constant) is best.
    
    # Let me verify: for any h with sum = N/2, the average overlap is exactly N/2 = 25.
    # Proof: sum_k O(k) = sum_k sum_i h[i]*(1-h[(i+k)%N])
    # Let j = (i+k)%N, so k = (j-i)%N. As k ranges 0..N-1, for fixed i, j ranges 0..N-1.
    # = sum_i h[i] * sum_j (1-h[j]) = sum_i h[i] * (N - sum_j h[j]) = (N/2) * (N/2) = N^2/4
    # Average = N/4 * N / N = N/4 * ... wait: sum_k O(k) = N^2/4, so average = N/4? No:
    # There are N shifts, so average overlap = (N^2/4)/N = N/4 = 25 for N=100.
    # Yes! Average overlap is always 25. So max overlap >= 25, and constant achieves 25.
    
    return np.full(N, 0.5)
