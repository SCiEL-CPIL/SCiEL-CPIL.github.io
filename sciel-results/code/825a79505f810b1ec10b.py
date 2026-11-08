"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def construct():
    """
    Construct a binary sequence of length N with exactly N/2 ones.
    We minimize the maximum circular cross-correlation between h and 1-h.
    
    The overlap at shift k is: sum_i h[i] * (1-h[(i+k) % N])
    = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    = N/2 - autocorr[k]
    
    where autocorr[k] = sum_i h[i]*h[(i+k)%N]
    
    So maximizing the minimum autocorr[k] for k != 0 minimizes the max overlap.
    Equivalently, we want to maximize min_{k!=0} autocorr[k].
    
    Since sum h = N/2 = 50, and h is binary, autocorr[0] = 50.
    For k != 0, we want autocorr[k] to be as large as possible (to minimize overlap).
    
    Actually, let's reconsider. The overlap is:
    overlap[k] = sum_i h[i] * (1 - h[(i+k)%N])
    
    For binary h with sum = N/2:
    overlap[k] = sum_i h[i] - sum_i h[i] h[(i+k)%N]
               = 50 - autocorr[k]
    
    So we want to maximize min_{k=1..N-1} autocorr[k].
    
    The average of autocorr[k] over k=1..N-1 is (sum h)^2 / N - sum h / N ... 
    Actually, sum_{k=0}^{N-1} autocorr[k] = (sum h)^2 = 2500
    So sum_{k=1}^{N-1} autocorr[k] = 2500 - 50 = 2450
    Average over k=1..N-1: 2450/99 ≈ 24.747
    
    So the minimum autocorr[k] is at most ~24.75 on average, but we want to maximize the min.
    
    The maximum possible min autocorr is bounded by the average, so ~24.75.
    This gives min overlap = 50 - 24.75 ≈ 25.25.
    
    But wait, the score is the upper bound on C5, which I believe is the maximum overlap.
    Let me re-read: "minimize the maximum overlap of h with shifted 1-h"
    
    So we want to minimize max_{k != 0} overlap[k] = max_{k != 0} (50 - autocorr[k])
    = 50 - min_{k != 0} autocorr[k]
    
    So we want to maximize min_{k != 0} autocorr[k].
    
    Let's use a local search approach starting from a random balanced binary sequence.
    """
    np.random.seed(42)
    
    # Start with a random balanced binary sequence
    h = np.zeros(N)
    ones = np.random.choice(N, N // 2, replace=False)
    h[ones] = 1.0
    
    def compute_autocorr(h):
        """Compute circular autocorrelation for all shifts."""
        autocorr = np.zeros(N)
        for k in range(N):
            autocorr[k] = np.dot(h, np.roll(h, k))
        return autocorr
    
    def objective(h):
        """Return the maximum overlap, which is what we want to minimize."""
        autocorr = compute_autocorr(h)
        # Overlap at shift k is 50 - autocorr[k] for k != 0
        # We want to minimize max_{k != 0} (50 - autocorr[k])
        # = 50 - min_{k != 0} autocorr[k]
        min_autocorr = np.min(autocorr[1:])
        return 50 - min_autocorr
    
    def try_swap(h, i, j):
        """Try swapping h[i] and h[j] if it improves the objective."""
        if h[i] == h[j]:
            return h, False
        
        h_new = h.copy()
        h_new[i], h_new[j] = h_new[j], h_new[i]
        
        # Check if still balanced (it should be since we're swapping)
        if np.sum(h_new) != N // 2:
            return h, False
        
        obj_old = objective(h)
        obj_new = objective(h_new)
        
        if obj_new < obj_old:
            return h_new, True
        return h, False
    
    # Local search with random restarts
    best_h = h.copy()
    best_obj = objective(h)
    
    for restart in range(20):
        if restart > 0:
            # New random start
            h = np.zeros(N)
            ones = np.random.choice(N, N // 2, replace=False)
            h[ones] = 1.0
        
        improved = True
        while improved:
            improved = False
            # Try all pairs of indices with different values
            for i in range(N):
                for j in range(i + 1, N):
                    if h[i] != h[j]:
                        h_new, did_improve = try_swap(h, i, j)
                        if did_improve:
                            h = h_new
                            improved = True
                            if objective(h) < best_obj:
                                best_obj = objective(h)
                                best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
