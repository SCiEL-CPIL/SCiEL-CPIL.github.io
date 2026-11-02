"""Rewritten program: use a Sidon-like set of disjoint spikes to maximize the ratio."""
import numpy as np
from scipy.signal import fftconvolve
import time

N = 50

# EVOLVE-BLOCK-START
def _score(heights):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup(h*h))."""
    h = np.asarray(heights, dtype=float)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 == 0.0:
        return 0.0
    # Autoconvolution (discrete convolution of h with itself)
    # The problem says "autoconvolution" which for a step function on equal intervals
    # corresponds to the discrete convolution of the height sequence.
    conv = fftconvolve(h, h, mode='full')
    sup_conv = float(np.max(conv))
    if sup_conv == 0.0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin from 150s kill
    
    best_score = 0.0
    best_heights = [1.0] * N
    
    # Strategy: Use a set of disjoint spikes (nonzero at specific indices)
    # For a set S of k indices, if we put height 1 at those indices:
    # l2_sq = k, l1 = k
    # conv at lag d = number of pairs (i,j) in S with i-j = d
    # sup_conv = max over d of (number of pairs with difference d)
    # For d=0, conv[0] = k (each spike with itself)
    # So sup_conv >= k
    # ratio = k / (k * sup_conv) = 1/sup_conv
    # To maximize, we want to minimize sup_conv.
    # If we can make sup_conv = k (i.e., no two spikes have the same difference except 0),
    # then ratio = 1. But we can do better.
    
    # Actually, for a Sidon set (all pairwise sums distinct, or equivalently all pairwise 
    # differences distinct in absolute value), the number of pairs with difference d is at most 2
    # for d != 0, and k for d=0. So sup_conv = k, giving ratio = 1.
    
    # But wait: we can also vary the heights. Let's think more carefully.
    
    # Let's use a different approach: start with a known good structure and optimize.
    
    # First, try a simple Sidon-like set of spikes.
    # A Sidon set in {0, ..., N-1} with maximum size ~ sqrt(N) * constant.
    # For N=50, we can find a Sidon set of size ~7-8.
    
    # Let's find a good set of indices for spikes using a greedy Sidon construction.
    def find_sidon_set(n, max_size):
        """Find a Sidon set of up to max_size elements in {0, ..., n-1}."""
        used_sums = set()
        result = []
        for x in range(n):
            if len(result) >= max_size:
                break
            valid = True
            for y in result:
                if (x + y) in used_sums:
                    valid = False
                    break
            if valid:
                for y in result:
                    used_sums.add(x + y)
                used_sums.add(2 * x)
                result.append(x)
        return result
    
    # Try various sizes of Sidon sets
    best_overall = 0.0
    best_heights_overall = [1.0] * N
    
    for max_size in range(3, 15):
        sidon = find_sidon_set(N, max_size)
        if len(sidon) < max_size and len(sidon) < 3:
            continue
        
        heights = [0.0] * N
        for idx in sidon:
            heights[idx] = 1.0
        
        s = _score(heights)
        if s > best_overall:
            best_overall = s
            best_heights_overall = heights[:]
        
        if time.time() > deadline - 30:
            break
    
    # Now try to optimize the heights of the best structure using simple coordinate descent
    if best_overall > 0 and time.time() < deadline - 20:
        h = np.array(best_heights_overall, dtype=float)
        
        # Try to improve by adjusting heights
        for iteration in range(200):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for i in range(N):
                if h[i] == 0.0:
                    # Try turning on this spike
                    h_old = h[i]
                    h[i] = 1.0
                    s_new = _score(h)
                    if s_new > best_overall + 1e-10:
                        best_overall = s_new
                        best_heights_overall = h.copy()
                        improved = True
                    else:
                        h[i] = h_old
                else:
                    # Try adjusting height
                    h_old = h[i]
                    for factor in [0.5, 1.5, 2.0, 0.8, 1.2]:
                        h[i] = h_old * factor
                        s_new = _score(h)
                        if s_new > best_overall + 1e-10:
                            best_overall = s_new
                            best_heights_overall = h.copy()
                            improved = True
                        else:
                            h[i] = h_old
            
            if not improved:
                break
    
    # Also try a few random initializations
    if time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random sparse vector
            n_spikes = rng.randint(3, 12)
            indices = rng.choice(N, size=n_spikes, replace=False)
            h = np.zeros(N)
            for idx in indices:
                h[idx] = rng.uniform(0.5, 2.0)
            
            s = _score(h)
            if s > best_overall:
                best_overall = s
                best_heights_overall = h.copy()
    
    # Return the best heights found
    if best_overall > 0 and best_heights_overall is not None:
        return [float(x) for x in best_heights_overall]
    else:
        return [1.0] * N
# EVOLVE-BLOCK-END
