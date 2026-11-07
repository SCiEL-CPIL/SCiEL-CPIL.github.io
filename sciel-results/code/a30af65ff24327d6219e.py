"""
Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2.

Strategy: Discrete Combinatorial Assignment on Ternary Grid.
1. Fix the multiset of values to {0, 0.5, 1} with counts ensuring sum(h) = N/2.
2. Optimize the permutation of these values using greedy local search (pairwise swaps).
3. The objective is the maximum of the full correlation vector.
4. This avoids continuous non-convexity and exploits discrete structure.
"""

import numpy as np
import time

N = 100
DEADLINE = 135.0  # Safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def compute_value(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def construct():
    """
    Discrete search over permutations of a ternary multiset {0, 0.5, 1}.
    
    We need sum(h) = 50.
    Let n0, n05, n1 be the counts of 0, 0.5, 1.
    n0 + n05 + n1 = 100
    0*n0 + 0.5*n05 + 1*n1 = 50  =>  n05 + 2*n1 = 100
    
    Possible (n1, n05, n0) combinations:
    If n1=0, n05=100, n0=0. (All 0.5s. Value=1.0)
    If n1=50, n05=0, n0=50. (Binary 0/1. Sum=50.)
    If n1=25, n05=50, n0=25.
    If n1=40, n05=20, n0=40.
    
    The binary case (50 zeros, 50 ones) is often strong because it maximizes variance,
    which can help "spread out" the correlation peaks if arranged correctly (e.g., Thue-Morse like).
    The ternary case might allow finer control.
    
    We will try a few specific multisets and run greedy local search on the permutation.
    """
    
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Define candidate multisets (n1, n05, n0)
    # We want to explore the space of discrete distributions.
    candidates = [
        (50, 0, 50),   # Binary: 50 ones, 50 zeros
        (0, 100, 0),   # Uniform: 100 halfs
        (25, 50, 25),  # Ternary: 25 ones, 50 halfs, 25 zeros
        (40, 20, 40),  # Ternary: 40 ones, 20 halfs, 40 zeros
        (10, 80, 10),  # Ternary: 10 ones, 80 halfs, 10 zeros
        (30, 40, 30),  # Ternary: 30 ones, 40 halfs, 30 zeros
    ]
    
    # Pre-allocate indices for swaps to speed up correlation updates?
    # For N=100, np.correlate is very fast (~microseconds). 
    # A simple full recompute is likely faster than complex incremental updates 
    # due to overhead, especially with NumPy.
    
    for n1, n05, n0 in candidates:
        if remaining_time() < 5.0:
            break
        
        # Construct the initial multiset
        vals = np.concatenate([
            np.ones(n1),
            np.full(n05, 0.5),
            np.zeros(n0)
        ])
        
        # Initial permutation: Random shuffle
        # Using a fixed seed for reproducibility in local search, 
        # but different seeds for different candidate restarts.
        rng = np.random.default_rng(42)
        
        best_h_candidate = None
        best_val_candidate = float('inf')
        
        # Multiple random restarts for each multiset
        max_restarts = 10
        if remaining_time() < 20.0:
            max_restarts = 3
            
        for restart in range(max_restarts):
            if remaining_time() < 2.0:
                break
                
            h = vals.copy()
            rng.shuffle(h)
            current_val = compute_value(h)
            
            # Greedy Local Search: Try all pairs (i, j), if swap improves max correlation, do it.
            # To be efficient, we can iterate until no improvement is found.
            improved = True
            while improved and remaining_time() > 1.0:
                improved = False
                # Shuffle the order of checks to avoid bias?
                # Or just systematic. Systematic is deterministic and fast.
                for i in range(N - 1):
                    if remaining_time() < 1.0:
                        break
                    for j in range(i + 1, N):
                        # Try swap
                        h[i], h[j] = h[j], h[i]
                        new_val = compute_value(h)
                        
                        if new_val < current_val - 1e-12:
                            current_val = new_val
                            improved = True
                            # Do not revert. Keep the swap.
                            # We can break out of the inner loop to restart the scan?
                            # Or just continue? Continuing might find another improvement immediately.
                            # Restarting the scan ensures we don't miss interactions.
                            # For speed, let's just continue the scan.
                        else:
                            # Revert
                            h[i], h[j] = h[j], h[i]
                
                # If we improved, the landscape changed. 
                # A single pass might not be enough. The while loop handles this.
            
            if current_val < best_val_candidate:
                best_val_candidate = current_val
                best_h_candidate = h.copy()
        
        if best_h_candidate is not None and best_val_candidate < best_val:
            best_val = best_val_candidate
            best_h = best_h_candidate.copy()
            
    # Final sanity check and projection (though discrete values are already valid)
    # Ensure sum is exactly N/2 within tolerance
    # The discrete construction guarantees sum is exactly N/2.
    # h[i] in [0,1] is guaranteed by the multiset.
    
    # One final refinement: Since we are on a discrete grid, we can't do gradient descent.
    # But we can try a "kick" if time permits: random perturbation to break symmetry,
    # then greedy again.
    if remaining_time() > 10.0:
        # Try to improve the best found so far with a random kick
        h_kick = best_h.copy()
        rng_kick = np.random.default_rng(123)
        for _ in range(50): # 50 kicks
            if remaining_time() < 2.0:
                break
            # Swap 2-3 random pairs
            idx1 = rng_kick.choice(N, size=2, replace=False)
            h_kick[idx1[0]], h_kick[idx1[1]] = h_kick[idx1[1]], h_kick[idx1[0]]
            
            # Greedy refine from this kicked state
            current_val = compute_value(h_kick)
            improved = True
            while improved and remaining_time() > 1.0:
                improved = False
                for i in range(N - 1):
                    if remaining_time() < 1.0:
                        break
                    for j in range(i + 1, N):
                        h_kick[i], h_kick[j] = h_kick[j], h_kick[i]
                        new_val = compute_value(h_kick)
                        if new_val < current_val - 1e-12:
                            current_val = new_val
                            improved = True
                        else:
                            h_kick[i], h_kick[j] = h_kick[j], h_kick[i]
            
            if current_val < best_val:
                best_val = current_val
                best_h = h_kick.copy()
                
    # Return the best sequence found
    # Ensure it's a numpy array of floats
    return np.array(best_h, dtype=float)


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
