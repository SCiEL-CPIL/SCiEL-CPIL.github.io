"""Combined program: vectorized sigma sieve + collision grouping."""
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Known seed solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: 
    # 1. Use a vectorized sieve to compute sigma(k) for k = 1..K
    # 2. Compute M_k = k * sigma(k) for all k
    # 3. Group by M_k value and find the M with most preimages
    # 4. Verify k <= sqrt(M) constraint (should hold for k > 1 since sigma(k) >= k+1)
    
    # We'll try to push K as high as possible within the time budget.
    # The sieve is O(K log K), grouping is O(K).
    # Memory: sigma array of size K+1 (int64), M_vals of size K (int64)
    
    # Try a sequence of increasing K values, stopping when we run out of time
    # or when we've used too much memory
    
    K_values = [5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
    
    for K in K_values:
        elapsed = time.time() - start_time
        if elapsed > deadline - 25:  # Need at least 25s for sieve + grouping
            break
        
        try:
            # Sieve for sigma using NumPy
            sigma = np.zeros(K + 1, dtype=np.int64)
            for j in range(1, K + 1):
                if time.time() - start_time > deadline - 15:
                    break
                sigma[j::j] += j
            
            # Check if we completed the sieve
            if time.time() - start_time > deadline - 15:
                break
            
            # Compute M_k = k * sigma[k]
            k_vals = np.arange(1, K + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]
            
            # Filter: M <= 10^18
            valid_mask = M_vals <= 10**18
            M_valid = M_vals[valid_mask]
            k_valid = k_vals[valid_mask]
            
            # Group k by M value
            # Sort by M, then count consecutive equal values
            # This is more memory-efficient than a dict for large K
            order = np.argsort(M_valid, kind='stable')
            M_sorted = M_valid[order]
            k_sorted = k_valid[order]
            
            # Find groups of equal M values
            # Use np.diff to find where M changes
            if len(M_sorted) > 0:
                diffs = np.diff(M_sorted)
                # Positions where M changes (or start)
                change_points = np.where(diffs != 0)[0] + 1
                start_points = np.concatenate(([0], change_points))
                end_points = np.concatenate((change_points, [len(M_sorted)]))
                
                # For each group, count size and check if it beats best
                for i in range(len(start_points)):
                    s = start_points[i]
                    e = end_points[i]
                    count = e - s
                    if count > best_count:
                        M_val = int(M_sorted[s])
                        # Extract the k values in this group
                        ks = [int(k_sorted[j]) for j in range(s, e)]
                        # Verify k <= sqrt(M)
                        sqrt_M = math.isqrt(M_val)
                        valid_ks = [k for k in ks if k <= sqrt_M]
                        if len(valid_ks) > best_count:
                            best_count = len(valid_ks)
                            best_M = M_val
                            best_k = valid_ks
                
        except MemoryError:
            break
        except Exception:
            break
    
    # Final return
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
