import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    best_m = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    def find_best(N):
        nonlocal best_m, best_ks, best_count
        
        # Check time
        if time.time() > deadline:
            return

        try:
            sigma = np.zeros(N + 1, dtype=np.int64)
            # Sieve
            # Optimization: Use a local variable for sigma to avoid global lookup? 
            # It's already local.
            for i in range(1, N + 1):
                if time.time() > deadline:
                    return
                sigma[i::i] += i
            
            ks = np.arange(1, N + 1, dtype=np.int64)
            vals = ks * sigma
            
            # Filter M <= 10^18
            mask = vals <= 10**18
            vals = vals[mask]
            ks = ks[mask]
            
            if len(vals) == 0:
                return

            # Find max frequency
            # np.unique is efficient
            unique_vals, counts = np.unique(vals, return_counts=True)
            max_count = np.max(counts)
            
            if max_count > best_count:
                best_count = int(max_count)
                best_val_idx = np.argmax(counts)
                best_val = int(unique_vals[best_val_idx])
                
                # Get ks
                # We need to map back to original ks
                # Since we filtered, we lost the original indices.
                # We can just recompute mask for best_val?
                # Or store indices before filtering?
                
                # Better: 
                # 1. Compute vals, ks
                # 2. mask = vals <= 10^18
                # 3. vals_f = vals[mask], ks_f = ks[mask]
                # 4. unique_vals, counts, inverse = np.unique(vals_f, return_counts=True, return_inverse=True)
                # 5. max_count = max(counts)
                # 6. if max_count > best_count:
                #    target_val = unique_vals[argmax(counts)]
                #    # Get indices in vals_f where vals_f == target_val
                #    # inverse tells which unique value each element in vals_f belongs to
                #    target_unique_idx = argmax(counts)
                #    mask_target = (inverse == target_unique_idx)
                #    target_ks = ks_f[mask_target]
                
                # Let's implement this way to avoid re-computation or full sort.
                
                # Re-doing the unique with inverse
                unique_vals2, counts2, inverse = np.unique(vals, return_counts=True, return_inverse=True)
                max_count2 = np.max(counts2)
                if max_count2 > best_count:
                    best_count = int(max_count2)
                    target_idx = np.argmax(counts2)
                    target_val = int(unique_vals2[target_idx])
                    target_mask = (inverse == target_idx)
                    target_ks = ks[target_mask]
                    best_m = target_val
                    best_ks = [int(x) for x in target_ks]
                    
        except Exception:
            # If memory error or timeout, keep previous best
            pass

    # Try N=2_000_000
    find_best(2_000_000)
    
    # Try N=5_000_000
    if time.time() < deadline - 20: # Reserve 20s for finalization
        find_best(5_000_000)
        
    # Try N=10_000_000
    if time.time() < deadline - 50:
        find_best(10_000_000)

    # Final check
    # Ensure types
    best_m = int(best_m)
    best_ks = list(set(int(k) for k in best_ks))
    best_ks.sort()
    
    return {'M': best_m, 'k': best_ks}
# EVOLVE-BLOCK-END
