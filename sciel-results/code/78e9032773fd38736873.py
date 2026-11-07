import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    # Initialize with known seed
    best_M = 333312
    best_ks = [336, 372, 434]
    
    def update_best(M, ks):
        nonlocal best_M, best_ks
        # Convert numpy arrays to python lists/ints for return
        if isinstance(ks, np.ndarray):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
            
        if len(ks_list) > len(best_ks):
            best_M = int(M)
            best_ks = ks_list

    def search_with_numpy(K):
        local_start = time.time()
        
        try:
            # 1. Sieve for sigma
            sigma = np.zeros(K + 1, dtype=np.int64)
            
            # Check time periodically to avoid exceeding deadline
            check_interval = max(1000, K // 500)  # Check at most ~500 times
            
            for i in range(1, K + 1):
                if i % check_interval == 0 and time.time() > deadline - 15:
                    return 1  # Time out
                
                sigma[i::i] += i
                
            if time.time() > deadline - 15:
                return 1
                
            # 2. Compute v(k) = k * sigma(k)
            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:K+1]
            
            # 3. Filter valid M
            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return 1
                
            vals = vals[valid_mask]
            ks = ks[valid_mask]
            
            if len(vals) < 2:
                return 1
                
            # 4. Sort and group
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_ks = ks[order]
            
            # Find consecutive equal values
            diffs = np.diff(sorted_vals)
            change_points = np.where(diffs > 0)[0] + 1
            
            start_points = np.concatenate(([0], change_points))
            end_points = np.concatenate((change_points, [len(sorted_vals)]))
            
            counts = end_points - start_points
            
            if len(counts) == 0:
                return 1
                
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            
            if max_count > len(best_ks):
                s = start_points[max_idx]
                e = end_points[max_idx]
                M = int(sorted_vals[s])
                ks_group = sorted_ks[s:e]
                update_best(M, ks_group)
                
            return 0  # Success
            
        except MemoryError:
            return 2  # Memory error
        except Exception as e:
            print(f"Error in search: {e}")
            return 3  # Other error
    
    # Try K values from largest to smallest
    # Estimate: sieve for K takes roughly K * H(K) operations where H(K) is harmonic
    # For K=10M, this is about 10M * 23 ≈ 230M numpy slice ops
    # Each op is fast but there are K of them in the Python loop
    
    # Let's try several K values, starting with what we think will fit
    target_Ks = [10000000, 5000000, 2000000, 1000000]
    
    for K in target_Ks:
        if time.time() > deadline - 25:
            break
            
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"Search K={K} finished with status {status}. Total elapsed: {elapsed:.2f}s")
        
        if status == 0:
            # Success, we can try smaller K to see if we find better M
            # But smaller K means fewer k values, so unlikely to find more solutions
            # Actually, we want to maximize the count, so larger K is better
            # If we succeeded with K=10M, that's our best shot
            break
        elif status == 2:
            # Memory error, try smaller K
            continue
        else:
            # Other error or timeout, try smaller K
            continue
    
    return {'M': best_M, 'k': best_ks}
