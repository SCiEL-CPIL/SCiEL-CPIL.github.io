import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit, 10s buffer)
    
    # Initialize with known seed
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    def update_best(M, ks):
        nonlocal best_M, best_ks, best_count
        if len(ks) > best_count:
            best_M = M
            best_ks = ks
            best_count = len(ks)
    
    def search_up_to(K):
        """
        Computes S(k) = k * sigma(k) for k in [1, K] using NumPy.
        Returns the elapsed time for this specific call.
        """
        local_start = time.time()
        
        # 1. Sieve for sigma using NumPy
        sigma = np.zeros(K + 1, dtype=np.int64)
        for i in range(1, K + 1):
            sigma[i::i] += i
        
        # 2. Compute k * sigma(k)
        k_vals = np.arange(1, K + 1, dtype=np.int64)
        products = k_vals * sigma[1:]
        
        # Filter: only keep products <= 10^18
        mask = products <= 10**18
        products_filtered = products[mask]
        k_filtered = k_vals[mask]
        
        # 3. Group by value using argsort
        if len(products_filtered) == 0:
            return time.time() - local_start
        
        # Sort products and find groups
        sorted_indices = np.argsort(products_filtered)
        sorted_products = products_filtered[sorted_indices]
        sorted_k = k_filtered[sorted_indices]
        
        # Find where products change
        diff = np.diff(sorted_products)
        change_points = np.where(diff != 0)[0] + 1
        boundaries = np.concatenate([[0], change_points, [len(sorted_products)]])
        
        # Update best for groups with > 1 element
        for idx in range(len(boundaries) - 1):
            start_idx = boundaries[idx]
            end_idx = boundaries[idx + 1]
            if end_idx - start_idx > 1:
                M = int(sorted_products[start_idx])
                ks = sorted_k[start_idx:end_idx].tolist()
                # Convert to Python ints
                ks = [int(k) for k in ks]
                update_best(M, ks)
        
        return time.time() - local_start
    
    # Strategy: Try increasing K values with NumPy
    # NumPy should be ~10-50x faster than pure Python for this sieve
    
    target_Ks = [5000000, 10000000, 20000000, 50000000, 100000000]
    
    # Benchmark with a small K first
    t_bench = search_up_to(1000000)
    elapsed = time.time() - start_time
    print(f"Benchmark 10^6 took {t_bench:.2f}s")
    
    def estimate_time(K):
        if K <= 0: return 0
        base_K = 1000000
        # NumPy sieve is roughly O(K log K) but with much lower constant
        # Empirically, for NumPy, time scales roughly as K * log(K)
        base_log = math.log(base_K)
        target_log = math.log(K)
        return t_bench * (K * target_log) / (base_K * base_log)
    
    # Find the largest K such that estimate_time(K) + elapsed < deadline
    chosen_K = 1000000
    for K in target_Ks:
        if K <= 1000000: continue
        est = estimate_time(K)
        if est + elapsed < deadline - 15:  # 15s buffer
            chosen_K = K
        else:
            break
            
    print(f"Chosen K: {chosen_K}")
    
    if chosen_K > 1000000:
        t_search = search_up_to(chosen_K)
        elapsed = time.time() - start_time
        print(f"Search {chosen_K} took {t_search:.2f}s. Total elapsed: {elapsed:.2f}s")
    
    # Check if we have time for an even bigger one
    remaining = deadline - (time.time() - start_time)
    if remaining > 30:
        # Try to push further
        next_K = chosen_K * 2
        if next_K <= 200000000:
            est = estimate_time(next_K)
            if est < remaining - 10:
                print(f"Trying larger K: {next_K}")
                t_search = search_up_to(next_K)
                elapsed = time.time() - start_time
                print(f"Search {next_K} took {t_search:.2f}s. Total elapsed: {elapsed:.2f}s")
    
    # Ensure we return valid Python types
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
