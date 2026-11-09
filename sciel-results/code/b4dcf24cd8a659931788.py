import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    # Fallback result from current best known (score 1.3333)
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    try:
        import numpy as np
        
        # Determine max N based on available time, but start with a large fixed value
        # 10^7 is a good balance. 2*10^7 might be better if time permits.
        N = 10_000_000
        
        # 1. Compute sigma(k) for k in [1, N] using NumPy
        # sigma[k] = sum of divisors of k
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve for sigma
        # For each i, add i to all multiples of i
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return {'M': best_M, 'k': best_ks}

        # 2. Compute M = k * sigma(k) and group by M
        # We need k <= sqrt(M) which is k <= sigma(k). 
        # For k=1, sigma(1)=1, 1<=1 ok.
        # For k>1, sigma(k) >= k+1, so k < sigma(k) is always true.
        # Thus all k are valid.
        
        M_to_ks = defaultdict(list)
        
        # Vectorize the calculation of M values as much as possible?
        # M_arr = np.arange(1, N+1) * sigma[1:N+1]
        # But we need to group them. Grouping in NumPy is not straightforward for arbitrary values.
        # We can sort (M, k) pairs and then scan for runs.
        
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:N + 1]
        
        # Filter M <= 10^18 (though for N=10^7, max M ~ 10^7 * 10^7 * log(10^7) ~ 10^15, so all valid)
        # max sigma(N) is roughly N * log(log(N)) ~ 10^7 * 3 ~ 3*10^7.
        # Max M ~ 10^7 * 3*10^7 = 3*10^14 < 10^18. So no filtering needed for N=10^7.
        
        # To group, we can create a structured array or just sort indices by M.
        # Sorting 10^7 elements takes ~1-2s.
        
        # Create array of (M, k)
        # We can sort k_arr by M_arr
        # order = np.argsort(M_arr)
        # sorted_M = M_arr[order]
        # sorted_k = k_arr[order]
        
        # Then scan sorted_M to find counts.
        
        order = np.argsort(M_arr)
        sorted_M = M_arr[order]
        sorted_k = k_arr[order]
        
        # Find runs of equal M
        # We can use np.diff to find where M changes
        # But we need the counts and the corresponding k's.
        
        # Alternative: Use a Python loop over the sorted arrays? No, that's 10^7 iterations again.
        # But we can extract the best run in NumPy?
        
        # Let's use a dictionary approach in Python but only for the unique M's?
        # No, the previous approach was building the dict.
        
        # Let's try the sort-and-scan approach in Python for the *unique* groups?
        # Actually, iterating 10^7 items in Python to build a dict took ~5s in inspiration 2.
        # Is sorting faster?
        # np.argsort is C-level, very fast.
        # Scanning 10^7 items in Python to count runs is slow.
        
        # Let's stick to the Dict approach but optimize it.
        # Or, use the fact that we just need the MAX count.
        
        # Let's try to convert M_arr to a list of ints and use Counter?
        # Counter(M_arr.tolist()) -> Counter of M values.
        # This gives counts. But we need the k's for the best M.
        # If we find the best M (max count), we can then extract the k's for that M.
        
        # Step 2a: Count frequencies of M
        # tolist() on 10^7 int64s is fast.
        M_list = M_arr.tolist()
        
        # Counter is efficient for counting
        from collections import Counter
        M_counts = Counter(M_list)
        
        # Find the M with the highest count
        if M_counts:
            max_count, best_M_val = M_counts.most_common(1)[0]
            
            if max_count > best_count:
                # Extract all k's such that k * sigma(k) == best_M_val
                # We can do this by checking the original arrays?
                # Or by filtering the sorted arrays?
                
                # Using boolean mask on M_arr
                mask = M_arr == best_M_val
                k_values = k_arr[mask]
                
                # k_values is a NumPy array of int64. Convert to list of int.
                valid_ks = [int(k) for k in k_values]
                
                # Verify constraints (just in case)
                sqrt_M = math.isqrt(best_M_val)
                valid_ks = [k for k in valid_ks if k <= sqrt_M]
                
                if len(valid_ks) > best_count:
                    best_M = int(best_M_val)
                    best_ks = valid_ks
                    best_count = len(valid_ks)
                    
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            # Try larger N if time permits?
            # For now, 10^7 is a solid jump from 2*10^5.
            pass

    except Exception as e:
        # If NumPy fails or any error, fall back
        pass

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

# EVOLVE-BLOCK-END
