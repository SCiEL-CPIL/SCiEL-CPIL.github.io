import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin out of 120s
    
    # We want to find M with max number of k such that k*sigma(k) = M
    # and k <= sqrt(M) (which is always true for k>1, and true for k=1)
    # and M <= 10^18
    
    # Strategy: Sieve sigma(k) up to K_max, compute M_k, group by M_k.
    # Start with a reasonable K_max and increase if time allows.
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Try increasing K_max
    # Memory for sigma: K_max * 8 bytes. 10^8 * 8 = 800MB. Acceptable.
    # Time for sieve: sum K/j ~ K log K.
    
    K_max = 10_000_000  # 10 million
    
    while True:
        # Check if we have time for this K_max
        # Rough estimate: sieve takes ~ K_max * 0.0000005 s? 
        # Let's just run it and check time after.
        
        try:
            sigma = np.zeros(K_max + 1, dtype=np.int64)
            
            # Sieve for sigma
            # Optimization: only iterate j up to K_max
            # Using slice assignment
            for j in range(1, K_max + 1):
                if time.time() > deadline - 10:
                    break
                sigma[j::j] += j
            
            # Compute M_k = k * sigma[k]
            # k is 1..K_max
            k_vals = np.arange(1, K_max + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]
            
            # Filter M_vals <= 10^18
            valid_mask = M_vals <= 10**18
            M_valid = M_vals[valid_mask]
            k_valid = k_vals[valid_mask]
            
            # Group k by M
            # Since M can be large, we use a dictionary
            # To speed up, we can sort and count, but dictionary is fine for 10^7 entries
            # However, iterating 10^7 items in Python loop is slow.
            # Let's use numpy unique with return_inverse or just a dict.
            
            # Using a dict is O(N). 10^7 iterations ~ 1s.
            mapping = {}
            for m, k in zip(M_valid, k_valid):
                m_int = int(m)
                k_int = int(k)
                if m_int in mapping:
                    mapping[m_int].append(k_int)
                else:
                    mapping[m_int] = [k_int]
                
                if time.time() > deadline - 5:
                    break
            
            # Find best
            for m, ks in mapping.items():
                if len(ks) > best_count:
                    best_count = len(ks)
                    best_M = m
                    best_k = ks
            
            # If we didn't break early, we can try larger K_max
            if time.time() > deadline - 20:
                break
            
            # Increase K_max
            # Doubling might be too aggressive. Let's increase by 50%
            K_max = int(K_max * 1.5)
            if K_max > 50_000_000: # Cap at 50M to avoid memory issues
                break
                
        except Exception:
            break
            
    return {'M': int(best_M), 'k': best_k}
