import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_sigma(limit):
    """
    Precompute sigma(n) for all n <= limit using a sieve.
    Returns a numpy array of size limit+1 where index i holds sigma(i).
    """
    sigma = np.zeros(limit + 1, dtype=np.int64)
    # Initialize sigma[i] = 1 for all i (identity element for product)
    # Actually, sigma(n) = prod (1 + p + ... + p^k)
    # We can do this by iterating divisors.
    # A standard way: for i from 1 to limit, add i to all multiples of i.
    # sigma[n] = sum of divisors.
    # This is O(N log N).
    # N = 10^7. log N ~ 16. 16 * 10^7 = 1.6 * 10^8 operations.
    # In Python this is too slow. In NumPy it is vectorized.
    
    # Vectorized approach:
    # sigma = np.zeros(limit + 1, dtype=np.int64)
    # for i in range(1, limit + 1):
    #     sigma[i::i] += i
    # This loop is still in Python, but the addition is vectorized.
    # 10^7 iterations of a vectorized add might be too slow in Python loop overhead?
    # Let's try to optimize the loop or use a different method.
    
    # Alternative: Factorization based.
    # But we need sigma for ALL n to do O(1) lookup.
    
    # Let's try the divisor sum sieve.
    # To speed up, we can process in chunks or just rely on NumPy speed.
    # 10^7 is large.
    # Let's try to be smart.
    # sigma[n] = sum_{d|n} d.
    
    # Efficient NumPy sieve for sigma:
    # sigma = np.zeros(limit + 1, dtype=np.int64)
    # for i in range(1, limit + 1):
    #     sigma[i::i] += i
    
    # This takes about 10-20 seconds in Python for 10^7.
    # We have 90s budget.
    # Let's implement it and see.
    
    sigma = np.zeros(limit + 1, dtype=np.int64)
    # We can skip even numbers? No, sigma is needed for all.
    # We can use a list of integers and convert? No, memory.
    
    # Optimization: Use a Python loop but with NumPy slice updates.
    for i in range(1, limit + 1):
        sigma[i::i] += i
        
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 80 # 80s limit to be safe
    
    limit = 10**7
    
    # Precompute sigma
    sigma_arr = _precompute_sigma(limit)
    
    if time.time() > deadline:
        return [11, 6]
        
    best_a, best_b = 11, 6
    best_raw = abs(11/6 - 1.5)
    
    # We search for pairs (a, b) such that sigma(a) == sigma(b)
    # and |a/b - 1.5| is minimized.
    # |2a - 3b| / (2b)
    
    # Iterate b from limit down to 1.
    # For each b, check if a = (3b + 1)/2 is integer and <= limit.
    # This requires 3b + 1 to be even => b is odd.
    # Also check a = (3b - 1)/2.
    
    # To maximize speed, we check the "best" candidates first?
    # No, we want the smallest error.
    # Error is proportional to |2a - 3b| / b.
    # If |2a - 3b| = 1, error is 1/(2b).
    # Smaller b means larger error.
    # So we want LARGEST b with |2a - 3b| = 1.
    # If none, look for |2a - 3b| = 3, etc.
    
    # Strategy:
    # 1. Check all b for k=1 (a = (3b+1)/2) and k=-1 (a=(3b-1)/2).
    #    Since we iterate b downwards, the first hit is the best for k=1/-1.
    # 2. If no hit, check k=3, etc?
    #    But k=1 is much better than k=3 for the same b.
    #    And k=1 for large b is better than k=3 for small b?
    #    1/(2*10^6) = 5e-7.
    #    3/(2*10) = 0.15.
    #    So yes, prioritize small k, then large b.
    
    # We will iterate b from limit down to 1.
    # For each b, check k=1 and k=-1.
    # If found, return immediately.
    # This finds the largest b with k=1 or -1.
    # Is it possible that a smaller b with k=1 is better than a larger b with k=1? No, 1/b is monotonic.
    # Is it possible that a larger b with k=3 is better than a smaller b with k=1?
    # 3/(2b_large) vs 1/(2b_small).
    # If b_large > 3 * b_small, then 3/b_large < 1/b_small.
    # So we might miss a better solution if we only check k=1.
    # However, finding ANY solution is the main hurdle.
    # The current score is ~4e6, which is huge.
    # Even k=1 with b=1000 is 0.0005, which is 2e-4.
    # Score = 1 / 2e-4 = 5000.
    # This is a massive improvement over 4e6.
    
    # So, finding ANY k=1 or k=-1 solution is a huge win.
    # Let's stick to checking k=1 and k=-1 first.
    
    # We can also check k=3, 5, etc in subsequent passes if time permits.
    
    # Pass 1: k = 1 and -1
    # b must be odd for (3b +/- 1) to be even.
    
    # Iterate b downwards
    # To avoid Python loop overhead being too high for 10^7 iterations,
    # we can vectorize the check?
    # No, sigma lookup is array access.
    # sigma_arr[a] == sigma_arr[b]
    
    # Let's try to vectorize the search for k=1.
    # b is odd.
    # a = (3b + 1) / 2
    # We need a <= limit.
    # 3b + 1 <= 2*limit => b <= (2*limit - 1) / 3.
    
    # Let's create an array of valid b's?
    # b_odd = np.arange(1, limit+1, 2)
    # a_candidates = (3 * b_odd + 1) // 2
    # valid_mask = a_candidates <= limit
    # b_valid = b_odd[valid_mask]
    # a_valid = a_candidates[valid_mask]
    
    # Check sigma equality:
    # sigma_a = sigma_arr[a_valid]
    # sigma_b = sigma_arr[b_valid]
    # matches = np.where(sigma_a == sigma_b)[0]
    
    # If matches is not empty, we have candidates.
    # We want the largest b.
    # So we sort b_valid descending? Or just take the last one if we sorted ascending?
    # b_odd is ascending.
    # So the last element in matches corresponds to the largest b.
    
    # This vectorized approach is MUCH faster than a Python loop.
    
    try:
        # Generate odd b's
        b_odd = np.arange(1, limit + 1, 2, dtype=np.int64)
        
        # Calculate a for k=1: 2a = 3b + 1
        a_k1 = (3 * b_odd + 1) // 2
        
        # Filter a <= limit
        mask_k1 = a_k1 <= limit
        b_k1 = b_odd[mask_k1]
        a_k1 = a_k1[mask_k1]
        
        if len(b_k1) > 0:
            # Get sigma values
            sig_a = sigma_arr[a_k1]
            sig_b = sigma_arr[b_k1]
            
            # Find matches
            match_indices = np.where(sig_a == sig_b)[0]
            
            if len(match_indices) > 0:
                # We want the largest b, which is the last one in b_k1 (since b_odd is sorted)
                # Wait, b_k1 preserves order of b_odd.
                # So the last index in match_indices corresponds to the largest b.
                idx = match_indices[-1]
                b_val = int(b_k1[idx])
                a_val = int(a_k1[idx])
                
                # Verify
                if a_val <= limit and b_val <= limit and a_val > 0 and b_val > 0:
                    if int(sigma_arr[a_val]) == int(sigma_arr[b_val]):
                        # Check if ratio is not exactly 1.5
                        # 2a == 3b?
                        # Here 2a = 3b + 1, so not 1.5.
                        return [a_val, b_val]
                        
        # Calculate a for k=-1: 2a = 3b - 1
        # b must be odd.
        a_km1 = (3 * b_odd - 1) // 2
        
        # Filter a >= 1 and a <= limit
        # 3b - 1 >= 2 => 3b >= 3 => b >= 1.
        mask_km1 = (a_km1 >= 1) & (a_km1 <= limit)
        b_km1 = b_odd[mask_km1]
        a_km1 = a_km1[mask_km1]
        
        if len(b_km1) > 0:
            sig_a = sigma_arr[a_km1]
            sig_b = sigma_arr[b_km1]
            
            match_indices = np.where(sig_a == sig_b)[0]
            
            if len(match_indices) > 0:
                idx = match_indices[-1]
                b_val = int(b_km1[idx])
                a_val = int(a_km1[idx])
                
                if a_val <= limit and b_val <= limit and a_val > 0 and b_val > 0:
                    if int(sigma_arr[a_val]) == int(sigma_arr[b_val]):
                        # 2a = 3b - 1, not 1.5.
                        return [a_val, b_val]
                        
    except Exception as e:
        pass
        
    # If no k=1/-1 found, try k=3/-3?
    # 2a = 3b + 3 => 2a = 3(b+1).
    # a = 3(b+1)/2.
    # b+1 must be even => b is odd.
    # a = 3 * (b+1) / 2.
    
    if time.time() < deadline:
        try:
            # b is odd.
            # a = (3 * (b + 1)) // 2
            # Check bounds.
            
            # b_odd is already defined.
            a_k3 = (3 * (b_odd + 1)) // 2
            mask_k3 = a_k3 <= limit
            b_k3 = b_odd[mask_k3]
            a_k3 = a_k3[mask_k3]
            
            if len(b_k3) > 0:
                sig_a = sigma_arr[a_k3]
                sig_b = sigma_arr[b_k3]
                match_indices = np.where(sig_a == sig_b)[0]
                if len(match_indices) > 0:
                    idx = match_indices[-1]
                    b_val = int(b_k3[idx])
                    a_val = int(a_k3[idx])
                    if int(sigma_arr[a_val]) == int(sigma_arr[b_val]):
                        # 2a = 3b + 3 => 2a - 3b = 3.
                        # Not 0.
                        return [a_val, b_val]
                        
            a_km3 = (3 * (b_odd - 1)) // 2
            mask_km3 = (a_km3 >= 1) & (a_km3 <= limit)
            b_km3 = b_odd[mask_km3]
            a_km3 = a_km3[mask_km3]
            
            if len(b_km3) > 0:
                sig_a = sigma_arr[a_km3]
                sig_b = sigma_arr[b_km3]
                match_indices = np.where(sig_a == sig_b)[0]
                if len(match_indices) > 0:
                    idx = match_indices[-1]
                    b_val = int(b_km3[idx])
                    a_val = int(a_km3[idx])
                    if int(sigma_arr[a_val]) == int(sigma_arr[b_val]):
                        return [a_val, b_val]
                        
        except Exception as e:
            pass
            
    return [best_a, best_b]

# EVOLVE-BLOCK-END
