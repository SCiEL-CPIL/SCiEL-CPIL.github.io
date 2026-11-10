import numpy as np
import time

def construct():
    """
    Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
    Returns the start index and the corresponding smallest positive square roots.
    
    Algorithm:
    1. Precompute the smallest positive root for each quadratic residue modulo p.
    2. Create a boolean indicator array for quadratic residues.
    3. Use a sliding window sum (via cumulative sum or convolution) to find the first 
       position where 5 consecutive values are residues.
    4. Extract the roots for the found start index.
    """
    p = 1009
    m = 5
    
    # Deadline guard: ensure we finish well before 60s/90s limit
    start_time = time.time()
    timeout = 50.0  # 50 seconds safety margin
    
    # 1. Precompute roots
    # We need the smallest positive root for each residue r in 1..p-1
    # root_map[r] = x such that x^2 % p == r and 1 <= x <= p-1
    # Since we iterate x from 1 to p-1, the first time we see a residue r, 
    # x is the smallest root.
    root_map = np.zeros(p, dtype=np.int32)
    is_res = np.zeros(p, dtype=np.bool_)
    
    # Vectorized computation of squares
    # x ranges from 1 to p-1
    x = np.arange(1, p, dtype=np.int64)
    squares = (x * x) % p
    
    # We want the smallest x for each square.
    # Since x is increasing, the first occurrence in the 'squares' array for a given value 
    # is the smallest x. 
    # We can use np.unique or simply iterate, but p is small.
    # To be safe and explicit, we fill root_map.
    # np.unique with return_inverse can help, but let's stick to a simple loop 
    # or vectorized assignment if possible. 
    # Actually, since p=1009, a python loop is fine, but let's try to be "strong".
    
    # Method: 
    # For each x, r = x^2 % p.
    # If root_map[r] == 0, set root_map[r] = x.
    # This is hard to vectorize easily without knowing which are new.
    # But p is small. Let's just do it efficiently.
    
    # Alternative: 
    # The set of residues is {x^2 % p for x in 1..p-1}.
    # We can compute all squares.
    # To get the min x for each r:
    # We can sort x by their square value? No, we want min x for each unique square.
    
    # Let's use the property that if r is a QR, its roots are x and p-x.
    # The smallest positive root is min(x, p-x).
    # We can compute the set of residues first.
    
    # Efficient way to get smallest roots:
    # Iterate x from 1 to p//2.
    # r = x^2 % p.
    # root_map[r] = x.
    # This covers all residues exactly once (except 0, which we exclude).
    # Because (p-x)^2 = x^2, and p-x > p//2 for x < p//2.
    # So the smallest root is always <= p//2.
    
    half_p = p // 2
    x_small = np.arange(1, half_p + 1, dtype=np.int64)
    squares_small = (x_small * x_small) % p
    
    # Now squares_small contains each non-zero QR exactly once.
    # We need to map them back.
    # We can use np.searchsorted or a dictionary, but let's use numpy indexing.
    # We want root_map[r] = x for each pair.
    # Since squares_small are unique, we can just assign.
    # But we need to handle the fact that squares_small is not sorted.
    
    # We can create a temporary array of size p and fill it.
    # root_map[squares_small] = x_small
    # This works if squares_small has no duplicates. 
    # For prime p, x^2 = y^2 mod p => x=y or x=-y.
    # Since 1 <= x,y <= p//2, x=y is the only solution.
    # So squares_small contains unique values.
    
    root_map[squares_small] = x_small
    
    # Mark residues
    is_res[squares_small] = True
    # Note: 0 is not in squares_small (since x>=1, p=1009 prime, x^2!=0).
    # So is_res[0] remains False. Correct.
    
    # 2. Find the first run of 5 consecutive True values in is_res[1..p-1]
    # We look for s in 1..p-m such that is_res[s..s+m-1] are all True.
    
    # Use convolution or cumulative sum for sliding window
    # Window sum of is_res.
    # We want sum(is_res[s:s+m]) == m.
    
    # Cumulative sum approach
    # cumsum[i] = sum(is_res[0..i-1])
    # sum(is_res[s:s+m]) = cumsum[s+m] - cumsum[s]
    
    # Ensure is_res is float or int for cumsum
    res_int = is_res.astype(np.int32)
    cumsum = np.cumsum(res_int)
    
    # We need to check indices s such that 1 <= s <= p-m
    # s+m <= p
    # The condition is cumsum[s+m] - cumsum[s] == m
    
    # Create an array of valid starts
    # s ranges from 1 to p-m
    max_s = p - m
    # We need to check s in [1, max_s]
    
    # Calculate sums for all valid s
    # s_indices = np.arange(1, max_s + 1)
    # sums = cumsum[s_indices + m] - cumsum[s_indices]
    
    # To avoid creating large arrays unnecessarily (though p is small):
    # We can just iterate, but let's use vectorization.
    
    s_vals = np.arange(1, max_s + 1)
    window_sums = cumsum[s_vals + m] - cumsum[s_vals]
    
    # Find the first index where window_sums == m
    # np.where returns indices in s_vals
    matches = np.where(window_sums == m)[0]
    
    if len(matches) == 0:
        # Should not happen for p=1009, m=5
        # But if it does, return the best we can? Or raise?
        # The task implies a solution exists.
        # Fallback to linear search just in case of numerical weirdness (unlikely)
        for s in range(1, max_s + 1):
            if all(is_res[s+i] for i in range(m)):
                return {'start': s, 'roots': [int(root_map[s+i]) for i in range(m)]}
        raise RuntimeError("No consecutive run found")
    
    # The first match is the minimum s
    # matches[0] is the index in s_vals, which corresponds to s = s_vals[matches[0]]
    # Since s_vals is [1, 2, ..., max_s], s = matches[0] + 1
    s_min = int(matches[0]) + 1
    
    # Check time
    if time.time() - start_time > timeout:
        # This shouldn't happen, but if it does, return best found
        pass
    
    # 3. Extract roots
    # Roots for s_min, s_min+1, ..., s_min+m-1
    roots = [int(root_map[s_min + i]) for i in range(m)]
    
    return {'start': s_min, 'roots': roots}
