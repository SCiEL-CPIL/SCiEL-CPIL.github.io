import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin
    
    n = 30
    
    # Compute L = lcm(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # a[i] = L // i
    a = [0] * (n + 1)
    for i in range(1, n + 1):
        a[i] = L // i
        
    mid = 15
    left_indices = list(range(1, mid + 1))
    right_indices = list(range(mid + 1, n + 1))
    
    # Function to generate all sums and their indices for a given list of indices
    # Returns a dictionary: sum_value -> index_in_generation_order
    # We generate in lexicographic order of coefficients? Or just any order?
    # Any order is fine as long as we can decode index back to coeffs.
    # Let's use a base-3 representation where digit 0->-1, 1->0, 2->1?
    # Or simpler: just iterate and store the coeffs if we want to avoid decoding?
    # Storing coeffs for 14M entries is too much memory.
    # Decoding index is safe.
    
    # We need a mapping from index to coeffs.
    # If we generate sequentially, index k corresponds to the k-th combination.
    # We can define the order explicitly.
    # Let's generate left sums.
    
    left_dict = {}
    
    # To make decoding easy, let's define the generation order.
    # We can use a recursive function or iterative.
    # Iterative is faster.
    
    # Let's just store the coeffs for the *first* occurrence of each sum?
    # No, we don't know if a sum repeats until we finish.
    # But we only need *one* solution.
    # So if we store `sum -> coeffs_tuple`, we use too much memory.
    # `sum -> index` is better.
    
    # How to map index to coeffs?
    # If we generate in a specific order, e.g., nested loops or base-3 counter.
    # Let's use a base-3 counter.
    # Index 0: (-1, -1, ..., -1)
    # Index 1: (-1, -1, ..., 0)
    # ...
    # This is standard base-3 if we map -1->0, 0->1, 1->2.
    
    def index_to_coeffs(idx, num_vars):
        coeffs = []
        for _ in range(num_vars):
            digit = idx % 3
            idx //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        coeffs.reverse()
        return coeffs

    # Generate left sums
    # We iterate idx from 0 to 3^15 - 1
    # This might be slow in pure Python loop for 14M items.
    # 14M iterations * (div/mod + dict lookup) might take 5-10 seconds.
    
    # Optimization:
    # We can generate sums iteratively without decoding every time?
    # We need the sum to put in dict.
    # We need the index to put in dict.
    
    # Let's try the iterative sum generation which is faster than decoding base-3 each time?
    # No, we need the index to store.
    # But we can just store the coeffs?
    # Memory check: 14M entries.
    # Each key is an int (~28 bytes).
    # Each value is a tuple of 15 ints (~15*28 + overhead = ~400 bytes).
    # 14M * 400 bytes = 5.6 GB. Too much.
    
    # So `sum -> index` is the way.
    
    # Is 14M iterations too slow in Python?
    # A simple loop `for i in range(10**7): pass` takes ~0.5s.
    # With dict ops and arithmetic, maybe 5-10s.
    # 85s deadline is generous.
    
    # Let's implement the base-3 counter approach.
    
    num_left = len(left_indices)
    num_right = len(right_indices)
    
    total_left = 3 ** num_left
    
    # Precompute a_i for left and right
    a_left = [a[i] for i in left_indices]
    a_right = [a[i] for i in right_indices]
    
    # Generate left
    # We can speed up by using a list of sums and building it up?
    # But we need the index.
    # If we build it up, the index is just the counter.
    
    # Let's do the base-3 counter.
    # To speed up, we can avoid function call for index_to_coeffs in the main loop.
    # We only need to decode if we find a match.
    
    # Main loop for left
    for idx in range(total_left):
        # Calculate sum
        # Decoding base-3 in the loop is slow.
        # Alternative: Generate sums recursively/iteratively and keep track of index?
        # Iterative generation:
        # sums = [0]
        # for each a:
        #   new_sums = [s - a, s, s + a for s in sums]
        #   sums = new_sums
        # This gives us the sums. But how do we map sum to index?
        # The index in the final list corresponds to the base-3 number if we order them correctly.
        # If we append -1, 0, 1 in that order, the index is exactly the base-3 number.
        
        # Let's use this iterative method to generate the list of sums.
        # It's 14M list elements.
        # Memory: 14M ints. ~100MB. OK.
        # Time: 15 iterations, each extending a list by 3x.
        # This is much faster than 14M individual base-3 decodes.
        
        pass # Placeholder for logic below

    # Let's restructure to use the iterative sum generation.
    
    # Left side
    sums_left = [0]
    # We don't need to store patterns, just sums.
    # But we need to map sum -> index.
    # The index in `sums_left` at the end will be the base-3 index if we construct it carefully.
    # Let's verify.
    # Start: [0] (idx 0)
    # Step 1 (a1): 
    #   idx 0: -a1
    #   idx 1: 0
    #   idx 2: a1
    #   List: [-a1, 0, a1]
    # Step 2 (a2):
    #   For each s in list:
    #     -s-a2, s, s+a2
    #   Order:
    #   (-a1)-a2, (-a1), (-a1)+a2
    #   0-a2, 0, 0+a2
    #   a1-a2, a1, a1+a2
    #   Indices 0,1,2 -> -1,0,1 for a1?
    #   Index 0: c1=-1, c2=-1. Base3: 00 -> 0.
    #   Index 1: c1=-1, c2=0. Base3: 01 -> 1.
    #   Index 2: c1=-1, c2=1. Base3: 02 -> 2.
    #   Index 3: c1=0, c2=-1. Base3: 10 -> 3.
    #   ...
    #   Yes, the index in the list generated by extending [s-a, s, s+a] corresponds to the base-3 number where digits are 0,1,2 for -1,0,1.
    
    # So we can generate the list of sums, and then iterate through it with `enumerate` to fill the dict.
    
    # Generate left sums list
    current_sums = [0]
    for val in a_left:
        new_sums = []
        # Optimization: pre-allocate or use list comprehension?
        # List comprehension is faster.
        # new_sums = [s - val for s in current_sums] + [s for s in current_sums] + [s + val for s in current_sums]
        # This creates 3 lists and concatenates.
        # Or:
        # new_sums = []
        # for s in current_sums:
        #    new_sums.append(s - val)
        #    new_sums.append(s)
        #    new_sums.append(s + val)
        # The loop version is often faster for appending in Python due to no intermediate list creation?
        # Actually, `list comprehension` + `extend` is usually good.
        # Let's use a simple loop for clarity and decent speed.
        
        # To speed up, we can use `itertools.chain`?
        # Or just:
        # new_sums = [0] * (len(current_sums) * 3)
        # ... fill it?
        
        # Let's try the simple append loop.
        for s in current_sums:
            new_sums.append(s - val)
            new_sums.append(s)
            new_sums.append(s + val)
        current_sums = new_sums
        
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    # Fill left_dict
    # sum -> index
    # If duplicate sums exist, we keep the first one (lowest index).
    for idx, s in enumerate(current_sums):
        if s not in left_dict:
            left_dict[s] = idx
            
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    # Generate right sums
    current_sums_right = [0]
    for val in a_right:
        new_sums = []
        for s in current_sums_right:
            new_sums.append(s - val)
            new_sums.append(s)
            new_sums.append(s + val)
        current_sums_right = new_sums
        
    if time.time() - start_time > deadline:
        return [0] * 28 + [1, -1]

    # Check for matches
    # We want N_total = N_left + N_right = 1 or -1
    # So N_left = 1 - N_right or N_left = -1 - N_right
    
    found = False
    best_coeffs = None
    
    for idx_r, s_r in enumerate(current_sums_right):
        target1 = 1 - s_r
        if target1 in left_dict:
            idx_l = left_dict[target1]
            # Decode idx_l to get left coeffs
            c_left = index_to_coeffs(idx_l, num_left)
            # Decode idx_r to get right coeffs
            c_right = index_to_coeffs(idx_r, num_right)
            
            # Construct full coeffs
            coeffs = [0] * (n + 1)
            for k, i in enumerate(left_indices):
                coeffs[i] = c_left[k]
            for k, i in enumerate(right_indices):
                coeffs[i] = c_right[k]
                
            # Verify sum is non-zero (it should be 1/L)
            # We can skip verification for speed if we trust the math, but better to be safe.
            # The problem states "signed sum ... must be nonzero".
            # If N_total = 1, S = 1/L != 0.
            
            best_coeffs = coeffs[1:]
            found = True
            break
            
        target2 = -1 - s_r
        if target2 in left_dict:
            idx_l = left_dict[target2]
            c_left = index_to_coeffs(idx_l, num_left)
            c_right = index_to_coeffs(idx_r, num_right)
            
            coeffs = [0] * (n + 1)
            for k, i in enumerate(left_indices):
                coeffs[i] = c_left[k]
            for k, i in enumerate(right_indices):
                coeffs[i] = c_right[k]
                
            best_coeffs = coeffs[1:]
            found = True
            break
            
        if time.time() - start_time > deadline:
            break

    if found:
        return best_coeffs
    else:
        # Fallback
        return [0] * 28 + [1, -1]
