def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 87  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    TOTAL = 1 << N  # 131072
    
    # Precompute weights
    pop_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
    w = np.arange(TOTAL, dtype=np.int64)
    weights = (pop_table[w & 0xFF] + 
               pop_table[(w >> 8) & 0xFF] + 
               pop_table[(w >> 16) & 0xFF])
    
    # Precompute all masks with weight < MIN_D
    masks_list = [m for m in range(TOTAL) if weights[m] < MIN_D]
    masks = np.array(masks_list, dtype=np.int64)
    num_masks = len(masks)
    
    # Build the [17,8,6] linear code (256 words)
    current_code = np.array([0], dtype=np.int64)
    basis = []
    
    for g in range(1, TOTAL):
        if time.time() > deadline:
            break
        xor_vals = g ^ current_code
        min_dist = np.min(weights[xor_vals])
        if min_dist >= MIN_D:
            basis.append(g)
            new_code = np.concatenate([current_code, current_code ^ g])
            current_code = new_code
            if len(basis) >= 8:
                break
    
    code_size = len(current_code)
    code_set = set(current_code.tolist())
    
    # Build forbidden bitset: forbidden[x] = True if x is within distance < 6 of any codeword
    forbidden = np.zeros(TOTAL, dtype=bool)
    for c in current_code:
        forbidden[c ^ masks] = True
    
    # For local search: we need to quickly check if a candidate word is valid
    # i.e., not forbidden by any OTHER codeword (excluding itself if it's in the code)
    
    # Simulated annealing local search
    # State: set of codewords. We try to swap one codeword for a non-codeword.
    # A swap is valid if the new word is at distance >= 6 from all other codewords.
    
    rng = np.random.default_rng(42)
    
    best_code_set = set(code_set)
    best_size = code_size
    
    # Precompute list of non-codeword candidates (words not forbidden by the linear code)
    # Actually, for the swap approach, we need to be more careful.
    # Let's use a different approach: iterative improvement with random restarts
    
    # Approach: Start with the 256-word linear code.
    # Try to add one word at a time using greedy, but with random ordering.
    # Then try to remove and re-add to find better configurations.
    
    # Faster approach: use the forbidden array but make updates O(num_masks) per add/remove
    
    def add_word(code_arr, forbidden_arr, word):
        """Add a word to the code, update forbidden. Returns new code array."""
        forbidden_arr[word ^ masks] = True
        return np.concatenate([code_arr, [word]])
    
    def remove_word(code_arr, forbidden_arr, word):
        """Remove a word from the code. Rebuild forbidden from remaining words."""
        remaining = [c for c in code_arr if c != word]
        new_forbidden = np.zeros(TOTAL, dtype=bool)
        for c in remaining:
            new_forbidden[c ^ masks] = True
        return np.array(remaining, dtype=np.int64), new_forbidden
    
    # Greedy pass: try to add words in a given order
    def greedy_add(code_arr, forbidden_arr, order, time_limit):
        """Try to add words from order that are not forbidden. Returns (new_code, new_forbidden, num_added)."""
        added = 0
        for word in order:
            if time.time() > time_limit:
                break
            word_int = int(word)
            if not forbidden_arr[word_int]:
                forbidden_arr[word_int ^ masks] = True
                code_arr = np.append(code_arr, word_int)
                added += 1
        return code_arr, forbidden_arr, added
    
    # Try multiple greedy passes with different orderings
    # First, precompute some useful orderings
    # Order by weight (high to low)
    order_weight_desc = np.argsort(-weights)
    order_weight_asc = np.argsort(weights)
    
    # Random orderings
    random_orders = []
    for i in range(10):
        random_orders.append(rng.permutation(TOTAL))
    
    all_orders = [order_weight_desc, order_weight_asc] + random_orders
    
    for order in all_orders:
        if time.time() > deadline - 5:
            break
        
        # Start fresh from the linear code
        code_arr = current_code.copy()
        forb = forbidden.copy()
        
        code_arr, forb, added = greedy_add(code_arr, forb, order, deadline - 3)
        
        if len(code_arr) > best_size:
            best_size = len(code_arr)
            best_code_set = set(code_arr.tolist())
    
    # Now try local search: swap operations
    # For each codeword, try replacing it with a non-codeword that is valid
    if time.time() < deadline - 10:
        code_arr = np.array(list(best_code_set), dtype=np.int64)
        forb = np.zeros(TOTAL, dtype=bool)
        for c in code_arr:
            forb[c ^ masks] = True
        
        # Local search with simulated annealing
        # We'll try to improve by swapping codewords
        # A swap: remove codeword i, add non-codeword j
        # Valid if j is not forbidden by any codeword other than i
        
        # Precompute: for each codeword, what words would be valid if we removed it?
        # That's expensive. Instead, let's just try random swaps.
        
        current_size = len(code_arr)
        current_set = set(code_arr.tolist())
        
        # Find candidate non-codewords: words not in current_set
        # We'll randomly sample from non-forbidden words (excluding those forbidden by the code)
        # Actually, a word is a valid replacement if it's at distance >= 6 from all codewords except the one being removed
        
        # Simpler: try adding a random word that is not forbidden, then if we exceed some size, remove a random codeword
        # This is getting complicated. Let's just do more greedy passes with different seeds.
        
        # Additional greedy passes with different random seeds
        for seed in range(20, 40):
            if time.time() > deadline - 3:
                break
            rng2 = np.random.default_rng(seed)
            order = rng2.permutation(TOTAL)
            
            code_arr = current_code.copy()
            forb = forbidden.copy()
            
            code_arr, forb, added = greedy_add(code_arr, forb, order, deadline - 2)
            
            if len(code_arr) > best_size:
                best_size = len(code_arr)
                best_code_set = set(code_arr.tolist())
    
    return list(best_code_set)
