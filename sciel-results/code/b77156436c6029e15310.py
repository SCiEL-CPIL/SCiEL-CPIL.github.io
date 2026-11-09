<<<<<<< SEARCH
    # ========== MIS BRANCH-AND-BOUND APPROACH ==========
    # The problem: find longest strictly increasing sequence from [2,128]
    # such that all consecutive block products are distinct.
    
    # Key insight: if we fix the SET of numbers, the optimal ordering is the
    # sorted order (since sequence must be strictly increasing). So the problem
    # reduces to: find the largest subset S of {2,...,128} such that when sorted,
    # all consecutive block products are distinct.
    
    # This is NOT a simple independent set problem because conflicts depend on
    # the full sequence context (block products involve multiple elements).
    # However, we can use a different approach: branch on which numbers to include.
    
    # Let's use a smarter search: build the sequence incrementally,
    # tracking which block products have been seen.
    
    def mis_search():
        """Branch and bound to find longest valid sequence."""
        # We'll build the sequence left to right, choosing from remaining numbers
        # in increasing order (since sequence must be strictly increasing).
        
        # State: (current_seq, last_element, seen_block_products)
        # But seen_block_products can be large, so we need to be careful.
        
        # Alternative: use the fact that we must pick a subset and sort it.
        # So we can think of it as: which subset of {2,...,128} is valid?
        
        # Let's use a different strategy: try to include as many numbers as possible
        # by doing a greedy with backtracking.
        
        # Precompute: for each pair of numbers (a,b) with a < b, check if they can
        # coexist in a valid sequence (i.e., there's no immediate conflict).
        # Actually, the conflict is global, so this doesn't simplify directly.
        
        # Instead, let's use a branch-and-bound where we decide for each number
        # (in order 2, 3, 4, ..., 128) whether to include it.
        
        # To make this tractable, we use the following:
        # 1. Greedy lower bound to get a good initial solution
        # 2. Upper bound: remaining numbers + current length
        # 3. Prune if upper bound <= best found
        
        # Get a good initial solution from randomized greedy
        best_seq = build_initial(42)
        best_len = len(best_seq)
        
        # Now try to improve via local search
        best_seq = local_search(best_seq, max_iter=200)
        best_len = len(best_seq)
        
        # Try multiple random restarts
        for seed in range(20):
            if time.time() > deadline:
                break
            seq = build_initial(seed * 12345 + 7)
            seq = local_search(seq, max_iter=100)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
        
        return best_seq
=======
    # ========== CSP BACKTRACKING APPROACH ==========
    # Build sequence incrementally using backtracking with conflict detection.
    
    def csp_search():
        """CSP-style search to find longest valid sequence."""
        # Get a good initial solution
        best_seq = build_initial(42)
        best_len = len(best_seq)
        best_prods = get_block_products(best_seq)
        
        # Try multiple starting points with CSP extension
        for seed in range(30):
            if time.time() > deadline:
                break
            
            # Build a smaller initial sequence
            rng = random.Random(seed * 999 + 1)
            prime_subset_size = rng.randint(8, 15)
            prime_subset = sorted(rng.sample(primes, prime_subset_size))
            
            if not check_sequence_valid(prime_subset):
                continue
            
            current_seq = prime_subset[:]
            current_prods = get_block_products(current_seq)
            
            # Try to extend via greedy + backtracking
            remaining = [x for x in all_numbers if x not in current_seq]
            
            # Greedy extension first
            progress = True
            while progress and time.time() < deadline:
                progress = False
                for c in sorted(remaining):
                    if current_seq and c <= current_seq[-1]:
                        continue
                    new_seq = current_seq + [c]
                    new_prods = get_block_products(new_seq)
                    if len(new_prods) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current_seq = new_seq
                        current_prods = new_prods
                        remaining.remove(c)
                        progress = True
                        break
            
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = current_seq[:]
                best_prods = current_prods
            
            # Try local search on this sequence
            improved = local_search(current_seq, max_iter=50)
            if len(improved) > best_len:
                best_len = len(improved)
                best_seq = improved[:]
                best_prods = get_block_products(best_seq)
        
        return best_seq
>>>>>>> REPLACE

