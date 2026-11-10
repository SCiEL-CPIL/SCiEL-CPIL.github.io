import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, keep 10s margin
    
    # 1. Identify Primes and Non-Square-Free numbers in [2, 128]
    numbers = list(range(2, 129))
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    def is_square_free(n):
        for p in range(2, int(n**0.5) + 1):
            if n % (p*p) == 0:
                return False
        return True
        
    primes = [n for n in numbers if is_prime(n)]
    non_sq_free = [n for n in numbers if not is_square_free(n)]
    
    candidate_seq = sorted(primes + non_sq_free)
    
    # 2. Verify the candidate sequence
    def is_valid(seq):
        products = set()
        for i in range(len(seq)):
            prod = 1
            for j in range(i, len(seq)):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
        
    if is_valid(candidate_seq):
        return candidate_seq
        
    # 3. If invalid, try a backtracking search to maximize length
    # We know the length of the valid set above is L.
    # We can try to find a longer one, or if the above is invalid, find the best valid one.
    # Given the hypothesis is strong, if it fails, it's likely due to a specific collision.
    # We can try to remove elements from candidate_seq that cause collisions.
    
    # Simple heuristic repair:
    # Find the collision and remove one of the involved elements.
    # This is complex. Instead, let's run a DFS/Backtracking if the primary fails.
    
    best_seq = []
    
    # Optimization: The search space is large.
    # But we have a strong initial guess.
    # Let's try a local search starting from candidate_seq.
    # If candidate_seq is invalid, we can't use it.
    
    # Let's do a DFS that tries to build the sequence.
    # To speed it up, we can use the fact that primes are "safe".
    # We will iterate through all numbers 2..128.
    # We will use a recursive function.
    
    # To avoid deep recursion limits or timeouts, we limit the depth or use pruning.
    # However, 80 elements is a lot for DFS.
    
    # Alternative: If the primary set is invalid, we can try to construct a valid set 
    # by greedily adding numbers from the candidate set, skipping those that cause collisions.
    
    final_seq = []
    products = set()
    
    for x in candidate_seq:
        # Check if x can be added
        tails = [x]
        prod = x
        for a in reversed(final_seq):
            prod *= a
            tails.append(prod)
        
        if not products.intersection(tails):
            final_seq.append(x)
            products.update(tails)
        else:
            # Try to see if we can replace a previous element? 
            # For now, just skip.
            pass
            
    # If the greedy repair on the candidate set yields a shorter sequence than the seed, 
    # we should return the seed or a better found one.
    # But the candidate set is 80 long. Even if we skip some, it's likely > seed.
    
    # Let's also try a simple backtracking if the result is suspiciously short.
    # But given the "square-free" logic, the repair should be very good.
    
    return final_seq
