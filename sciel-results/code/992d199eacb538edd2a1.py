import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    best_seq = []
    best_len = 0

    def check_and_get_new_products(current_seq, x):
        """
        Returns a list of new subarray products ending at x, 
        or None if there's a collision with existing products or internal collisions.
        Note: Internal collisions among new products are impossible for strictly increasing ints >= 2.
        Proof: Let new products be x, x*a_k, x*a_k*a_{k-1}, ...
        If x*a_k*a_{k-1}...a_j == x*a_k*...*a_i with j < i, then a_{k-1}...a_j == a_k...a_i.
        Since a_i < a_{i+1} < ..., the product of a shorter suffix cannot equal the product of a longer suffix 
        containing it and larger numbers? 
        Actually, let's just check against the global set. 
        """
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(current_seq):
            prod *= a
            new_prods.append(prod)
        return new_prods

    def dfs(last_picked, current_seq, current_products):
        nonlocal best_seq, best_len

        if time.time() > deadline:
            return

        current_len = len(current_seq)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]

        # Pruning: Max possible extension
        # Remaining numbers are last_picked+1 to 128.
        # Count is 128 - last_picked.
        if current_len + (128 - last_picked) <= best_len:
            return

        if last_picked >= 128:
            return

        # Try adding next numbers
        for x in range(last_picked + 1, 129):
            if time.time() > deadline:
                return

            new_prods = check_and_get_new_products(current_seq, x)
            
            # Check collision with existing products
            valid = True
            for p in new_prods:
                if p in current_products:
                    valid = False
                    break
            
            if valid:
                # Add x
                current_seq.append(x)
                current_products.update(new_prods)
                
                dfs(x, current_seq, current_products)
                
                # Undo
                current_seq.pop()
                for p in new_prods:
                    current_products.remove(p)
                    
                if time.time() > deadline:
                    return

    # 1. Greedy initialization to get a good lower bound quickly
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        new_prods = check_and_get_new_products(greedy_seq, x)
        if not any(p in greedy_prods for p in new_prods):
            greedy_seq.append(x)
            greedy_prods.update(new_prods)
    
    best_seq = greedy_seq
    best_len = len(greedy_seq)
    
    # 2. DFS to find optimal solution
    # Start with empty sequence, last_picked=1 (so we can pick 2)
    dfs(1, [], set())
    
    return best_seq
