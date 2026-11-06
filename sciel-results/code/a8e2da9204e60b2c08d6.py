def construct():
    def greedy_solution():
        values = []
        products = set()
        suffixes = []
        
        for x in range(2, 129):
            new_suffixes = [x]
            for s in suffixes:
                new_suffixes.append(x * s)
            
            if not products.intersection(new_suffixes):
                values.append(x)
                products.update(new_suffixes)
                suffixes = new_suffixes
        
        return values

    best_seq = greedy_solution()
    best_len = len(best_seq)
    
    # If best_len is already high, DFS might be fast.
    # We use in-place sets for products.
    
    # We need to be careful with recursion limit.
    # sys.setrecursionlimit(1000)
    
    def dfs(start, seq, prods, suffixes):
        nonlocal best_seq, best_len
        
        # Pruning
        # Max possible length from here
        max_possible = len(seq) + (128 - start + 1)
        if max_possible <= best_len:
            return
        
        # Try to take 'start'
        new_suffixes = [start]
        for s in suffixes:
            new_suffixes.append(start * s)
        
        # Check collision
        # We can check intersection efficiently
        has_conflict = False
        for p in new_suffixes:
            if p in prods:
                has_conflict = True
                break
        
        if not has_conflict:
            # Add to state
            prods.update(new_suffixes)
            seq.append(start)
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq.copy()
                
            dfs(start + 1, seq, prods, new_suffixes)
            
            # Backtrack
            seq.pop()
            prods.difference_update(new_suffixes)
            
        # Try to skip 'start'
        # Check if skipping can still beat best_len
        if len(seq) + (128 - start) > best_len:
            dfs(start + 1, seq, prods, suffixes)

    dfs(2, [], set(), [])
    
    return best_seq
