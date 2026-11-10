def construct():
    """
    Finds the longest strictly increasing sequence of integers in [2, 128]
    such that the products of every non-empty consecutive block are distinct.
    """
    
    # Global best
    best_seq = []
    
    def dfs(last_val, current_seq, current_suffixes, all_products):
        nonlocal best_seq
        
        # Update best if current is longer
        if len(current_seq) > len(best_seq):
            best_seq = current_seq[:]
            
        # Pruning: Even if we pick every remaining number, can we beat the best?
        # Max possible additional numbers = 128 - last_val
        if len(current_seq) + (128 - last_val) <= len(best_seq):
            return
            
        # Try adding the next number
        for x in range(last_val + 1, 129):
            # Generate new suffixes
            # Suffixes are products of subarrays ending at the last element
            # New subarrays ending at x:
            # 1. [x] -> product x
            # 2. [last_elem, ..., x] -> product (suffix_of_last) * x
            new_suffixes = [x]
            for p in current_suffixes:
                new_suffixes.append(p * x)
            
            # Check for conflicts with existing products
            # If any new product is already in all_products, it's a conflict
            has_conflict = False
            for p in new_suffixes:
                if p in all_products:
                    has_conflict = True
                    break
            
            if has_conflict:
                continue
            
            # If no conflict, update state and recurse
            # Create copies to allow backtracking (immutable style)
            new_all_products = set(all_products)
            new_all_products.update(new_suffixes)
            
            dfs(x, current_seq + [x], new_suffixes, new_all_products)
            
    # Start DFS
    # last_val = 1 (since sequence starts >= 2)
    # current_seq = []
    # current_suffixes = []
    # all_products = set()
    
    dfs(1, [], [], set())
    
    return best_seq
