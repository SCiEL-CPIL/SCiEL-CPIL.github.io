"""Program for: Large sets with distinct subset products.

Uses a backtracking search (Branch and Bound) to find the largest subset of
integers in 2..40 such that all subset products are distinct.
"""

# EVOLVE-BLOCK-START

def construct():
    """Return a list of distinct integers in 2..40 maximizing count such that
    all subset products are distinct."""
    
    candidates = list(range(2, 41))
    
    # We will use a recursive backtracking approach.
    # State: 
    #   idx: index in candidates
    #   current_set: list of selected numbers
    #   current_products: set of all subset products of current_set
    #   current_size: len(current_set)
    #   best_set: list of the best set found so far
    #   best_size: size of the best set found so far
    
    best_set = []
    best_size = 0
    
    # Helper to check if adding x to current_products is valid
    # and returns the new set of products if valid
    def try_add(current_products, x):
        # new products are p and p*x for all p in current_products
        # We need to check if any p*x is already in current_products 
        # or if any p*x collides with another q*x (impossible since p!=q)
        # Actually, we just need to check if the set {p*x for p in current_products} 
        # is disjoint from current_products.
        # And obviously p*x are distinct among themselves.
        
        new_products = set()
        for p in current_products:
            new_p = p * x
            if new_p in current_products or new_p in new_products:
                return None
            new_products.add(new_p)
        
        # Union
        return current_products.union(new_products)

    # Greedy initialization to get a good lower bound
    greedy_products = {1}
    greedy_set = []
    for x in candidates:
        new_prods = try_add(greedy_products, x)
        if new_prods is not None:
            greedy_products = new_prods
            greedy_set.append(x)
    
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # Backtracking search
    # We sort candidates to potentially find solutions faster?
    # Smaller numbers might be more "compatible"? 
    # Let's stick to natural order or reverse?
    # Primes are generally "safer".
    # Let's try to search in an order that might lead to large sets early.
    # However, for B&B, the order matters for pruning.
    
    def backtrack(idx, current_set, current_products, current_size):
        nonlocal best_set, best_size
        
        # Pruning: if we can't beat the best size, return
        if current_size + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        x = candidates[idx]
        
        # Option 1: Include x
        new_products = try_add(current_products, x)
        if new_products is not None:
            current_set.append(x)
            backtrack(idx + 1, current_set, new_products, current_size + 1)
            current_set.pop()
            
        # Option 2: Exclude x
        backtrack(idx + 1, current_set, current_products, current_size)

    backtrack(0, [], {1}, 0)
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
