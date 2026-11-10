import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Candidates are 2..40
    candidates = list(range(2, 41))
    
    best_set = []
    best_size = 0
    
    # Pre-compute prime factorization exponents? 
    # Or just use the product set method.
    # Product set method is exact and simple for small N.
    
    def is_valid_addition(current_products, x):
        # current_products is a set of products
        # New products are x * p for p in current_products
        # Check if any new product is in current_products
        # Also check if x itself is in current_products (subset {x} vs existing)
        # Actually, x * 1 = x. So if x in current_products, collision.
        if x in current_products:
            return False
        
        new_products = set()
        for p in current_products:
            np = p * x
            if np in current_products:
                return False
            new_products.add(np)
            
        # Check if any new product collides with another new product?
        # No, if p1 != p2, then p1*x != p2*x.
        # So we just need to check against old products.
        return True

    def get_new_products(current_products, x):
        return {p * x for p in current_products}

    def greedy_search(candidates, deadline):
        # Sort candidates to prioritize smaller numbers? Or random?
        # Smaller numbers have smaller products, might collide more?
        # Larger numbers have larger products, might be "safer"?
        # Let's try random shuffling.
        
        max_iterations = 500
        for i in range(max_iterations):
            if time.time() > deadline:
                return best_set
            
            random.shuffle(candidates)
            current_set = []
            current_products = {1}
            
            for x in candidates:
                if time.time() > deadline:
                    return best_set
                
                if is_valid_addition(current_products, x):
                    current_set.append(x)
                    # Update products
                    new_prods = get_new_products(current_products, x)
                    current_products.update(new_prods)
            
            if len(current_set) > len(best_set):
                best_set = current_set[:]
                if len(best_set) >= 18: # Theoretical max?
                    return best_set

    # Initial run with sorted candidates (often good for primes)
    candidates_sorted = sorted(candidates)
    current_set = []
    current_products = {1}
    for x in candidates_sorted:
        if is_valid_addition(current_products, x):
            current_set.append(x)
            new_prods = get_new_products(current_products, x)
            current_products.update(new_prods)
    
    if len(current_set) > len(best_set):
        best_set = current_set[:]
        
    # Run randomized searches
    greedy_search(candidates, deadline)
    
    return best_set
