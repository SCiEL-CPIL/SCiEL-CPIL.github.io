"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

_DEADLINE = 85.0  # safety margin before 90s kill

def _has_distinct_subset_products(nums):
    """Check if all 2^k subset products are distinct."""
    k = len(nums)
    if k == 0:
        return True
    # Use a set to track products. Start with 1 (empty product).
    products = {1}
    for x in nums:
        new_products = set()
        for p in products:
            new_products.add(p * x)
        # If any collision happens within the new products or between old and new, invalid
        # Actually, we need to check if the combined set has size 2^(i+1)
        # Better approach: maintain the full set of products as we build up
        # But that's 2^k memory. For k up to ~15-16, 2^16 = 65536, which is fine.
        # Let's just build the full set incrementally and check size.
        pass
    
    # More efficient: build all subset products and check for duplicates
    # For small k (<= 18), 2^18 = 262144, which is manageable.
    # But we want to detect collisions early.
    
    # Incremental approach:
    # Start with products = {1}
    # For each new element x, new_products = {p * x for p in products}
    # If len(products ∪ new_products) != len(products) + len(new_products), there's a collision
    # Actually, collision means some p1 * x == p2 * x (impossible since x != 0) 
    # or p1 * x == p2 where p1, p2 in old products.
    # So we just need to check if new_products intersects with products.
    
    products = {1}
    for x in nums:
        new_products = set()
        for p in products:
            np = p * x
            if np in products or np in new_products:
                return False
            new_products.add(np)
        products.update(new_products)
    return True

def _build_max_set(candidates, max_size, deadline):
    """Greedy repair: start with all candidates, remove elements causing collisions."""
    # Start with the full candidate set
    current = list(candidates)
    
    # We want to maximize the size. 
    # Strategy: try to keep as many as possible.
    # A simple greedy: sort candidates, add one by one if it doesn't break distinctness.
    # But which order? 
    # Heuristic: elements that are "more independent" (e.g., prime powers with large exponents) might be better?
    # Or just try all permutations? No, too slow.
    
    # Let's try a different approach: 
    # 1. Generate a pool of promising candidates (prime powers).
    # 2. Use a backtracking search to find the largest subset with distinct products.
    
    # For now, let's implement a simple backtracking with pruning.
    # Sort candidates by some heuristic (e.g., value, or prime factor count).
    # Try to include each candidate if it doesn't cause a collision with the current set.
    
    # To speed up collision checks, we can maintain the set of products.
    # But backtracking requires undoing, which is expensive with large product sets.
    
    # Alternative: since the max size is small (<= 18), and we have a deadline,
    # we can try random restarts with greedy construction.
    
    best_set = []
    best_size = 0
    
    # Candidate pool: prime powers in [2, 40]
    # 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37
    # Let's generate them properly.
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def is_prime_power(n):
        if n < 2:
            return False
        # Check if n is a power of a prime
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                # Check if n is a power of p
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
                else:
                    return False
        return is_prime(n)
    
    candidates = [x for x in range(2, 41) if is_prime_power(x)]
    
    # Sort candidates by value
    candidates.sort()
    
    # Greedy: try to add candidates in order
    # But we might want to try different orders.
    # For now, let's do a simple greedy with a few restarts with shuffled orders.
    
    import random
    
    start_time = time.time()
    attempts = 0
    
    while time.time() < deadline and attempts < 1000:
        attempts += 1
        # Shuffle candidates
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        products = {1}
        
        for x in shuffled:
            if len(current_set) >= max_size:
                break
            
            # Check if adding x causes a collision
            new_products = set()
            collision = False
            for p in products:
                np = p * x
                if np in products or np in new_products:
                    collision = True
                    break
                new_products.add(np)
            
            if not collision:
                current_set.append(x)
                products.update(new_products)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we reached max_size, we can stop early
            if best_size == max_size:
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Get the maximum set
    max_set = _build_max_set(range(2, 41), 18, deadline)
    
    # Ensure we return a list of Python ints
    return [int(x) for x in max_set]
# EVOLVE-BLOCK-END
