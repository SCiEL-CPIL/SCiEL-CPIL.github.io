"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd
from functools import reduce

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _distinct_subset_products_count(numbers):
    """
    Given a list of integers, compute the number of distinct subset products.
    Returns True if all 2^k subset products are distinct (i.e., count == 2^len(numbers)).
    Uses bitset for efficiency.
    """
    k = len(numbers)
    if k == 0:
        return 1
    # Start with product 1 (empty set)
    # Products can get very large, so we use a set.
    # Optimization: check for 1 in input -> if 1 is in numbers, then subset {1} and {} both product to 1.
    if 1 in numbers:
        return False
    
    # We can use a set of products.
    # For k=18, 2^18 = 262,144 products.
    # Storing them in a set is fine memory-wise.
    products = {1}
    for x in numbers:
        if x == 1:
            return False
        # Update products
        new_products = set()
        for p in products:
            new_products.add(p * x)
        # Check for collision: if any product in new_products was already in products, it's a collision.
        # Actually, we just need to ensure the final set size is 2^k.
        # A faster way: 
        # products = products.union(new_products)
        # if len(products) != (1 << (i+1)): return False
        
        # Let's do it incrementally
        products.update(new_products)
        expected_size = 1 << (len(products) and len(numbers) and 0) # placeholder
        # Simpler:
        # We know we are processing the i-th number (0-indexed).
        # After processing i numbers, we should have 2^(i+1) distinct products.
        pass
        
    # The incremental check is better.
    # Let's rewrite _check_valid to do the incremental check properly.

def _check_valid(numbers):
    """
    Returns True if all subset products of 'numbers' are distinct.
    """
    if not numbers:
        return True
    if 1 in numbers:
        return False
    
    products = {1}
    for i, x in enumerate(numbers):
        if x == 1:
            return False
        # Generate new products by multiplying existing ones by x
        # If any of these new products are already in 'products', we have a collision.
        # Note: Since all numbers > 1, p*x > p for p>=1. 
        # So p*x cannot be equal to p.
        # However, p1*x could equal p2*x? No, x!=0.
        # p1*x could equal p3 (an existing product)? Yes.
        # Example: numbers = [2, 4]. 
        # i=0: x=2. products={1}. new={2}. products={1,2}.
        # i=1: x=4. products={1,2}. new={4,8}. 
        # 4 is not in {1,2}. 8 is not in {1,2}. 
        # products={1,2,4,8}. Size 4 = 2^2. OK.
        
        # Example: numbers = [2, 2].
        # i=0: x=2. products={1}. new={2}. products={1,2}.
        # i=1: x=2. products={1,2}. new={2,4}.
        # 2 is in {1,2}! Collision.
        
        # So we just need to check if new_products intersects products.
        # But wait, we are adding them to products.
        # new_products = {p * x for p in products}
        # if new_products & products: return False
        # products |= new_products
        
        new_prods = [p * x for p in products]
        # Check intersection
        # It is faster to just check if any np is in products
        # Since sets are small, this is fine.
        for np in new_prods:
            if np in products:
                return False
        products.update(new_prods)
        
    return True

def _try_greedy(start_time, deadline, start_list):
    """
    Greedy search: start with a base list, try to add more numbers.
    """
    current = list(start_list)
    if not _check_valid(current):
        return current  # Return whatever we have, even if invalid? No, must be valid.
        # But the base list should be valid.
        
    candidates = [i for i in range(2, 41) if i not in current]
    # Sort candidates? 
    # Smaller numbers are "harder" to fit? Or primes are "easier"?
    # Primes are generally good.
    # Let's sort by "primeness" or just iterate through all.
    
    # Try adding candidates one by one.
    # Order matters.
    # Strategy 1: Try primes first, then composites.
    # Strategy 2: Try numbers that don't share prime factors with current.
    
    # Let's do a simple greedy: try each candidate, if it fits, add it.
    # We might need to be smarter about the order.
    
    # Heuristic: Numbers with fewer prime factors (or distinct primes) are better.
    # Let's just try all candidates in a specific order.
    
    # Order: Primes first, then composites.
    primes = [i for i in range(2, 41) if _is_prime(i)]
    composites = [i for i in range(2, 41) if not _is_prime(i)]
    
    # We want to add numbers that are "independent" in terms of prime factorization.
    
    # Let's try adding primes first.
    for p in primes:
        if p in current:
            continue
        if time.time() > deadline:
            break
        test_list = current + [p]
        if _check_valid(test_list):
            current = test_list
            
    # Now try composites.
    # Sort composites by something? 
    # Maybe composites that are powers of primes already in the set are bad.
    # Composites with new prime factors are good.
    
    # Let's just iterate through composites.
    for c in composites:
        if c in current:
            continue
        if time.time() > deadline:
            break
        test_list = current + [c]
        if _check_valid(test_list):
            current = test_list
            
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial valid sets
    # We know {2, 3, 5, ...} primes are valid.
    # But we can do better by mixing.
    
    # Let's start with a few primes.
    base_primes = [2, 3, 5, 7, 11]
    
    # Try greedy from different starting points.
    best = []
    
    # Starting point 1: Small primes
    start1 = [2, 3, 5]
    res1 = _try_greedy(start_time, deadline, start1)
    if len(res1) > len(best):
        best = res1
        
    if time.time() > deadline:
        return best
        
    # Starting point 2: Larger primes
    start2 = [7, 11, 13, 17, 19]
    res2 = _try_greedy(start_time, deadline, start2)
    if len(res2) > len(best):
        best = res2
        
    if time.time() > deadline:
        return best
        
    # Starting point 3: Mix
    start3 = [2, 7, 13]
    res3 = _try_greedy(start_time, deadline, start3)
    if len(res3) > len(best):
        best = res3
        
    if time.time() > deadline:
        return best
        
    # Try to extend best with local search?
    # Or just return best.
    
    # The greedy might get stuck.
    # Let's try a few more random starts if time permits.
    
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random start of size 3-5
        size = random.randint(3, 5)
        start = random.sample(range(2, 41), size)
        # Ensure valid start
        if not _check_valid(start):
            continue
        res = _try_greedy(start_time, deadline, start)
        if len(res) > len(best):
            best = res
            # If we hit a very good score, we can break or continue
            if len(best) >= 18:
                break
                
    return best

# EVOLVE-BLOCK-END
