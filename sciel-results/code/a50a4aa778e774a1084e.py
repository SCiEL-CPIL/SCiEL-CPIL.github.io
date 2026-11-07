import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Candidate numbers 2 to 40
    candidates = list(range(2, 41))
    
    # To maximize the chance of finding a large set, we can try multiple random permutations
    # or a structured greedy approach.
    # The structured approach:
    # 1. Include all primes (12 of them).
    # 2. Try to add composites that are "multiplicatively independent" from the rest.
    
    best_set = []
    best_products = {1}
    
    # Strategy: Greedy addition.
    # Order candidates such that "independent" ones are likely to be picked.
    # Primes are always safe if not used in factorization of others?
    # Actually, if we pick a prime, we can't pick its multiples?
    # No, we can pick multiples if the subset product uniqueness holds.
    # E.g., {2, 4} is valid. {2, 4, 8} is not.
    
    # Let's try a greedy approach with a specific order.
    # Order: Primes first, then powers of 2, then powers of 3, etc.
    # Or simply sort by number of prime factors?
    
    # A robust greedy:
    # Try adding candidates in increasing order of value.
    # If collision, skip.
    
    # However, order matters. 
    # Let's try a few different orderings and keep the best result.
    
    def try_greedy(order):
        current_set = []
        current_products = {1}
        
        for c in order:
            if time.time() > deadline:
                break
            
            new_products = {p * c for p in current_products}
            
            # Check for collision
            if new_products.isdisjoint(current_products):
                current_set.append(c)
                current_products |= new_products
            else:
                # Optional: Backtracking? 
                # For now, just skip.
                pass
        
        return current_set, current_products

    # Generate a few candidate orders
    orders = []
    
    # Order 1: Increasing
    orders.append(candidates)
    
    # Order 2: Primes first, then composites
    primes = [n for n in candidates if all(n % i for i in range(2, int(n**0.5)+1))]
    composites = [n for n in candidates if n not in primes]
    orders.append(primes + composites)
    
    # Order 3: Small prime powers first
    # 2, 4, 8, 16, 32
    # 3, 9, 27
    # 5, 25
    # 7
    # Primes > 7
    powers_of_2 = [2, 4, 8, 16, 32]
    powers_of_3 = [3, 9, 27]
    powers_of_5 = [5, 25]
    powers_of_7 = [7]
    other_primes = [11, 13, 17, 19, 23, 29, 31, 37]
    
    # We know we can take 2, 4, 16 (not 8)
    # Let's try adding them in specific groups
    # Group 1: 2, 4, 16
    # Group 2: 3, 9
    # Group 3: 5, 25
    # Group 4: 7
    # Group 5: Other primes
    
    # Constructing the specific set we derived:
    derived_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Let's verify derived_set is valid
    prods = {1}
    valid = True
    for c in derived_set:
        new_prods = {p * c for p in prods}
        if not new_prods.isdisjoint(prods):
            valid = False
            break
        prods |= new_prods
    
    if valid:
        best_set = derived_set
    
    # Try to improve by swapping or adding?
    # Can we add 12 to derived_set?
    # 12 = 2^2 * 3.
    # Let's check if we can swap 16 for something else to allow more?
    # Unlikely to exceed 16.
    
    # Let's run a quick greedy on remaining numbers to see if any fit into derived_set
    # Actually, derived_set is likely optimal or near optimal.
    
    # To be safe, run a greedy search on a shuffled list of candidates 
    # and see if we can beat 16.
    
    import random
    random.seed(42)
    
    best_len = len(best_set)
    
    for _ in range(10):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        s, _ = try_greedy(shuffled)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s
            
    return list(best_set)
# EVOLVE-BLOCK-END
