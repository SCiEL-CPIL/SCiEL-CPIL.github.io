import time
import random

def _is_valid_subset_products(nums):
    k = len(nums)
    if k == 0:
        return True
    
    # Use a set to track products
    # Start with empty product 1
    prods = {1}
    for n in nums:
        new_prods = set()
        for p in prods:
            new_prods.add(p * n)
        # Check if any new product already exists
        # Actually, if we just add to the set, we lose the ability to distinguish
        # if a product was generated in this step or a previous one.
        # But we just need to know if the TOTAL set of products is distinct.
        # So we can just update the set and check size.
        # However, if p1*n == p2*n, then p1==p2. Since prods are distinct, p*n are distinct for fixed n.
        # But p1*n might equal p2 (from previous step).
        # So we must check intersection.
        
        # Efficient check:
        # The new products are {p*n for p in prods}.
        # If any of these are in prods, we have a collision.
        # Also, if p1*n == p2*n, impossible since p1!=p2.
        # So we just need to check if (prods * n) intersects prods.
        
        # But wait, we also need to add them to prods for the next iteration.
        # So:
        # new_prods = {p * n for p in prods}
        # if new_prods & prods: return False
        # prods |= new_prods
        
        # This is O(2^k).
        
        pass

    # The above loop is correct.
    # Let's implement it.
    
    prods = {1}
    for n in nums:
        new_prods = {p * n for p in prods}
        if new_prods & prods:
            return False
        prods |= new_prods
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    all_nums = list(range(2, 41))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_set = set(primes)
    best_size = len(best_set)
    
    # Helper to try to expand a set
    def try_expand(s):
        s = set(s)
        candidates = [n for n in all_nums if n not in s]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in candidates[:]:
                if time.time() > deadline:
                    break
                new_s = s | {x}
                if _is_valid_subset_products(list(new_s)):
                    s = new_s
                    candidates.remove(x)
                    improved = True
        return s

    # Initial greedy from primes
    best_set = try_expand(best_set)
    best_size = len(best_set)
    
    # Random restarts
    for i in range(100):
        if time.time() > deadline:
            break
        
        # Random start
        size = random.randint(5, 15)
        start_set = set(random.sample(all_nums, size))
        
        # Validate and fix start set
        # If invalid, remove random elements until valid
        while not _is_valid_subset_products(list(start_set)):
            if len(start_set) > 1:
                start_set.remove(random.choice(list(start_set)))
            else:
                break
        
        # Expand
        new_set = try_expand(start_set)
        
        if len(new_set) > best_size:
            best_set = new_set
            best_size = len(best_set)
            
    return [int(x) for x in best_set]
