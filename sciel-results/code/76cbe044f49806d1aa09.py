import time
import random

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 45
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Determine max exponent for each prime in numbers 2..40
    # 2^5=32, 2^6=64 > 40
    # 3^3=27, 3^4=81 > 40
    # 5^2=25, 5^3=125 > 40
    # 7^1=7, 7^2=49 > 40
    # Others are 1
    max_exps = [5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # Bases must be > max_exp
    bases = [m + 1 for m in max_exps]
    
    # Precompute encoded vectors for 2..40
    # code[n] is the integer representation of the prime exponent vector of n
    codes = [0] * 41
    for n in range(2, 41):
        temp = n
        code = 0
        mult = 1
        for i, p in enumerate(primes):
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            code += exp * mult
            mult *= bases[i]
        codes[n] = code
    
    candidates = list(range(2, 41))
    best_set = []
    best_len = 0
    
    # We will use randomized greedy restarts
    # To increase diversity, we can also try different orderings
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        sums = {0}  # Set of subset sum codes
        
        for cand in shuffled:
            v = codes[cand]
            
            # Check for collision
            # If any s + v is in sums, collision
            collision = False
            # Optimization: iterate over sums
            # Using a set of integers is fast
            for s in sums:
                if (s + v) in sums:
                    collision = True
                    break
            
            if not collision:
                current_set.append(cand)
                # Update sums
                # We need to add s+v for all s in sums
                # To avoid modifying while iterating, create a list of new sums
                # sums is a set, so we can iterate over a copy or create new list
                # Since we are adding to the set, we must not modify it during iteration
                # So we create a list of the current sums
                current_sums_list = list(sums)
                for s in current_sums_list:
                    sums.add(s + v)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    if best_len == 18:
                        # Max possible? 18 is the upper bound in prompt.
                        # If we hit 18, we can return immediately
                        return list(best_set)

    # Fallback
    if not best_set:
        # Return primes as a safe fallback (size 12)
        return primes
        
    return list(best_set)
