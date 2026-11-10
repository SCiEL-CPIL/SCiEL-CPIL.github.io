import time
import random

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    Uses vector encoding of prime exponents for fast collision detection.
    """
    deadline = time.time() + 45  # Safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Assign bit shifts to each prime. 
    # We need enough bits to hold the maximum possible sum of exponents.
    # Max exponent for 2 in any single number is 5 (32).
    # In a subset of 18 numbers, the sum of exponents for 2 could be high.
    # Let's estimate: If we pick 18 numbers, the max sum of exponents for 2 is 
    # roughly sum of top 18 exponents. 
    # Exponents of 2 in 2..40:
    # 2:1, 4:2, 6:1, 8:3, 10:1, 12:2, 14:1, 16:4, 18:1, 20:2, 22:1, 24:3, 
    # 26:1, 28:2, 30:1, 32:5, 34:1, 36:2, 38:1, 40:3.
    # Sum of all is 1+2+1+3+1+2+1+4+1+2+1+3+1+2+1+5+1+2+1+3 = 39.
    # So 6 bits (0-63) is safe.
    BITS_PER_PRIME = 6
    MASK = (1 << BITS_PER_PRIME) - 1
    
    # Precompute codes for each number 2..40
    # code is an integer where bits [i*BITS_PER_PRIME : (i+1)*BITS_PER_PRIME] 
    # store the exponent of primes[i]
    codes = [0] * 41
    for n in range(2, 41):
        x = n
        c = 0
        for i, p in enumerate(primes):
            exp = 0
            while x % p == 0:
                x //= p
                exp += 1
            c |= (exp << (i * BITS_PER_PRIME))
        codes[n] = c
    
    candidates = list(range(2, 41))
    best_set = []
    best_len = 0
    best_codes = []
    
    def is_valid_addition(cand_code, product_codes):
        """
        Check if adding cand_code to the set represented by product_codes
        keeps all subset products distinct.
        product_codes is a set of integer codes representing all subset products.
        """
        # New products are {p + cand_code for p in product_codes}
        # Collision if any (p + cand_code) is already in product_codes
        for p in product_codes:
            if (p + cand_code) in product_codes:
                return False
        return True

    def try_greedy(order):
        """Try to build a valid set by greedily adding candidates in given order."""
        current_set = []
        current_codes = []
        product_codes = {0}  # 0 represents the empty product (exponent vector 0)
        
        for cand in order:
            cand_code = codes[cand]
            if is_valid_addition(cand_code, product_codes):
                current_set.append(cand)
                current_codes.append(cand_code)
                # Update product codes
                # new_prods = [p + cand_code for p in product_codes]
                # product_codes.update(new_prods)
                # Optimization: list comprehension
                product_codes.update(p + cand_code for p in product_codes)
                
        return current_set, current_codes, product_codes

    def local_search(current_set, current_codes, product_codes, remaining_candidates, time_limit):
        """Try to improve the set by adding elements and doing 1-for-1 swaps."""
        if not current_set:
            return current_set, current_codes, product_codes
        
        best = current_set[:]
        best_codes = current_codes[:]
        best_prods = set(product_codes)
        
        remaining = [c for c in remaining_candidates if c not in best]
        
        # Try to add any remaining candidate
        for cand in remaining:
            if time.time() > time_limit:
                break
            cand_code = codes[cand]
            if is_valid_addition(cand_code, best_prods):
                best.append(cand)
                best_codes.append(cand_code)
                best_prods.update(p + cand_code for p in best_prods)
                remaining = [c for c in remaining if c != cand]
        
        # Try 1-for-1 swaps
        if time.time() < time_limit - 3:
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    removed = best[i]
                    removed_code = best_codes[i]
                    
                    # Temporarily remove
                    temp_set = best[:i] + best[i+1:]
                    temp_codes = best_codes[:i] + best_codes[i+1:]
                    
                    # Recalculate product codes for temp_set
                    # This is expensive. 
                    # Alternative: Keep track of products?
                    # For now, recompute. It's O(2^k).
                    temp_prods = {0}
                    for c in temp_codes:
                        temp_prods.update(p + c for p in temp_prods)
                    
                    found = False
                    for cand in remaining:
                        if time.time() > time_limit:
                            break
                        if cand == removed:
                            continue
                        cand_code = codes[cand]
                        if is_valid_addition(cand_code, temp_prods):
                            # Swap successful
                            best = temp_set + [cand]
                            best_codes = temp_codes + [cand_code]
                            best_prods = set(temp_prods)
                            best_prods.update(p + cand_code for p in temp_prods)
                            remaining = [c for c in remaining if c != cand]
                            remaining.append(removed)
                            found = True
                            improved = True
                            break
                    
                    if not found:
                        # Restore if no swap found? 
                        # Actually, we only update best if found.
                        # If not found, we don't change best.
                        pass
        
        return best, best_codes, best_prods

    num_restarts = 20000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r % 4 == 0:
            order = candidates[:]
            random.shuffle(order)
        elif r % 4 == 1:
            # Sort by number of prime factors (fewer = better)
            def omega(n):
                count = 0
                d = 2
                while d * d <= n:
                    if n % d == 0:
                        count += 1
                        while n % d == 0:
                            n //= d
                    d += 1
                if n > 1:
                    count += 1
                return count
            order = sorted(candidates, key=lambda x: (omega(x), -x))
            random.shuffle(order[::2])
        elif r % 4 == 2:
            order = sorted(candidates, reverse=True)
            random.shuffle(order[::2])
        else:
            order = sorted(candidates)
            random.shuffle(order[::3])
        
        current_set, current_codes, current_prods = try_greedy(order)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            best_codes = current_codes[:]
            
            if time.time() < deadline - 5:
                improved_set, improved_codes, improved_prods = local_search(
                    best_set, best_codes, current_prods, candidates, deadline - 5
                )
                if len(improved_set) > best_len:
                    best_len = len(improved_set)
                    best_set = improved_set[:]
                    best_codes = improved_codes[:]
    
    if not best_set:
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Verify the best_set is valid before returning
    # We already have best_codes. We can verify by checking distinctness of subset sums of best_codes.
    # But we trust the search. Let's do a quick check.
    k = len(best_set)
    if k > 18:
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    all_prods = {0}
    for c in best_codes:
        all_prods.update(p + c for p in all_prods)
        
    if len(all_prods) != (1 << k):
        # Fallback
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return list(best_set)
