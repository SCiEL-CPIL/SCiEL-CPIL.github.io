import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    def factorize(n):
        """Return dict of prime -> exponent for n."""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    def exponent_vector(n, primes):
        """Return tuple of exponents for n in terms of given primes."""
        vec = []
        temp = n
        for p in primes:
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            vec.append(count)
        if temp != 1:
            # n has a prime factor not in our list
            vec.append(temp)  # This shouldn't happen for n <= 40
        return tuple(vec)
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def check_valid(subset):
        """Check if all subset products are distinct."""
        prods = {1}
        for x in subset:
            new_prods = set()
            for p in prods:
                np = p * x
                if np in prods:
                    return False
                new_prods.add(np)
            prods.update(new_prods)
        return True
    
    def check_valid_fast(subset):
        """Faster check using a set of products built incrementally."""
        prods = {1}
        for x in subset:
            new_prods = []
            for p in prods:
                np = p * x
                if np in prods:
                    return False
                new_prods.append(np)
            prods.update(new_prods)
        return True
    
    # Start with a known good 16-element set
    best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    
    # Verify it's valid
    if not check_valid_fast(best_set):
        # Fallback to smaller known valid set
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_size = len(best_set)
    
    # Try to add elements one by one
    all_candidates = list(range(2, 41))
    
    # Greedy: try to add each candidate
    current = list(best_set)
    for c in all_candidates:
        if c not in current:
            test_set = current + [c]
            if check_valid_fast(test_set):
                current = test_set
    
    if len(current) > best_size:
        best_set = current
        best_size = len(current)
    
    # Randomized search: try different orderings and swaps
    rng = random.Random(42)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Try random greedy construction
        candidates = all_candidates[:]
        rng.shuffle(candidates)
        current = []
        for c in candidates:
            if len(current) >= best_size:
                # Only add if it improves
                test_set = current + [c]
                if check_valid_fast(test_set):
                    current = test_set
            else:
                test_set = current + [c]
                if check_valid_fast(test_set):
                    current = test_set
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
        
        # Local search: try swaps
        if len(current) == best_size:
            for i in range(len(current)):
                for c in all_candidates:
                    if c not in current:
                        test_set = current[:i] + current[i+1:] + [c]
                        if check_valid_fast(test_set):
                            current = test_set
                            break
    
    # Final: try to add any remaining candidates to best_set
    final = list(best_set)
    for c in all_candidates:
        if c not in final:
            test_set = final + [c]
            if check_valid_fast(test_set):
                final = test_set
    
    if len(final) > len(best_set):
        best_set = final
    
    return list(best_set)
