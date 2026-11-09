import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Known good set of 16
    base_set = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    
    # Verify base set
    def is_valid(s):
        prods = {1}
        for x in s:
            new_prods = set()
            for p in prods:
                np = x * p
                if np in prods:
                    return False
                new_prods.add(np)
            prods = new_prods
        return True
    
    best_set = base_set.copy() if is_valid(base_set) else [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_size = len(best_set)
    
    # Try to find a larger set using local search
    # Strategy: start from base_set, try swapping elements to see if we can add more
    
    # All numbers from 2 to 40
    all_nums = list(range(2, 41))
    
    # Precompute prime factorizations for efficiency
    def prime_factors(n):
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
    
    # Check if a set has distinct subset products using vector representation
    def is_valid_fast(s):
        # Represent each number as a vector of prime exponents
        # Two subsets have the same product iff their exponent vectors are equal
        # So we need all subset sums of exponent vectors to be distinct
        
        # Get all primes involved
        all_primes = set()
        for x in s:
            all_primes.update(prime_factors(x).keys())
        primes = sorted(all_primes)
        prime_to_idx = {p: i for i, p in enumerate(primes)}
        
        # Convert each number to exponent vector
        vectors = []
        for x in s:
            pf = prime_factors(x)
            vec = [0] * len(primes)
            for p, e in pf.items():
                vec[prime_to_idx[p]] = e
            vectors.append(tuple(vec))
        
        # Check all 2^k subset sums
        k = len(s)
        if k > 20:
            # Too many, use randomized check
            import random
            for _ in range(1000):
                subset = set(random.sample(range(k), random.randint(0, k)))
                subset2 = set(random.sample(range(k), random.randint(0, k)))
                if subset == subset2:
                    continue
                sum1 = [0] * len(primes)
                sum2 = [0] * len(primes)
                for i in subset:
                    for j in range(len(primes)):
                        sum1[j] += vectors[i][j]
                for i in subset2:
                    for j in range(len(primes)):
                        sum2[j] += vectors[i][j]
                if sum1 == sum2:
                    return False
            return True
        
        # Exact check
        subset_sums = set()
        for mask in range(1 << k):
            vec_sum = [0] * len(primes)
            for i in range(k):
                if mask & (1 << i):
                    for j in range(len(primes)):
                        vec_sum[j] += vectors[i][j]
            if tuple(vec_sum) in subset_sums:
                return False
            subset_sums.add(tuple(vec_sum))
        return True
    
    # Try greedy addition: start with base_set, try adding each remaining number
    remaining = [x for x in all_nums if x not in best_set]
    
    # Sort by some heuristic - prefer numbers with more unique prime factors
    def uniqueness_score(x):
        pf = prime_factors(x)
        return sum(pf.values())
    
    remaining.sort(key=uniqueness_score, reverse=True)
    
    current_set = best_set.copy()
    
    for x in remaining:
        if time.time() > deadline:
            break
        test_set = current_set + [x]
        if is_valid_fast(test_set):
            current_set = test_set
    
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)
    
    # Try local search: swap elements to potentially enable adding more
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Try removing one element and adding two others
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            without_i = best_set[:i] + best_set[i+1:]
            
            # Try adding pairs from remaining
            rem = [x for x in all_nums if x not in without_i]
            rem.sort(key=uniqueness_score, reverse=True)
            
            found_improvement = False
            for a in rem:
                if time.time() > deadline:
                    break
                test1 = without_i + [a]
                if is_valid_fast(test1):
                    # Can we add one more?
                    rem2 = [x for x in all_nums if x not in test1]
                    rem2.sort(key=uniqueness_score, reverse=True)
                    for b in rem2:
                        if time.time() > deadline:
                            break
                        test2 = test1 + [b]
                        if is_valid_fast(test2):
                            if len(test2) > best_size:
                                best_set = test2
                                best_size = len(test2)
                            found_improvement = True
                            break
                if found_improvement:
                    break
            
            if not found_improvement:
                # Just try adding one
                for a in rem:
                    if time.time() > deadline:
                        break
                    test1 = without_i + [a]
                    if is_valid_fast(test1) and len(test1) > best_size:
                        best_set = test1
                        best_size = len(test1)
                        break
    
    # Final verification
    if not is_valid(best_set):
        # Fallback
        best_set = base_set.copy()
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)
