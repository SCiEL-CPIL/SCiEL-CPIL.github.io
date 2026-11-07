import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    n = 256
    all_nums = list(range(2, n + 1))
    
    # Precompute: for each a, which pairs (b,c) with b<c, b!=a, c!=a have a|b*c
    # But that's expensive. Instead, for each a, find which other numbers "conflict"
    # Actually, the constraint is on triples. Let's think differently.
    
    # A number a is "safe" in set S if for all distinct b,c in S\{a}, a does not divide b*c.
    
    # Greedy removal: start with all numbers, remove the one that causes most conflicts
    # A conflict for a is a pair (b,c) in S\{a} where a|b*c
    
    # Better approach: use the fact that if we keep numbers with large prime factors,
    # they're less likely to divide products of others.
    
    # Let me try: start with all numbers, iteratively remove the element with highest
    # "conflict degree" = number of pairs (b,c) in current set (excluding a) where a|b*c
    
    def conflict_count(a, S_set):
        """Count pairs (b,c) in S_set, b!=c, b!=a, c!=a, where a | b*c"""
        S_list = [x for x in S_set if x != a]
        count = 0
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                b = S_list[i]
                c = S_list[j]
                if (b * c) % a == 0:
                    count += 1
        return count
    
    # This is O(n^2) per check, and we do it n times per iteration, so O(n^3) per iteration.
    # With n=255, that's ~16M operations per iteration. Too slow for many iterations.
    
    # Alternative: use a smarter approach.
    # Key insight: if a has a prime factor p with exponent e, and b*c has p^e or more, 
    # then p^e | b*c. For a|b*c, ALL prime power factors of a must divide b*c.
    
    # Let me try a different strategy: 
    # Numbers that are prime powers of large primes, or products of distinct large primes, are safer.
    # Let's try to construct a set where each number has a "unique" large prime factor.
    
    # Actually, let me try the greedy removal with optimization:
    # Precompute for each a, the list of b where gcd(a,b) > 1 or something useful.
    
    # Better: for each a, precompute which b have the property that for some c, a|b*c.
    # a|b*c iff for each prime p^e || a, the total exponent of p in b*c is >= e.
    
    # Let me try a completely different approach: 
    # Start with a candidate set (e.g., all numbers > 128, or primes, etc.)
    # Then try to add more elements greedily.
    
    # Actually, let me try: the set of all primes is valid with size ~54 (primes up to 256).
    # Can we do better? What about including some composites?
    
    # A composite a = p*q (p,q distinct primes) can divide b*c if b has p and c has q, 
    # or b has p*q, etc. So if we include a=p*q, we must ensure no pair (b,c) in S\{a} 
    # has p|b and q|c (or p|c and q|b, or p*q|b, etc.)
    
    # This is complex. Let me try a heuristic:
    # 1. Start with all primes
    # 2. Try to add composite numbers in order of "safety" (numbers with large prime factors)
    # 3. For each candidate, check if it can be added without violating the constraint
    
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [i for i in range(2, n + 1) if is_prime(i)]
    composites = [i for i in range(2, n + 1) if not is_prime(i)]
    
    # Sort composites by "safety": numbers with larger smallest prime factor are safer
    # Or numbers that are prime powers of large primes
    def smallest_prime_factor(x):
        if x % 2 == 0:
            return 2
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return i
        return x  # x is prime
    
    # Sort composites by descending smallest prime factor (larger SPF = safer)
    composites.sort(key=lambda x: -smallest_prime_factor(x))
    
    best_set = set(primes)
    best_size = len(best_set)
    
    def can_add(a, S_set):
        """Check if a can be added to S_set without violating the constraint"""
        S_list = list(S_set)
        # Check: for all distinct b,c in S_set, a does not divide b*c
        # And: for all distinct b,c in S_set, b does not divide a*c
        # And: for all distinct b,c in S_set, c does not divide a*b
        # i.e., the new element a must not create any bad triple with two existing elements,
        # and the existing elements must not create bad triples involving a.
        
        # Actually, we need to check ALL triples involving a:
        # For distinct b,c in S_set: 
        #   a | b*c ?  (a as the divisor)
        #   b | a*c ?  (b as the divisor)  
        #   c | a*b ?  (c as the divisor)
        
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                b = S_list[i]
                c = S_list[j]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True
    
    # Greedily add composites
    for a in composites:
        if time.time() > deadline:
            break
        if can_add(a, best_set):
            best_set.add(a)
            best_size = len(best_set)
    
    # Now try local search: try swapping elements
    # For each element in best_set, try removing it and adding a different one
    # This is expensive, so limit it
    
    if time.time() < deadline - 5:
        # Try some random swaps
        current_set = set(best_set)
        removed = list(current_set)
        added_candidates = [x for x in all_nums if x not in current_set]
        
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            if not removed or not added_candidates:
                break
            r = random.choice(removed)
            c = random.choice(added_candidates)
            
            new_set = set(current_set)
            new_set.discard(r)
            new_set.add(c)
            
            # Quick check: is new_set valid?
            # This is expensive, so only do it if promising
            # For now, skip full validation and just track size
            # Actually, we need to validate. Let's do a quick check.
            
            # For now, just try adding without removal (already done above)
            pass
    
    # Final validation
    def is_valid(S):
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                for k, c in enumerate(S_list):
                    if i == k or j == k:
                        continue
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not is_valid(best_set):
        # Fallback to primes
        return list(primes)
    
    result = list(best_set)
    result.sort()
    return result
