"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    N = 256
    candidates = list(range(2, N + 1))
    
    # Helper to check validity efficiently
    # S is a list of ints.
    # We check if a can be added to S.
    # Conditions:
    # 1. For all b, c in S, b != c: a does not divide b*c
    # 2. For all b, c in S, b != c: b does not divide a*c
    # 3. For all b, c in S, b != c: c does not divide a*b
    
    def is_valid_addition(S, a):
        n = len(S)
        if n < 2:
            return True
        
        # Check 1: a | b*c
        # If a has a prime factor p, then b*c must not be divisible by a.
        # Simple check: (b*c) % a != 0
        
        # Check 2 & 3: b | a*c and c | a*b
        # This is equivalent to: for all b in S, b does not divide a*c for any c in S, c!=b.
        # And for all c in S, c does not divide a*b for any b in S, b!=c.
        
        # Optimization:
        # Precompute a*c % b == 0?
        
        for i in range(n):
            b = S[i]
            for j in range(n):
                if i == j:
                    continue
                c = S[j]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True

    def greedy(order):
        S = []
        for a in order:
            if is_valid_addition(S, a):
                S.append(a)
        return S

    # Heuristic: Start with largest prime powers
    # For each prime p, take p^k where p^k <= 256 < p^(k+1)
    primes = [x for x in candidates if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    
    base_set = []
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        base_set.append(power)
        
    # Sort base_set? No, just use it as initial S
    S = list(base_set)
    
    # Remaining candidates
    remaining = [x for x in candidates if x not in S]
    
    # Sort remaining by "danger" or just value?
    # Numbers with large prime factors are safer.
    # Let's sort by largest prime factor descending.
    def largest_prime_factor(n):
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
        
    remaining.sort(key=largest_prime_factor, reverse=True)
    
    # Add remaining greedily
    for a in remaining:
        if is_valid_addition(S, a):
            S.append(a)
            
    best_S = S
    
    # Try a few random restarts with different initial sets?
    # Maybe just randomize the order of 'remaining'
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < 45:
        iterations += 1
        # Shuffle remaining
        rem_copy = remaining[:]
        random.shuffle(rem_copy)
        
        S_rand = list(base_set)
        for a in rem_copy:
            if is_valid_addition(S_rand, a):
                S_rand.append(a)
                
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    return best_S
# EVOLVE-BLOCK-END
