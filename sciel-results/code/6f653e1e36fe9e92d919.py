from math import gcd
from itertools import combinations

def construct():
    N = 256
    
    # Find all primes up to 256
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Calculate largest power of p <= 256
    prime_powers = {}
    for p in primes:
        power = 1
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Function to calculate max smooth number for a set of primes
    def max_smooth(prime_set):
        # prime_set is a tuple of primes
        # We want max n <= 256 such that all prime factors of n are in prime_set
        # We can just iterate n from 256 down to 1
        # This is fast enough for small sets
        for n in range(N, 0, -1):
            temp = n
            valid = True
            for p in prime_set:
                while temp % p == 0:
                    temp //= p
            if temp != 1:
                continue
            return n
        return 1 # Only 1 is smooth for empty set, but we always have primes
        
    # Initialize groups: each prime is its own group
    # Group represented by tuple of primes
    groups = [ (p,) for p in primes ]
    
    # Calculate initial score
    def get_score(groups):
        score = 0
        for g in groups:
            sum_pp = sum(prime_powers[p] for p in g)
            smooth = max_smooth(g)
            score += max(sum_pp, smooth)
        return score
        
    current_score = get_score(groups)
    
    # Local Search: Try to merge pairs of groups
    improved = True
    while improved:
        improved = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                # Try merging groups[i] and groups[j]
                new_group = tuple(set(groups[i]) | set(groups[j]))
                
                # Create new list of groups
                new_groups = []
                for k in range(len(groups)):
                    if k == i or k == j:
                        continue
                    new_groups.append(groups[k])
                new_groups.append(new_group)
                
                new_score = get_score(new_groups)
                
                if new_score > current_score:
                    groups = new_groups
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
                
    # Construct the final set of integers
    selected = [1] # 1 is always good
    for g in groups:
        sum_pp = sum(prime_powers[p] for p in g)
        smooth = max_smooth(g)
        if smooth > sum_pp:
            selected.append(smooth)
        else:
            for p in g:
                selected.append(prime_powers[p])
                
    return selected
