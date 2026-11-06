"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute prime factors for all numbers 1..511
    factor_cache = {}
    for n in range(1, 512):
        factor_cache[n] = _prime_factors(n)
    
    # We want to maximize sum 1/(512 - a)
    # This is equivalent to minimizing sum 1/(512 - a) in reverse, but let's think differently.
    # The value 1/(512-a) is largest when a is closest to 511.
    # a=511: 1/1 = 1
    # a=510: 1/2 = 0.5
    # a=509: 1/3 ≈ 0.333
    # etc.
    
    # Greedy approach: try to add numbers in order of decreasing value (i.e., a from 511 down to 1)
    # but check pairwise coprimality.
    # However, greedy might not be optimal. Let's try a better approach.
    
    # Since the number of pairwise coprime numbers we can pick is limited,
    # and the values drop off quickly, let's try a greedy with backtracking or local search.
    
    # First, let's do a simple greedy to get a baseline.
    def greedy_construct():
        chosen = []
        for a in range(511, 0, -1):
            if all(math.gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best_set = greedy_construct()
    best_score = sum(1.0 / (512 - a) for a in best_set)
    
    # Now try to improve via local search / simulated annealing
    # Actually, let's think about this more carefully.
    # The set of pairwise coprime numbers is constrained.
    # If we pick a composite number, it "uses up" its prime factors.
    # Two numbers are coprime iff they share no prime factors.
    
    # So the problem is: select a subset of {1, ..., 511} such that no two share a prime factor,
    # maximizing sum 1/(512-a).
    
    # This is like a weighted matching / independent set problem on a conflict graph.
    # But the graph is defined by shared prime factors.
    
    # Let's try a different approach: for each prime, we can only pick numbers that are
    # powers of that prime (or 1). Wait, no. If we pick 6 = 2*3, we can't pick any number
    # divisible by 2 or 3.
    
    # Actually, the constraint is: for any prime p, at most one selected number can be divisible by p.
    # (Except for the number 1, which has no prime factors.)
    
    # So we can think of it as: we have a set of primes. Each number a (except 1) is associated
    # with its set of prime factors. We can pick at most one number for each prime, but a number
    # with multiple primes "blocks" all those primes.
    
    # This is a complex constraint. Let's try a smarter greedy or local search.
    
    # Local search: start with greedy, then try to swap elements or add/remove.
    
    def score_of_set(s):
        return sum(1.0 / (512 - a) for a in s)
    
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if math.gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    # Try local search with swaps
    current_set = list(best_set)
    current_score = best_score
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        improved = False
        # Try adding a number
        for a in range(511, 0, -1):
            if a in current_set:
                continue
            new_set = current_set + [a]
            if is_valid(new_set):
                new_score = score_of_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping
        for i in range(len(current_set)):
            for a in range(511, 0, -1):
                if a in current_set:
                    continue
                new_set = current_set[:i] + [a] + current_set[i+1:]
                if is_valid(new_set):
                    new_score = score_of_set(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return current_set
# EVOLVE-BLOCK-END
