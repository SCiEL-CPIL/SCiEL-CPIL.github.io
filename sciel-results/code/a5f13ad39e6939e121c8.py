"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s budget
    
    N = 512
    # We want to select distinct pairwise coprime integers a in 1..511
    # Score: sum 1/(512 - a)
    
    # Build conflict graph: two numbers conflict if gcd(a,b) > 1
    # We want maximum weight independent set where weight(a) = 1/(512-a)
    
    # Precompute weights
    weights = {}
    for a in range(1, 512):
        weights[a] = 1.0 / (N - a)
    
    # Build adjacency list for non-coprime pairs
    # Use a more efficient approach: for each number, find all numbers that share a factor
    # Instead of O(n^2) gcd checks, use prime factorization
    
    # First, get all primes up to 511
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    # Factorize each number 1..511 and build conflict sets
    # For efficiency, group numbers by their smallest prime factor
    conflicts = set()  # frozenset({a, b}) where gcd(a,b) > 1
    
    # For each prime p <= 511, all multiples of p conflict with each other
    # This is efficient: for p=2, multiples are 2,4,6,...,510
    # For each prime, collect all numbers in 1..511 that are divisible by p
    numbers_by_prime = {}
    for p in primes:
        multiples = []
        for m in range(p, 512, p):
            if m >= 1 and m <= 511:
                multiples.append(m)
        if len(multiples) >= 2:
            numbers_by_prime[p] = multiples
    
    # Build adjacency: for each number, set of conflicting numbers
    adj = {a: set() for a in range(1, 512)}
    for p, multiples in numbers_by_prime.items():
        for i in range(len(multiples)):
            for j in range(i + 1, len(multiples)):
                u, v = multiples[i], multiples[j]
                adj[u].add(v)
                adj[v].add(u)
    
    # Now solve maximum weight independent set via branch and bound
    # Greedy upper bound: sort by weight descending, greedily add if no conflict with already added
    
    # Order vertices by weight descending
    vertices = sorted(range(1, 512), key=lambda x: -weights[x])
    
    best_score = 0.0
    best_set = set()
    
    # Branch and bound
    # State: (index in vertices, current_set, current_score)
    # We process vertices in order; at each step, we can include or exclude
    
    # For upper bound: from current index onward, greedily pick non-conflicting high-weight vertices
    def upper_bound(idx, current_set):
        """Greedy upper bound: from vertices[idx:], greedily add if compatible"""
        bound = 0.0
        taken = set(current_set)
        for i in range(idx, len(vertices)):
            v = vertices[i]
            if all(v not in adj.get(t, set()) for t in taken):
                bound += weights[v]
                taken.add(v)
        return bound
    
    # Use recursive branch and bound with time check
    import sys
    sys.setrecursionlimit(10000)
    
    def bb(idx, current_set, current_score):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        # Upper bound
        ub = current_score + upper_bound(idx, current_set)
        if ub <= best_score:
            return
        
        if idx >= len(vertices):
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        v = vertices[idx]
        
        # Option 1: include v
        if all(v not in adj.get(t, set()) for t in current_set):
            new_set = current_set | {v}
            bb(idx + 1, new_set, current_score + weights[v])
        
        # Option 2: exclude v
        bb(idx + 1, current_set, current_score)
    
    # Start branch and bound
    bb(0, set(), 0.0)
    
    # Convert best_set to list
    result = sorted(best_set)
    
    # Validate: pairwise coprime
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to greedy
                chosen = []
                for a in range(511, 0, -1):
                    if all(gcd(a, b) == 1 for b in chosen):
                        chosen.append(a)
                return chosen
    
    return result
# EVOLVE-BLOCK-END
