# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # We want to select a subset S of {1, 2, ..., 256} such that:
    # 1. Every pair in S is coprime
    # 2. The sum of S is maximized
    
    # Key insight: If a number n has a prime factor p, then no other number in S can be divisible by p.
    # For each prime p, we can choose at most one number from {multiples of p in 1..256}.
    # If we choose a composite number m = p1 * p2 * ... * pk, it "uses up" primes p1, p2, ..., pk.
    # This is a complex constraint satisfaction problem.
    
    # Simpler approach: Greedy + Local Search
    # Start with all primes and 1, then try to swap/add composites if beneficial.
    
    # Actually, the optimal solution is likely to include:
    # - 1 (it's coprime with everything)
    # - All primes > 128 (they can't share factors with any other number except multiples, but multiples of p > 128 would be > 256, so only p itself is available)
    #   Wait, multiples of p where p > 128: 2p > 256, so no multiples. So primes > 128 are "free" in the sense that they don't compete with composites.
    # - For primes p <= 128, we choose the largest multiple of p in 1..256 that is coprime with all other chosen numbers.
    
    # Let's use a greedy approach:
    # 1. Start with S = {1}
    # 2. For each number from 256 down to 2, if it's coprime with all in S, add it.
    
    # This greedy might not be optimal, but it's a good starting point.
    # Then we can try local search: swap elements to improve the sum.
    
    S = [1]
    for n in range(256, 1, -1):
        if all(gcd(n, x) == 1 for x in S):
            S.append(n)
    
    # Local search: try to replace pairs of elements with a single larger element or vice versa
    # Actually, a better approach: for each prime p, consider the multiples of p in 1..256.
    # We can choose at most one. The optimal choice for prime p, given other constraints, is complex.
    
    # Let's try a different strategy:
    # The set of numbers we choose must be such that for every prime p, at most one chosen number is divisible by p.
    # This is equivalent to: the chosen numbers are pairwise coprime.
    
    # Greedy from largest to smallest is a heuristic. Let's enhance it with local optimization.
    
    # Try all numbers from 256 down to 2 in descending order, and add if possible.
    # This gives a maximal set (can't add any more without violating coprimality).
    
    # To improve, try removing a small number and adding a larger one that becomes possible.
    
    best_set = set(S)
    best_sum = sum(S)
    
    # Local search: iterate, try to swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)
        for i, a in enumerate(S_list):
            if a == 1:
                continue
            for b in range(a + 1, 257):
                # Try to replace a with b
                if b in best_set:
                    continue
                new_set = (best_set - {a}) | {b}
                if all(gcd(x, y) == 1 for x in new_set for y in new_set if x < y):
                    new_sum = sum(new_set)
                    if new_sum > best_sum:
                        best_set = new_set
                        best_sum = new_sum
                        improved = True
                        break
            if improved and time.time() > deadline - 5:
                break
    
    # Ensure the result is a list of Python ints
    result = [int(x) for x in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
