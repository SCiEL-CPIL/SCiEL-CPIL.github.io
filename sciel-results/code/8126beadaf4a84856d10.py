"""Improved program: beam search with bitmask-based coprime checks."""
import time
import heapq
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin within 150s limit
    
    # Precompute gcd relationships
    # For each a in 1..511, compute which other numbers it's coprime to
    # Use bitmask for fast intersection checks
    
    # First, let's think about the structure:
    # We want to select a subset S of {1, ..., 511} such that:
    # 1. All elements are pairwise coprime
    # 2. Sum of 1/(512 - a) for a in S is maximized
    
    # Key insight: if a number has multiple prime factors, it "uses up" multiple primes.
    # A number that is a prime power p^k only uses prime p.
    # Primes themselves are very efficient.
    
    # Strategy: For each prime p < 512, we can choose at most one number from {p^k : p^k < 512}
    # Additionally, we might choose composite numbers with multiple prime factors, but those
    # use up multiple "slots" so are generally less efficient unless the weight is high.
    
    # Let's first compute all prime powers < 512
    # And also consider composites
    
    # Actually, let's use a smarter approach:
    # The constraint is pairwise coprime. This means no two selected numbers share a prime factor.
    # So we can think of it as: for each prime p < 512, we select at most one number that is divisible by p.
    
    # Numbers that are products of multiple primes use multiple prime slots.
    # For maximizing sum 1/(512-a), we prefer larger a (closer to 511).
    
    # Let's enumerate all numbers and their prime factor sets
    # Then use a beam search / branch and bound approach
    
    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Numbers 1 has no prime factors
    # Numbers 2..511 have their prime factors
    
    # Weight for each number: 1/(512 - a)
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
    
    # Greedy initial solution: pick numbers in order of decreasing weight
    # subject to pairwise coprimality
    # This gives the baseline (score 1.0)
    
    # For improvement, use beam search
    # State: (selected_set_bitmask, current_sum)
    # But 511 numbers is too many for bitmask... let's think differently.
    
    # Alternative: branch and bound on primes
    # For each prime p < 512, decide which multiple of p to select (or none)
    # But composites complicate this since they involve multiple primes.
    
    # Let's use a different approach: 
    # Since the number of primes below 512 is about 97, and each prime can "own" at most one selected number,
    # we can do a search over prime assignments.
    
    # But actually, a number like 6 = 2*3 uses both prime 2 and prime 3.
    # So it's more like a set packing problem.
    
    # Let's try a beam search where we build up the solution incrementally.
    # At each step, we consider adding a number that is coprime to all currently selected.
    
    # To make this efficient, precompute for each number a bitmask of which other numbers it conflicts with.
    # But 511 numbers means 511-bit masks, which is manageable with Python integers.
    
    # Precompute conflict masks
    # conflict[a] is a bitmask where bit b is set if gcd(a,b) > 1
    conflict = [0] * 512
    for a in range(1, 512):
        mask = 0
        for b in range(1, 512):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << b)
        conflict[a] = mask
    
    # Sort numbers by weight descending
    sorted_nums = list(range(511, 0, -1))
    sorted_nums.sort(key=lambda x: -weights[x])
    
    # Beam search
    # State: (selected_mask, current_sum, next_idx)
    # But this might be too large. Let's use a different representation.
    
    # Actually, let's use a recursive beam search:
    # At depth d, we've selected d numbers. We try adding each candidate number that is coprime to all selected.
    
    # For efficiency, maintain the set of "available" numbers (those coprime to all selected).
    
    # Let's do a simpler approach: local search / simulated annealing style
    # Start with greedy solution, then try swapping pairs.
    
    # First, get the greedy solution
    greedy = []
    for a in sorted_nums:
        if all(gcd(a, b) == 1 for b in greedy):
            greedy.append(a)
    
    greedy_score = sum(weights[a] for a in greedy)
    
    # Try local search: for each pair (a_in, a_out), check if swapping improves
    # a_in is in greedy, a_out is not in greedy
    # Check if a_out is coprime to all (greedy - {a_in})
    
    best_solution = list(greedy)
    best_score = greedy_score
    
    # Local search iterations
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i, a_in in enumerate(best_solution):
            # Try removing a_in and adding some other number
            remaining = [x for j, x in enumerate(best_solution) if j != i]
            # Find best a_out not in best_solution that is coprime to all in remaining
            best_a_out = None
            best_gain = 0
            for a_out in sorted_nums:
                if a_out in best_solution:
                    continue
                if any(gcd(a_out, b) > 1 for b in remaining):
                    continue
                gain = weights[a_out] - weights[a_in]
                if gain > best_gain:
                    best_gain = gain
                    best_a_out = a_out
            if best_a_out is not None and best_gain > 1e-12:
                best_solution[i] = best_a_out
                best_score += best_gain
                improved = True
    
    # Also try adding numbers that fit
    for a in sorted_nums:
        if a not in best_solution:
            if all(gcd(a, b) == 1 for b in best_solution):
                best_solution.append(a)
                best_score += weights[a]
    
    return best_solution
# EVOLVE-BLOCK-END
