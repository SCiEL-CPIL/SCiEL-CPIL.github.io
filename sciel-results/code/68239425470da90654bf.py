"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # 1. Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors
    
    def is_compatible(a, b):
        if a == b:
            return False
        return not (prime_factors[a] & prime_factors[b])
    
    # 2. Build conflict graph
    # vertices are 1..256
    # edge between u, v if gcd(u,v) > 1
    conflict = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if gcd(u, v) > 1:
                conflict[u].add(v)
                conflict[v].add(u)
                
    # Precompute compatibility for speed
    # compatible[u] is a set of v such that u and v are compatible
    compatible = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(1, N + 1):
            if u != v and u not in conflict[v]:
                compatible[u].add(v)
    
    # 3. Maximum Weight Independent Set via Branch and Bound
    # We want to select a subset S of {1..256} such that no two elements in S share a prime factor,
    # maximizing sum(S).
    
    # Greedy initial solution
    def greedy_solution():
        selected = set()
        total = 0
        # Try multiple orderings
        best_set = set()
        best_sum = 0
        
        # Order 1: descending value
        for v in sorted(range(1, N + 1), key=lambda x: -x):
            if all(v in compatible[u] for u in selected):
                selected.add(v)
                total += v
        if total > best_sum:
            best_sum = total
            best_set = set(selected)
        
        # Order 2: ascending value
        selected = set()
        total = 0
        for v in sorted(range(1, N + 1)):
            if all(v in compatible[u] for u in selected):
                selected.add(v)
                total += v
        if total > best_sum:
            best_sum = total
            best_set = set(selected)
            
        # Order 3: random
        for _ in range(50):
            selected = set()
            total = 0
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for v in candidates:
                if all(v in compatible[u] for u in selected):
                    selected.add(v)
                    total += v
            if total > best_sum:
                best_sum = total
                best_set = set(selected)
                
        return best_set, best_sum
    
    best_set, best_sum = greedy_solution()
    
    # Recursive B&B
    # candidates: set of numbers that can still be added
    # current_set: set of numbers already selected
    # current_sum: sum of current_set
    
    # Optimization: Use a tighter bound.
    # A simple valid upper bound is current_sum + sum(candidates).
    # A tighter bound: For each prime p, we can pick at most one number from candidates divisible by p.
    # However, numbers can be divisible by multiple primes.
    # Let's stick to the sum bound but with good branching.
    
    # We will use a bitmask or list representation if possible, but sets are okay for N=256.
    # To speed up, we can pre-sort candidates or use a specific branching strategy.
    
    def branch_and_bound(candidates, current_set, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        # Pruning
        # Upper bound: current_sum + sum(candidates)
        # We can improve this slightly.
        # If we pick a candidate v, we remove all its conflicts.
        # Let's just use the sum bound.
        if current_sum + sum(candidates) <= best_sum:
            return
            
        # Branching variable selection:
        # Pick the candidate with the largest value? Or the one with the most conflicts?
        # Usually, picking the one with the highest value is good for finding high sums early.
        # Picking the one with the most constraints (conflicts) is good for pruning.
        # Let's try picking the candidate with the highest value.
        
        # Optimization: Instead of max(candidates) which is O(N), we can maintain a sorted list or heap.
        # But for simplicity and given N=256, max() is acceptable if the tree is pruned well.
        
        # Let's try a heuristic: pick the candidate v that maximizes (value * (1 - degree/candidates_size))?
        # Or just max value.
        
        v = max(candidates)
        
        # Option 1: Include v
        # New candidates: candidates that are compatible with v
        new_candidates = set()
        for u in candidates:
            if u != v and u in compatible[v]:
                new_candidates.add(u)
                
        current_set.add(v)
        branch_and_bound(new_candidates, current_set, current_sum + v)
        current_set.remove(v)
        
        # Option 2: Exclude v
        candidates.remove(v)
        branch_and_bound(candidates, current_set, current_sum)
        candidates.add(v)
        
    # Initial candidates: all numbers 1..N
    initial_candidates = set(range(1, N + 1))
    current_set = set()
    
    # Run B&B
    branch_and_bound(initial_candidates, current_set, 0)
    
    # Final check
    if time.time() > deadline:
        pass
        
    # Ensure types are correct
    result = [int(x) for x in best_set]
    
    # Verify pairwise coprime (sanity check)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to primes + 1
                primes = [p for p in range(2, N + 1) if not prime_factors[p] or len(prime_factors[p]) == 1]
                # Wait, prime_factors logic: prime_factors[p] is empty for p=1, and for primes p, it contains {p}.
                # So primes are those with len(prime_factors[p]) == 1 and p > 1.
                # Actually, let's just use the sieve from the original or simple check.
                is_prime = [True] * (N + 1)
                is_prime[0] = is_prime[1] = False
                for i in range(2, int(N**0.5) + 1):
                    if is_prime[i]:
                        for j in range(i*i, N + 1, i):
                            is_prime[j] = False
                primes = [p for p in range(2, N + 1) if is_prime[p]]
                result = [1] + primes
                
    return result

# EVOLVE-BLOCK-END
