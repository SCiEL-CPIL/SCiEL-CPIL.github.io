"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

_N = 256

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def _factorize(n, primes):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 1) + 1
    return factors

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Get all primes up to 256
    primes = _sieve_primes(_N)
    
    # Factorize all numbers from 2 to 256
    factorizations = {}
    for i in range(2, _N + 1):
        factorizations[i] = _factorize(i, primes)
    
    # We need to select a subset S of {1, 2, ..., 256} such that:
    # 1. Every pair in S is coprime (gcd = 1)
    # 2. Maximize the sum of elements in S
    #
    # Key insight: If two numbers share a prime factor, they cannot both be in S.
    # So we can think of this as: for each prime p, at most one selected number
    # can be divisible by p.
    #
    # This is a weighted independent set / matching-like problem.
    # For each prime p, let's consider the numbers in 2..256 that are divisible by p.
    # We can pick at most one from each such set, and if we pick one, we "use up"
    # all its prime factors.
    #
    # Actually, the constraint is: no two selected numbers can share ANY prime factor.
    # This is equivalent to: the selected numbers have disjoint prime factor sets.
    #
    # Since 1 is coprime to everything, we always want to include 1.
    #
    # Let's use a different approach: this is a maximum weight independent set in a
    # conflict graph where vertices are 2..256 and edges connect non-coprime pairs.
    # But that graph has ~255 nodes which is too large for brute force.
    #
    # Better approach: Use the fact that the problem structure is constrained by primes.
    # Numbers with prime factors from different "groups" don't conflict.
    #
    # Actually, let's think greedily with local search:
    # 1. Start with a greedy solution
    # 2. Try local improvements
    
    # Greedy: sort numbers by value descending, add if coprime to all already selected
    def is_coprime_set(num, selected):
        for s in selected:
            if math.gcd(num, s) != 1:
                return False
        return True
    
    # Build greedy solution
    selected = [1]
    for num in range(_N, 1, -1):
        if is_coprime_set(num, selected):
            selected.append(num)
    
    current_sum = sum(selected)
    
    # Try local search: try to swap or add elements
    # Represent selected as a set for fast lookup
    selected_set = set(selected)
    
    # Try adding each unselected number if it doesn't conflict
    # Then try swapping: remove one or more, add others
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try single additions
        for num in range(_N, 1, -1):
            if num not in selected_set and is_coprime_set(num, selected_set):
                selected.append(num)
                selected_set.add(num)
                current_sum += num
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one element, add one or more
        for remove_idx in range(len(selected)):
            if time.time() > deadline:
                break
            if selected[remove_idx] == 1:
                continue  # Never remove 1
            
            removed = selected[remove_idx]
            # Temporarily remove
            selected_set.remove(removed)
            selected_without = [x for x in selected if x != removed]
            
            # Try to add numbers that conflict with removed but were previously excluded
            best_addition = None
            best_gain = 0
            
            for num in range(_N, 1, -1):
                if num in selected_set or num == removed:
                    continue
                if is_coprime_set(num, selected_set):
                    gain = num - removed
                    if gain > best_gain:
                        best_gain = gain
                        best_addition = num
            
            if best_addition is not None:
                # Perform the swap
                selected.append(best_addition)
                selected_set.add(best_addition)
                current_sum += best_addition
                improved = True
            else:
                # Put it back
                selected_set.add(removed)
        
        if not improved:
            break
    
    # Ensure result is a list of Python ints
    result = [int(x) for x in selected]
    return result
# EVOLVE-BLOCK-END
