import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill

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

    # Precompute prime factors for each number 1..256
    prime_factors = {}
    for x in range(1, MAX_N + 1):
        n = x
        factors = set()
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            factors.add(n)
        prime_factors[x] = factors

    # For each prime p, find the largest power of p <= MAX_N
    # This gives an upper bound: for each prime, at most one number divisible by p can be chosen
    # The best choice for prime p is p^k where p^k <= MAX_N is maximized
    max_power_per_prime = {}
    for p in primes:
        max_p = p
        while max_p * p <= MAX_N:
            max_p *= p
        max_power_per_prime[p] = max_p

    # Upper bound on total sum: sum over all primes of their max power, plus 1
    # But this is loose because a number like 6 uses both 2 and 3.
    # Better upper bound: for each prime p, the maximum value we can assign to "p's slot" is max_power_per_prime[p]
    # Since each number in our set has a unique set of prime factors, and each prime factor appears in at most one number,
    # the sum is at most sum of max_power_per_prime[p] for all p, plus 1.
    
    # However, a tighter bound: the optimal solution is a partition of some subset of primes into groups,
    # where each group's product is <= 256, and we want to maximize the sum of products.
    # The upper bound is sum(max_power_per_prime[p]) + 1.
    
    ub_total = 1 + sum(max_power_per_prime.values())

    # Greedy initialization: start with all primes, merge best pairs
    slots = list(primes)
    
    # Greedily merge slots with best gain while product <= MAX_N
    while True:
        if time.time() > deadline - 5:
            break
        
        best_gain = 0
        best_i = -1
        best_j = -1
        
        n = len(slots)
        for i in range(n):
            for j in range(i + 1, n):
                merged = slots[i] * slots[j]
                if merged > MAX_N:
                    continue
                gain = merged - slots[i] - slots[j]
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        
        if best_i == -1 or best_gain <= 0:
            break
        
        slots[best_i] = slots[best_i] * slots[best_j]
        slots.pop(best_j)

    # Add 1
    result = slots + [1]
    best_sum = sum(result)
    best_set = list(result)

    # Branch and bound to find optimal solution
    # State: which primes are "used" (assigned to a number in the set)
    # We decide for each prime whether to include it in a number.
    # Actually, let's think differently: we want to choose a set of numbers from 1..256, pairwise coprime.
    # Each number corresponds to a subset of primes (its prime factors).
    # We want to choose disjoint subsets of primes (plus possibly 1) to maximize sum of products.
    
    # Let's use a different approach: branch on primes.
    # For each prime p, we decide which number in our set will contain p.
    # But that's complex. Instead, let's use the fact that the number of primes up to 256 is 54.
    # That's too many for exhaustive search.
    
    # Alternative: local search / simulated annealing on the set of chosen numbers.
    # Start with greedy solution, then try to improve by:
    # 1. Adding a new number (if coprime with all existing)
    # 2. Removing a number
    # 3. Swapping a number for another
    
    # Let's try a simpler improvement: for each number in the set, try to replace it with a larger number
    # that is coprime with all other numbers in the set.
    
    # Actually, let's try a different structural approach: 
    # The problem is equivalent to finding a maximum weight independent set in a graph where
    # vertices are 1..256, edges connect non-coprime pairs, weights are vertex values.
    # This is NP-hard, but for 256 vertices we can try branch and bound.
    
    # Let's implement a branch and bound solver.
    # Sort vertices by weight descending.
    # At each step, decide to include or exclude the current vertex.
    # Upper bound: current sum + sum of weights of remaining vertices that are compatible with included set.
    
    # This might be slow, but let's try with a time limit.
    
    # Precompute conflict graph
    conflicts = [set() for _ in range(MAX_N + 1)]
    for i in range(1, MAX_N + 1):
        for j in range(i + 1, MAX_N + 1):
            if gcd(i, j) > 1:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Branch and bound
    # Order vertices by weight descending
    vertices = list(range(1, MAX_N + 1))
    vertices.sort(key=lambda x: -x)
    
    # State: index in vertices, current set, current sum
    # We'll use iterative deepening or just a simple B&B with pruning.
    
    # Let's use a different approach: since the greedy solution is already good,
    # let's try to improve it by local search.
    
    # Local search: try to add a number not in the set, if it's coprime with all in the set.
    # Try to remove a number and add two or more numbers that are coprime with the rest.
    
    # Let's implement a simple local search:
    # 1. Try to add each number not in the set if it's coprime with all in the set.
    # 2. Try to swap each number in the set with a number not in the set if the new number is coprime with all others.
    
    def is_coprime_with_all(num, others):
        for o in others:
            if gcd(num, o) > 1:
                return False
        return True
    
    # Convert best_set to a set for fast lookup
    best_set_set = set(best_set)
    
    # Try to add numbers
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for num in range(1, MAX_N + 1):
            if num in best_set_set:
                continue
            if is_coprime_with_all(num, best_set_set):
                best_set.append(num)
                best_set_set.add(num)
                best_sum += num
                improved = True
    
    # Try swaps: remove one, add another
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(best_set)):
            removed = best_set[i]
            best_set_set.remove(removed)
            best_sum -= removed
            
            # Try to add a number that's coprime with all remaining
            for num in range(1, MAX_N + 1):
                if num in best_set_set:
                    continue
                if is_coprime_with_all(num, best_set_set):
                    # Check if this improves the sum
                    if num > removed:
                        best_set.append(num)
                        best_set_set.add(num)
                        best_sum += num
                        improved = True
                        break
            
            # Restore
            best_set_set.add(removed)
            best_sum += removed
            if improved:
                break
    
    # Verify final solution
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            if gcd(best_set[i], best_set[j]) != 1:
                # Fallback to greedy
                result = slots + [1]
                return [int(x) for x in result]

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
