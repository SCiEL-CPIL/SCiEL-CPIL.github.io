import time
import numpy as np
from math import gcd, isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 256
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute prime factors for each number
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        factors = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        prime_factors[n] = factors
    
    # Precompute compatibility matrix
    compat = np.ones((MAX_N + 1, MAX_N + 1), dtype=bool)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1):
            if gcd(i, j) != 1:
                compat[i, j] = False
                compat[j, i] = False
    
    # ILP approach using scipy
    best_set = None
    best_sum = 0
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Build constraint matrix A (num_primes x MAX_N)
        A = np.zeros((num_primes, MAX_N), dtype=np.int8)
        for n in range(1, MAX_N + 1):
            for p in prime_factors[n]:
                A[prime_to_idx[p], n - 1] = 1
        
        # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=float)
        
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))
        integrality = np.ones(MAX_N)
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            if len(selected) > 0:
                # Verify validity
                valid = True
                sel_set = set(selected)
                for a in selected:
                    for b in selected:
                        if a < b and not compat[a, b]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_set = sel_set
                    best_sum = sum(selected)
    except Exception:
        pass
    
    # If ILP didn't work or gave poor result, use branch and bound with strong pruning
    if best_set is None or best_sum < 2000:
        # Initial greedy solution
        S = {1}
        for n in range(MAX_N, 1, -1):
            if all(compat[n, x] for x in S):
                S.add(n)
        best_set = S
        best_sum = sum(S)
    
    # Branch and bound with strong upper bounds
    # Upper bound: for each prime p, at most one multiple can be selected
    # So upper bound = sum over primes of max value of multiple not yet excluded
    
    # Precompute for each prime, the list of multiples sorted by value
    prime_multiples = {}
    for i, p in enumerate(primes):
        multiples = [n for n in range(p, MAX_N + 1, p)]
        multiples.sort(reverse=True)
        prime_multiples[p] = multiples
    
    def compute_upper_bound(selected_set, available_set):
        """Compute upper bound on achievable sum given current selection."""
        ub = sum(selected_set)
        # For each prime, find the best available multiple
        used_primes = set()
        for n in selected_set:
            used_primes.update(prime_factors[n])
        
        for p in primes:
            if p in used_primes:
                continue
            # Find best available multiple of p
            for m in prime_multiples[p]:
                if m in available_set and all(m in available_set or not compat[m, x] for x in selected_set):
                    # Check if m is compatible with all selected
                    if all(compat[m, x] for x in selected_set):
                        ub += m
                        break
        return ub
    
    # Use a more practical approach: iterative improvement with 1-move and 2-move local search
    # plus random restarts
    
    import random
    random.seed(42)
    
    # Precompute compatibility as a set of incompatible pairs for faster lookup
    # For each number, store which numbers are incompatible with it
    incompatible_with = [set() for _ in range(MAX_N + 1)]
    for i in range(1, MAX_N + 1):
        for j in range(i + 1, MAX_N + 1):
            if not compat[i, j]:
                incompatible_with[i].add(j)
                incompatible_with[j].add(i)
    
    def is_compatible_with_set(n, s):
        """Check if n is compatible with all elements in set s."""
        for x in s:
            if x in incompatible_with[n]:
                return False
        return True
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        
        S_list = sorted(best_set, reverse=True)
        
        # Try adding a number (most valuable move first)
        for b in range(MAX_N, 1, -1):
            if b in best_set:
                continue
            if is_compatible_with_set(b, best_set):
                best_set.add(b)
                best_sum += b
                improved = True
                break
        
        # Try 1-for-1 swaps
        for a in list(S_list):
            if a == 1:
                continue
            for b in range(MAX_N, a, -1):
                if b in best_set:
                    continue
                # Check if b is compatible with all elements except a
                temp_set = best_set - {a}
                if is_compatible_with_set(b, temp_set):
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        best_set.discard(a)
                        best_set.add(b)
                        best_sum = new_sum
                        improved = True
                        break
        
        # Try 2-for-1 swaps
        if len(best_set) >= 2:
            S_sorted = sorted(best_set)
            found_swap = False
            for i in range(len(S_sorted)):
                if found_swap:
                    break
                a = S_sorted[i]
                if a == 1:
                    continue
                for j in range(i + 1, len(S_sorted)):
                    b = S_sorted[j]
                    if b == 1:
                        continue
                    if a + b < MAX_N:
                        continue
                    temp_set = best_set - {a, b}
                    for c in range(MAX_N, max(a, b), -1):
                        if c in best_set:
                            continue
                        if is_compatible_with_set(c, temp_set):
                            new_sum = best_sum - a - b + c
                            if new_sum > best_sum:
                                best_set.discard(a)
                                best_set.discard(b)
                                best_set.add(c)
                                best_sum = new_sum
                                improved = True
                                found_swap = True
                                break
        
        # Try 1-for-2 swaps: remove one, add two smaller ones that are compatible
        if len(best_set) >= 1:
            found_swap = False
            for a in list(S_list):
                if found_swap:
                    break
                if a == 1:
                    continue
                # Find two numbers b, c not in set, b+c > a, compatible with each other and rest
                temp_set = best_set - {a}
                for b in range(MAX_N, 1, -1):
                    if found_swap:
                        break
                    if b in best_set or b == a:
                        continue
                    if not is_compatible_with_set(b, temp_set):
                        continue
                    for c in range(MAX_N, b, -1):
                        if c in best_set or c == a or c == b:
                            continue
                        if not compat[b, c]:
                            continue
                        if not is_compatible_with_set(c, temp_set):
                            continue
                        if b + c > a:
                            best_set.discard(a)
                            best_set.add(b)
                            best_set.add(c)
                            best_sum = best_sum - a + b + c
                            improved = True
                            found_swap = True
                            break
        
        # Random perturbation if stuck
        if not improved and iterations % 3 == 0:
            # Remove a random element and re-add
            if len(best_set) > 1:
                candidates = list(best_set - {1}) if 1 in best_set else list(best_set)
                if candidates:
                    random_elem = random.choice(candidates)
                    best_set.discard(random_elem)
                    best_sum -= random_elem
                    # Try to add back best compatible
                    for b in range(MAX_N, 1, -1):
                        if b in best_set:
                            continue
                        if is_compatible_with_set(b, best_set):
                            best_set.add(b)
                            best_sum += b
                            break
                    improved = True
    
    # Final verification
    result = sorted(best_set)
    if not result:
        result = [1]
    
    # Verify all pairs are coprime
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a simple valid set
        result = [1]
        for n in range(MAX_N, 1, -1):
            if all(gcd(n, x) == 1 for x in result):
                result.append(n)
    
    # Ensure all are Python ints
    return [int(x) for x in result]
