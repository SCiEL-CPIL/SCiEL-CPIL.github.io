import time
import numpy as np
from math import gcd, isqrt
import random

def construct():
    start_time = time.time()
    # Keep a safety margin. 90s limit. Use 85s for search, 5s for overhead/safety.
    deadline = start_time + 85.0

    MAX_N = 256

    # 1. Pre-compute primes and conflicts
    # We need to know which pairs (i, j) have gcd(i, j) > 1
    # Efficient way: for each prime p, mark all multiples of p as conflicting with each other.
    
    # conflict_mask[i] is a bitset of indices j such that gcd(i+1, j+1) > 1
    # But for ILP, we just need the constraints.
    
    # Let's build the constraint matrix A for the ILP directly.
    # Variables x_1, ..., x_256 (0-indexed 0 to 255)
    # Constraint for each prime p: sum(x_n for n in multiples of p) <= 1
    
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
    
    # Map prime to index for A matrix rows
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Build A matrix (num_primes x MAX_N)
    # A[p_idx, n_idx] = 1 if primes[p_idx] divides (n_idx + 1)
    A = np.zeros((num_primes, MAX_N), dtype=float)
    for n in range(1, MAX_N + 1):
        # Find prime factors of n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                A[prime_to_idx[p], n - 1] = 1.0
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime factor
            if temp in prime_to_idx:
                A[prime_to_idx[temp], n - 1] = 1.0

    # Objective: Maximize sum(n * x_n) -> Minimize -sum(n * x_n)
    c = -np.arange(1, MAX_N + 1, dtype=float)

    # Bounds: 0 <= x <= 1
    lb = np.zeros(MAX_N)
    ub = np.ones(MAX_N)
    bounds = (lb, ub)

    # Constraints: A @ x <= 1
    # In scipy.optimize.milp, LinearConstraint(A, lb, ub) means lb <= A@x <= ub
    # We want A@x <= 1, so lb=0, ub=1 for each row.
    # Note: A@x is a vector of length num_primes.
    constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))
    
    # Integrality
    integrality = np.ones(MAX_N)

    best_result = None
    best_sum = 0

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Solve ILP
        # HiGHS is the default and recommended solver.
        # We set a time limit slightly less than our deadline to allow for fallback.
        # 60s for ILP is safe.
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=lb, ub=ub),
            options={
                'time_limit': 60,
                'mip_rel_gap': 0.0, # Try for exact
                'presolve': 'on',
                'parallel': 'on'
            }
        )

        if res.x is not None:
            # Check if solution is valid (x is close to 0 or 1)
            x = res.x
            # Round to nearest integer
            x_int = np.round(x).astype(int)
            # Verify bounds and integrality roughly
            if np.all((x_int >= 0) & (x_int <= 1)):
                selected = [int(i + 1) for i in range(MAX_N) if x_int[i] == 1]
                
                # Verify pairwise coprime
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if gcd(selected[i], selected[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = selected
                    best_sum = sum(selected)
                    # If we found a solution, we can check if it's optimal
                    # res.status == 0 means optimal
                    if res.status == 0:
                        # This is the exact optimum
                        return [int(x) for x in best_result]
                    else:
                        # Not proven optimal, but we have a valid incumbent.
                        # We can try to improve with local search if time permits.
                        pass
    except Exception as e:
        # Fallback if ILP fails
        pass

    # Fallback / Polish: Local Search
    # If we didn't return from ILP (either failed or not proven optimal), we use local search.
    # Start with the ILP result if available, otherwise greedy.
    
    if best_result is None:
        # Greedy: pick largest coprime numbers
        S = []
        for n in range(256, 0, -1):
            if all(gcd(n, x) == 1 for x in S):
                S.append(n)
        best_set = set(S)
        cur_sum = sum(S)
    else:
        best_set = set(best_result)
        cur_sum = best_sum

    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(list(best_set), reverse=True)
        
        # 1. Try to add any missing number
        for b in range(256, 1, -1):
            if b in best_set:
                continue
            if all(gcd(b, x) == 1 for x in best_set):
                best_set.add(b)
                cur_sum += b
                improved = True
        
        # 2. 1-for-1 swaps
        for a in S_list:
            if a == 1:
                continue
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    new_sum = cur_sum - a + b
                    if new_sum > cur_sum:
                        best_set.remove(a)
                        best_set.add(b)
                        cur_sum = new_sum
                        improved = True
                        break # Restart loop for efficiency? Or just continue.
        
        # 3. 2-for-1 swaps (remove two small, add one big)
        if time.time() < deadline:
            S_list = sorted(list(best_set))
            for i in range(len(S_list)):
                if time.time() >= deadline:
                    break
                a1 = S_list[i]
                if a1 == 1:
                    continue
                for j in range(i + 1, len(S_list)):
                    a2 = S_list[j]
                    if a2 == 1:
                        continue
                    # Try to find a big number b that fits if a1, a2 removed
                    for b in range(256, a1 + a2, -1):
                        if b in best_set or b == a1 or b == a2:
                            continue
                        # Check if b is coprime to rest
                        if all(gcd(b, x) == 1 for x in best_set if x not in (a1, a2)):
                            best_set.remove(a1)
                            best_set.remove(a2)
                            best_set.add(b)
                            cur_sum = cur_sum - a1 - a2 + b
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                continue

        # 4. 2-for-2 swaps
        if time.time() < deadline:
            S_list = sorted(list(best_set))
            for i in range(len(S_list)):
                if time.time() >= deadline:
                    break
                a1 = S_list[i]
                if a1 == 1:
                    continue
                for j in range(i + 1, len(S_list)):
                    a2 = S_list[j]
                    if a2 == 1:
                        continue
                    for b1 in range(256, a1, -1):
                        if b1 in best_set or b1 == a1 or b1 == a2:
                            continue
                        if not all(gcd(b1, x) == 1 for x in best_set if x not in (a1, a2)):
                            continue
                        for b2 in range(256, max(a1, a2), -1):
                            if b2 in best_set or b2 == a1 or b2 == a2 or b2 == b1:
                                continue
                            if not all(gcd(b2, x) == 1 for x in best_set if x not in (a1, a2)):
                                continue
                            if gcd(b1, b2) != 1:
                                continue
                            new_sum = cur_sum - a1 - a2 + b1 + b2
                            if new_sum > cur_sum:
                                best_set.remove(a1)
                                best_set.remove(a2)
                                best_set.add(b1)
                                best_set.add(b2)
                                cur_sum = new_sum
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue

    # Final polish: add 1 if possible
    if 1 not in best_set:
        if all(gcd(1, x) == 1 for x in best_set):
            best_set.add(1)
            cur_sum += 1

    return [int(x) for x in sorted(best_set)]
