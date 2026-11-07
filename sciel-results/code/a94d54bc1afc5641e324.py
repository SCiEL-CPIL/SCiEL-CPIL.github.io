"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    N = 256

    # Sieve to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for multiple in range(p * p, N + 1, p):
                is_prime[multiple] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Precompute for each number 1..N:
    # - its distinct prime factors (as a frozenset of primes)
    # - for each prime, the list of numbers that have that prime as a factor
    num_to_primes = {}
    prime_to_nums = {p: [] for p in primes}
    
    for n in range(1, N + 1):
        temp = n
        factors = set()
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        num_to_primes[n] = factors
        for p in factors:
            prime_to_nums[p].append(n)
    
    # For each number, precompute the set of numbers it conflicts with (shares a prime factor)
    # Two numbers conflict if they share at least one prime factor.
    # Note: 1 has no prime factors, so it conflicts with nothing.
    conflict_with = {}
    for n in range(1, N + 1):
        conflicts = set()
        for p in num_to_primes[n]:
            conflicts.update(prime_to_nums[p])
        conflicts.discard(n)
        conflict_with[n] = conflicts
    
    # Try to solve using ILP if scipy is available
    best_result = None
    best_val = -1
    
    # First, try ILP formulation
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csc_matrix
        import numpy as np
        
        # Variables: x_n for n in 1..N, x_n in {0, 1}
        # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
        # Constraints:
        # 1. For each prime p, sum of x_n for n in prime_to_nums[p] <= 1
        # 2. 0 <= x_n <= 1
        
        n_vars = N  # variables x_1, ..., x_N
        n_constraints = len(primes)  # one per prime
        
        # Objective: c = [-1, -2, ..., -N]
        c = [-float(n) for n in range(1, N + 1)]
        
        # Constraint matrix A: n_constraints x n_vars
        # A[i][j] = 1 if number (j+1) has prime primes[i] as a factor, else 0
        A = []
        for p in primes:
            row = [0] * n_vars
            for n in prime_to_nums[p]:
                row[n - 1] = 1
            A.append(row)
        
        # Bounds: 0 <= x_n <= 1
        bounds = [(0, 1)] * n_vars
        
        # A_ub @ x <= b_ub
        b_ub = [1] * n_constraints
        
        # Solve using linprog
        # Use method='highs' which is fast for small problems
        if time.time() < deadline - 5:
            result = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.status == 0 and result.fun is not None:
                # Extract solution
                solution = []
                for i in range(n_vars):
                    if result.x[i] > 0.5:  # binary variable
                        solution.append(i + 1)
                
                # Validate the solution
                if len(solution) > 0:
                    # Check pairwise coprime
                    valid = True
                    for i in range(len(solution)):
                        for j in range(i + 1, len(solution)):
                            if math.gcd(solution[i], solution[j]) != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        val = sum(solution)
                        if val > best_val:
                            best_val = val
                            best_result = solution
        
        # If ILP didn't find a good solution or failed, fall back to heuristic
    except Exception:
        pass
    
    # Heuristic fallback: greedy + local search
    if time.time() < deadline - 10:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        # Greedy by descending value
        selected = set()
        for n in range(N, 0, -1):
            if all(gcd(n, s) == 1 for s in selected):
                selected.add(n)
        
        val = sum(selected)
        if val > best_val:
            best_val = val
            best_result = list(selected)
        
        # Local search: try to add or swap
        if best_result and time.time() < deadline - 5:
            selected = set(best_result)
            current_sum = sum(selected)
            best_sum = current_sum
            best_set = set(selected)
            
            # Try adding numbers
            for n in range(1, N + 1):
                if n not in selected:
                    if all(gcd(n, s) == 1 for s in selected):
                        new_sum = current_sum + n
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_set = selected | {n}
                            selected.add(n)
                            current_sum = new_sum
            
            # Try swaps
            for n_out in list(selected):
                for n_in in range(1, N + 1):
                    if n_in not in selected:
                        new_selected = (selected - {n_out}) | {n_in}
                        if all(gcd(a, b) == 1 for a, b in zip(sorted(new_selected), sorted(new_selected)[1:])):
                            new_sum = current_sum - n_out + n_in
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_set = new_selected
                                selected = new_selected
                                current_sum = new_sum
            
            if best_sum > best_val:
                best_val = best_sum
                best_result = list(best_set)
    
    # Final validation
    if best_result:
        vals = set(best_result)
        if len(vals) == len(best_result) and all(isinstance(v, int) and 1 <= v <= N for v in best_result):
            valid = True
            for i in range(len(best_result)):
                for j in range(i + 1, len(best_result)):
                    if math.gcd(best_result[i], best_result[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return best_result
    
    # Fallback: just primes + 1
    return [1] + [p for p in primes]

# EVOLVE-BLOCK-END
