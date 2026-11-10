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
    
    # For each number 1..256, find its prime factors
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        if n == 1:
            prime_factors[n] = []
        else:
            factors = []
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    factors.append(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                factors.append(temp)
            prime_factors[n] = factors
    
    def is_feasible(S):
        """Check if all pairs in S are coprime."""
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if gcd(S[i], S[j]) != 1:
                    return False
        return True
    
    def local_search(S, time_limit=None):
        """Improve solution S via local search: try adding/removing/swapping elements."""
        if time_limit is None:
            time_limit = deadline
        S = list(S)
        S_set = set(S)
        S_sum = sum(S)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding a new number
            for n in range(MAX_N, 0, -1):
                if n in S_set:
                    continue
                # Check if n is coprime with all in S
                if all(gcd(n, x) == 1 for x in S):
                    S.append(n)
                    S_set.add(n)
                    S_sum += n
                    improved = True
                    break
            
            # Try swapping: remove one element, add one or more
            if not improved:
                for i in range(len(S)):
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_sum_temp = S_sum - removed
                    # Try adding numbers that are coprime with S_temp
                    for n in range(MAX_N, 0, -1):
                        if n == removed or n in S_temp:
                            continue
                        if all(gcd(n, x) == 1 for x in S_temp):
                            new_sum = S_sum_temp + n
                            if new_sum > S_sum:
                                S = S_temp + [n]
                                S_sum = new_sum
                                S_set = set(S)
                                improved = True
                                break
                    if improved:
                        break
        
        return S
    
    best_solution = [1]
    
    # Use ILP via scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build constraint matrix: one row per prime, one column per number 1..256
        # A[p_idx, n-1] = 1 if p divides n, else 0
        n_primes = len(primes)
        n_vars = MAX_N
        
        # Build A matrix
        A = np.zeros((n_primes, n_vars), dtype=np.int8)
        for p_idx, p in enumerate(primes):
            for multiple in range(p, MAX_N + 1, p):
                A[p_idx, multiple - 1] = 1
        
        # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=np.float64)
        
        # Constraints: A @ x <= 1 for all primes
        constraint = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_n <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: all variables are integer (binary)
        integrality = np.ones(n_vars)
        
        result = milp(c=c, constraints=constraint, bounds=bounds, integrality=integrality)
        
        if result.success:
            selected = [i + 1 for i in range(n_vars) if result.x[i] > 0.5]
            if is_feasible(selected):
                # Polish with local search
                polished = local_search(selected, time_limit=deadline - 5)
                if sum(polished) > sum(best_solution):
                    best_solution = polished
        else:
            pass
    except Exception:
        pass
    
    # Fallback: Greedy approach (improved)
    S = [1]
    for n in range(MAX_N, 1, -1):
        if all(gcd(n, x) == 1 for x in S):
            S.append(n)
    
    # Polish the greedy solution
    polished = local_search(S, time_limit=deadline - 5)
    if sum(polished) > sum(best_solution):
        best_solution = polished
    
    # Final deadline check
    if time.time() > deadline - 2:
        pass  # Return immediately
    
    # Ensure the result is a list of Python ints
    return [int(x) for x in sorted(best_solution)]
# EVOLVE-BLOCK-END
