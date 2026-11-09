import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Sieve of Eratosthenes to find primes <= N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Map prime to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Build constraint matrix A (num_primes x N)
    # A[p_idx, i] = 1 if primes[p_idx] divides numbers[i]
    # Note: numbers[i] is i+1 (since numbers is 1..N)
    A = np.zeros((num_primes, N), dtype=int)
    for i in range(N):
        val = i + 1
        for p in primes:
            if val % p == 0:
                A[prime_to_idx[p], i] = 1
    
    # Objective: Maximize sum of numbers
    # scipy.optimize.milp minimizes c @ x
    # So we minimize -sum(numbers)
    c = -np.array(numbers, dtype=float)
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * N
    
    # Integrality: All variables are binary
    integrality = np.ones(N)
    
    # Constraints: A @ x <= 1
    from scipy.optimize import LinearConstraint, milp
    
    constraint = LinearConstraint(A, lb=None, ub=1)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if not res.success:
             # Fallback to heuristic if MILP fails
             return _heuristic_fallback(numbers, primes, prime_to_idx)
        
        x = res.x
        # x should be 0 or 1
        selected = [numbers[i] for i in range(N) if x[i] > 0.5]
        
        # Sanity check: Ensure distinct and coprime
        # The MILP guarantees coprimality via constraints.
        # We just need to ensure they are ints.
        
        if time.time() > deadline:
             return _heuristic_fallback(numbers, primes, prime_to_idx)
             
        return selected
        
    except Exception:
        return _heuristic_fallback(numbers, primes, prime_to_idx)

def _heuristic_fallback(numbers, primes, prime_to_idx):
    # Greedy approach:
    # 1. Start with 1.
    # 2. Iterate numbers from N down to 2.
    # 3. If number is coprime to all selected, add it.
    # This is not optimal but valid.
    
    selected = [1]
    used_primes = set()
    
    for i in range(len(numbers) - 1, 0, -1): # Start from N down to 2
        val = numbers[i]
        # Check if val shares any prime factor with used_primes
        compatible = True
        for p in primes:
            if val % p == 0:
                if p in used_primes:
                    compatible = False
                    break
                else:
                    # We will add p to used_primes if we select val
                    # But we need to check all primes of val
                    pass
        
        # Better check:
        # Get prime factors of val
        val_factors = set()
        temp = val
        for p in primes:
            if temp % p == 0:
                val_factors.add(p)
                while temp % p == 0:
                    temp //= p
            if p * p > temp:
                break
        if temp > 1:
            val_factors.add(temp)
            
        if not (val_factors & used_primes):
            selected.append(val)
            used_primes.update(val_factors)
            
    return selected
