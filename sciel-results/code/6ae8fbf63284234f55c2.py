import numpy as np
import time
from math import isqrt

def construct():
    # Deadline guard setup
    start_time = time.time()
    
    # Fallback solution: Primes + 1
    # This is a valid, if not optimal, solution.
    N = 256
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(N) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    primes = [i for i, prime in enumerate(sieve) if prime]
    fallback_result = [1] + primes
    
    try:
        # Import inside to handle potential missing SciPy gracefully
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # 1. Identify primes <= 256
        # (Reusing primes computed above)
        
        # 2. Prepare items (2..256)
        items = list(range(2, N + 1))
        n_items = len(items)
        n_primes = len(primes)
        
        # 3. Build constraint matrix A
        # A[p, i] = 1 if items[i] is divisible by primes[p]
        A = np.zeros((n_primes, n_items), dtype=float)
        
        # Map prime to index for quick lookup
        prime_index = {p: i for i, p in enumerate(primes)}
        
        for j, n in enumerate(items):
            temp_n = n
            for p in primes:
                if p * p > temp_n:
                    break
                if temp_n % p == 0:
                    A[prime_index[p], j] = 1.0
                    while temp_n % p == 0:
                        temp_n //= p
            if temp_n > 1:
                if temp_n in prime_index:
                    A[prime_index[temp_n], j] = 1.0
        
        # 4. Objective function: Maximize sum(items[i] * x[i])
        # milp minimizes, so we use -sum
        c = -np.array(items, dtype=float)
        
        # 5. Bounds and Integrality
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_items)
        
        # 6. Constraints: For each prime, sum of selected multiples <= 1
        # A x <= 1  =>  0 <= A x <= 1
        cons = LinearConstraint(A, 0, 1)
        
        # 7. Solve
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        # 8. Process result
        if res.status == 0:  # Success
            x = res.x
            selected = []
            for i in range(n_items):
                if x[i] > 0.5:
                    selected.append(items[i])
            
            # Add 1
            result = [1] + selected
            result.sort()
            
            # Ensure types are Python int
            return [int(v) for v in result]
        else:
            return [int(v) for v in fallback_result]
            
    except Exception:
        return [int(v) for v in fallback_result]
