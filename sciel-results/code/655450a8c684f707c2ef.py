import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    N = 256
    
    # 1. Generate primes up to N
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(N)
    num_primes = len(primes)
    
    # 2. Build constraint matrix A
    # Rows: primes, Cols: numbers 1..N
    # A[p_idx, num_idx] = 1 if primes[p_idx] divides (num_idx+1)
    # We use 0-based indexing for numbers in the vector, so num_idx corresponds to value num_idx+1
    A = np.zeros((num_primes, N), dtype=int)
    
    for p_idx, p in enumerate(primes):
        for multiple in range(p, N + 1, p):
            A[p_idx, multiple - 1] = 1
            
    # 3. Setup ILP
    # Minimize c^T x  <=>  Maximize sum(i * x_i)
    # c_i = -i
    c = -np.arange(1, N + 1, dtype=float)
    
    # Constraints: A @ x <= 1
    # scipy.optimize.milp expects LinearConstraint(A, lb, ub)
    # lb = -inf (or 0), ub = 1
    # Since x >= 0, A @ x >= 0 is implicit, but we specify bounds.
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    # Bounds for x: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: 0 <= A @ x <= 1
    # Actually, we just need A @ x <= 1.
    # LinearConstraint(A, -np.inf, np.ones(num_primes))
    constraints = LinearConstraint(A, lb=-np.inf, ub=np.ones(num_primes))
    
    # Integrality: all integer
    integrality = np.ones(N)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if not res.success:
            # Fallback: Greedy approach if ILP fails
            # Sort by value descending, pick if coprime with all picked
            picked = [1]
            for i in range(N, 1, -1):
                compatible = True
                for p in picked:
                    if gcd(i, p) > 1:
                        compatible = False
                        break
                if compatible:
                    picked.append(i)
            return sorted(picked)
        
        # Extract solution
        # res.x is the array of x values
        # Note: res.x might contain values like 0.9999999, so round them
        x = np.round(res.x).astype(int)
        
        # Collect indices where x == 1
        selected = [i + 1 for i in range(N) if x[i] == 1]
        
        # Ensure 1 is included (it should be, but just in case)
        if 1 not in selected:
            selected.append(1)
            
        # Check time
        if time.time() - start_time > 50:
             # This shouldn't happen, but if it does, return what we have
             pass
             
        return selected

    except Exception as e:
        # Fallback: Greedy approach if ILP throws exception
        picked = [1]
        for i in range(N, 1, -1):
            compatible = True
            for p in picked:
                if gcd(i, p) > 1:
                    compatible = False
                    break
            if compatible:
                picked.append(i)
        return sorted(picked)

# EVOLVE-BLOCK-END
