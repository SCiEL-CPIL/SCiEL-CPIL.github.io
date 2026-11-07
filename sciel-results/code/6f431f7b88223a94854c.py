"""Program to find maximum sum of pairwise-coprime integers from 1..256."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd

P = {'n': 256}

def _build_data():
    """Build prime list and prime powers up to 256."""
    n = P['n']
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n+1) if is_prime[i]]
    
    # For each prime, find the largest prime power <= n
    prime_powers = {}
    for p in primes:
        pk = p
        while pk * p <= n:
            pk *= p
        prime_powers[p] = pk
    
    return primes, prime_powers

def construct():
    """Find maximum sum of pairwise-coprime integers from 1..256."""
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    primes, prime_powers = _build_data()
    n = P['n']
    
    # Try ILP first for exact optimal solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Include 1: 1 has no prime factors, so it never conflicts. Always include it.
        # Solve for 2..256 and add 1 to the result.
        
        candidates = list(range(2, n+1))
        num_candidates = len(candidates)
        
        all_primes = primes
        num_primes = len(all_primes)
        
        # Build constraint matrix A: num_primes x num_candidates
        # A[j][i] = 1 if prime j divides candidate i
        A = np.zeros((num_primes, num_candidates))
        for i, x in enumerate(candidates):
            for j, p in enumerate(all_primes):
                if p > x:
                    break
                if x % p == 0:
                    A[j, i] = 1
        
        # Objective: maximize sum of candidates
        c = -np.array(candidates, dtype=float)  # milp minimizes, so negate
        
        # Constraints: A @ x <= 1 for each prime
        constraints = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Solve
        res = milp(c, constraints=constraints, bounds=bounds, integrality=np.ones(num_candidates))
        
        if res.status == 0 and res.x is not None:
            selected = [candidates[i] for i in range(num_candidates) if res.x[i] > 0.5]
            result = [1] + selected
            
            # Verify
            result_set = set(result)
            if len(result_set) != len(result):
                raise ValueError("Duplicates found")
            
            if any(x < 1 or x > n for x in result):
                raise ValueError("Out of range")
            
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    if gcd(result[i], result[j]) != 1:
                        raise ValueError("Not coprime")
            
            return [int(x) for x in result]
        else:
            # ILP failed, fall through to fallback
            pass
    except Exception:
        pass
    
    # Fallback: Greedy approach with prime powers
    # For each prime, select the largest power of that prime <= 256
    result = [1]
    for p in primes:
        result.append(prime_powers[p])
    
    # Verify
    result_set = set(result)
    if len(result_set) != len(result):
        # Shouldn't happen, but return just 1
        return [1]
    
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1]
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
