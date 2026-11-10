import time
import numpy as np

# Global start time for deadline guard
START_TIME = time.time()
TIME_LIMIT = 85.0  # 5 seconds margin before 90s

def _check_time():
    return (time.time() - START_TIME) > TIME_LIMIT

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _greedy_fallback(n_max):
    """
    Fallback strategy: Greedy replacement.
    Start with all primes and 1.
    Iterate composites from high to low.
    Replace conflicting primes if the composite is larger than their sum.
    """
    primes = _get_primes(n_max)
    # Initialize with 1 and all primes
    current_set = {1} | set(primes)
    current_sum = 1 + sum(primes)
    
    # Generate prime factors for each number for quick check?
    # Or just use gcd.
    
    # Iterate from n_max down to 2
    # We only need to consider composites, but checking all is fine.
    for n in range(n_max, 1, -1):
        if _check_time():
            break
        
        # If n is prime, it's already in the set (unless removed, but primes are only removed if replaced by composite).
        # Actually, if a prime p was removed, it was replaced by a composite multiple of p.
        # So we just check if n is coprime to the current set.
        
        # Check if n conflicts with any element in current_set
        # And calculate the sum of conflicting elements
        conflict_sum = 0
        conflicts = []
        for s in current_set:
            if s == 1:
                continue
            if np.gcd(n, s) > 1:
                conflict_sum += s
                conflicts.append(s)
                
        # If n is better than the sum of its conflicts
        if n > conflict_sum:
            # Remove conflicts
            for c in conflicts:
                current_set.remove(c)
            # Add n
            current_set.add(n)
            current_sum = current_sum - conflict_sum + n
            
    return list(current_set)

def construct():
    n_max = 256
    
    # Try ILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        primes = _get_primes(n_max)
        k = len(primes)
        N = n_max
        
        # Objective: Minimize -sum(n * x_n)
        c = np.array([-n for n in range(1, N + 1)], dtype=float)
        
        # Constraints: For each prime p_i, sum(x_n for n divisible by p_i) <= 1
        A = np.zeros((k, N), dtype=int)
        for idx, p in enumerate(primes):
            for m in range(p, N + 1, p):
                # m is the number, index is m-1
                A[idx, m - 1] = 1
                
        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(N)
        
        # Solve
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x
            # Extract indices where x > 0.5
            result_indices = np.where(x > 0.5)[0]
            # Convert to 1-based integers
            result = [int(i + 1) for i in result_indices]
            # Ensure 1 is included? 
            # x_1 (index 0) corresponds to number 1.
            # c[0] = -1. No constraints on index 0.
            # So x[0] should be 1.
            # If for some reason it's not, we add it.
            if 1 not in result:
                result.append(1)
            return result
        else:
            raise RuntimeError("ILP failed")
            
    except Exception as e:
        # Fallback to greedy
        if _check_time():
             # If we are out of time, return the simplest valid set: primes + 1
             primes = _get_primes(n_max)
             return [1] + primes
        return _greedy_fallback(n_max)
