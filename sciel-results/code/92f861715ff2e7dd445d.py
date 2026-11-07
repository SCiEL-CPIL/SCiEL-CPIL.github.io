import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_start = time.time()
_DEADLINE = 85

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    max_val = 256
    primes = [p for p in range(2, max_val + 1) if _is_prime(p)]
    
    # Split primes into small (<=128) and large (>128)
    small_primes = [p for p in primes if p <= 128]
    large_primes = [p for p in primes if p > 128]
    
    # Large primes are always included as they don't conflict with anything else
    # and are the only multiples of themselves in range.
    result_set = set(large_primes)
    
    # U: Numbers in 1..256 that are NOT large primes.
    # These are composed of small primes.
    U = [x for x in range(1, max_val + 1) if x not in large_primes]
    
    # Setup ILP
    # Variables: z_i for x_i in U
    n_vars = len(U)
    c = np.array([-x for x in U])  # Minimize -sum
    
    # Constraints: For each small prime p, sum(z_i for x_i divisible by p) <= 1
    # A matrix: rows are small_primes, cols are U
    A = np.zeros((len(small_primes), n_vars))
    for i, p in enumerate(small_primes):
        for j, x in enumerate(U):
            if x % p == 0:
                A[i, j] = 1
                
    constraints = LinearConstraint(A, ub=np.ones(len(small_primes)))
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=np.ones(n_vars), bounds=bounds)
        if res.success:
            # Extract solution
            for j, x in enumerate(U):
                if res.x[j] > 0.5:
                    result_set.add(int(x))
        else:
            # Fallback to greedy if ILP fails
            raise Exception("ILP failed")
    except Exception:
        # Fallback: Greedy + Local Search
        # Initialize with largest prime powers
        current_set = set()
        # Add 1
        current_set.add(1)
        
        # Greedy: sort U by value desc
        for x in sorted(U, reverse=True):
            # Check if x is coprime to all in current_set
            ok = True
            for y in current_set:
                if gcd(x, y) != 1:
                    ok = False
                    break
            if ok:
                current_set.add(x)
        
        # Local Search: Try to swap
        improved = True
        while improved:
            improved = False
            current_list = list(current_set)
            # Try adding
            for x in U:
                if x in current_set:
                    continue
                ok = True
                for y in current_set:
                    if gcd(x, y) != 1:
                        ok = False
                        break
                if ok:
                    current_set.add(x)
                    improved = True
            # Try swapping: remove y, add x
            for y in list(current_set):
                if y == 1: continue
                # Try to find x > y such that x is coprime to current_set - {y}
                # and x > y (to ensure improvement)
                # Actually, we just want max sum.
                # Simple swap: remove y, try all x not in set, if coprime to rest, add if sum increases.
                rest = current_set - {y}
                best_x = None
                best_val = y
                for x in U:
                    if x in rest or x == y:
                        continue
                    if x <= best_val:
                        continue
                    ok = True
                    for z in rest:
                        if gcd(x, z) != 1:
                            ok = False
                            break
                    if ok:
                        best_x = x
                        best_val = x
                if best_x:
                    current_set.remove(y)
                    current_set.add(best_x)
                    improved = True
        result_set = current_set

    # Convert to list of ints
    result_list = [int(x) for x in result_set]
    return result_list
