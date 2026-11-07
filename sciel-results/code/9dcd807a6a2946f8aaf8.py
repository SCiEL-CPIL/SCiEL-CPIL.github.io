import math
import random
import time

def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    n = 256
    deadline = time.time() + 70  # safety margin before 90s kill
    
    primes = _get_primes(n)
    num_primes = len(primes)
    
    # Variables: x_2, x_3, ..., x_256 (255 variables)
    # Maximize sum of i*x_i
    # Constraints: for each prime p, sum(x_i for i divisible by p) <= 1
    
    # Build constraint matrix
    rows = []
    for p in primes:
        row = np.zeros(255, dtype=int)
        for i in range(2, n + 1):
            if i % p == 0:
                row[i - 2] = 1
        rows.append(row)
    
    A = np.array(rows, dtype=int)
    lb = np.zeros(num_primes)
    ub = np.ones(num_primes)
    
    # Objective: maximize sum(i * x_i) for i=2..256
    # milp minimizes, so negate
    c_obj = -np.arange(2, n + 1, dtype=float)
    
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(0, 1)
    
    try:
        result = milp(c_obj, constraints=constraints, integrality=np.ones(255), bounds=bounds)
        
        if result.status == 0:  # success
            x = result.x.astype(int)
            selected = [i for i, xi in enumerate(x) if xi == 1]
            selected.append(1)
            return selected
    except Exception:
        pass
    
    # Fallback: greedy approach
    best_sum = 0
    best_set = []
    
    for _ in range(100):
        if time.time() > deadline:
            break
        groups = list(primes)
        
        while True:
            max_gain = -1
            best_pair = None
            candidates = []
            
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    prod = groups[i] * groups[j]
                    if prod <= n:
                        gain = prod - groups[i] - groups[j]
                        if gain > max_gain:
                            max_gain = gain
                            best_pair = (i, j)
                            candidates = [(i, j)]
                        elif gain == max_gain:
                            candidates.append((i, j))
            
            if best_pair is None:
                break
            
            i, j = random.choice(candidates)
            groups[i] = groups[i] * groups[j]
            groups.pop(j)
        
        current_sum = sum(groups)
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = groups[:]
    
    if best_set:
        best_set.append(1)
        return best_set
    
    # Ultimate fallback: just primes + 1
    return primes + [1]
