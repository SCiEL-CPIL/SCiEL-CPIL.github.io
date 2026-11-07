import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_MAX_N = 256

# Precompute primes
def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

_PRIMES = _sieve(_MAX_N)
_NUM_PRIMES = len(_PRIMES)

# Precompute prime factors for each number 1..256
_prime_factors = [[] for _ in range(_MAX_N + 1)]
for i in range(2, _MAX_N + 1):
    x = i
    factors = []
    for p in _PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            factors.append(p)
            while x % p == 0:
                x //= p
    if x > 1:
        factors.append(x)
    _prime_factors[i] = factors

_prime_index = {p: i for i, p in enumerate(_PRIMES)}

def construct():
    """
    Solves the maximum sum of pairwise-coprime integers from 1..256 using Integer Linear Programming.
    """
    # Variables x_2 ... x_256 (index 1 to 255 in 0-based array for nums 2..256)
    # We exclude 1 from the ILP because it has no constraints and is always optimal to include.
    n_vars = _MAX_N - 1
    if n_vars <= 0:
        return [1]
        
    # Objective: maximize sum(i * x_i) for i in 2..256
    # milp minimizes, so we use c = -i
    # The variable at index k in the array corresponds to number (k + 2)
    c = -np.arange(2, _MAX_N + 1, dtype=float)
    
    # Constraints: For each prime p, sum(x_i for i divisible by p) <= 1
    # A is _NUM_PRIMES x n_vars
    A = np.zeros((_NUM_PRIMES, n_vars), dtype=float)
    
    for num in range(2, _MAX_N + 1):
        idx = num - 2 # 0-based index in the reduced variable set
        for p in _prime_factors[num]:
            A[_prime_index[p], idx] = 1.0
            
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all variables are integer (0 or 1)
    integrality = np.ones(n_vars)
    
    # Linear Constraint: A x <= 1
    constraint = LinearConstraint(A, ub=np.ones(_NUM_PRIMES))
    
    # Options to ensure optimality and use available time
    options = {
        'mip_rel_gap': 0.0,
        'time_limit': 59.0
    }
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds, options=options)
    
    if res.status != 0: 
        # Fallback to greedy if ILP fails
        selected = [1]
        used_primes = set()
        for num in range(_MAX_N, 1, -1):
            conflict = False
            for p in _prime_factors[num]:
                if p in used_primes:
                    conflict = True
                    break
            if not conflict:
                selected.append(num)
                used_primes.update(_prime_factors[num])
        return [int(x) for x in selected]
        
    x = res.x
    result = [1] # Always include 1
    for i in range(n_vars):
        if x[i] > 0.5:
            result.append(i + 2) # Map back to original number
            
    # Convert to list of ints
    return [int(v) for v in result]
