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
    # Variables x_1 ... x_256
    # We can ignore x_1 as it's always optimal to include it (value 1, no conflicts).
    # Let's include it in the model for simplicity, or just add it at the end.
    # If we include it, it has no constraints.
    
    n_vars = _MAX_N
    # Objective: maximize sum(i * x_i)
    # milp minimizes, so we use c = -i
    c = -np.arange(1, _MAX_N + 1, dtype=float)
    
    # Constraints: For each prime p, sum(x_i for i divisible by p) <= 1
    # A is _NUM_PRIMES x n_vars
    A = np.zeros((_NUM_PRIMES, n_vars), dtype=float)
    
    for i in range(1, _MAX_N + 1):
        for p in _prime_factors[i]:
            A[_prime_index[p], i - 1] = 1.0
            
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all variables are integer (0 or 1)
    integrality = np.ones(n_vars)
    
    # Linear Constraint: A x <= 1
    constraint = LinearConstraint(A, ub=np.ones(_NUM_PRIMES))
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.status != 0: # 0 is optimal
        # Fallback to greedy if ILP fails
        selected = []
        used_primes = set()
        for num in range(_MAX_N, 0, -1):
            if num == 1:
                selected.append(1)
                continue
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
    result = []
    for i in range(n_vars):
        if x[i] > 0.5:
            result.append(i + 1)
            
    # Ensure 1 is included (it should be, but just in case)
    if 1 not in result:
        result.append(1)
        
    # Convert to list of ints
    return [int(v) for v in result]
