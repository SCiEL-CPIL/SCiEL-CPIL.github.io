# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n_max = 256
    
    # Generate all primes up to n_max
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(n_max)
    prime_set = set(primes)
    
    # For each number 1..n_max, get its distinct prime factors
    def prime_factors(num):
        factors = set()
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors.add(d)
                num //= d
            d += 1
        if num > 1:
            factors.add(num)
        return factors
    
    # Number 1 has no prime factors
    factors_map = {}
    for num in range(1, n_max + 1):
        if num == 1:
            factors_map[num] = set()
        else:
            factors_map[num] = prime_factors(num)
    
    # Build ILP:
    # Variables: x_i for i in 1..n_max (binary)
    # Objective: maximize sum(i * x_i)
    # Constraints: for each prime p, sum(x_i for i where p divides i) <= 1
    
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    n_vars = n_max  # variables x_1 to x_256
    
    # Objective: maximize sum(i * x_i) => minimize -sum(i * x_i)
    c = [-float(i) for i in range(1, n_max + 1)]
    
    # Constraints: for each prime p, sum of x_i over i divisible by p <= 1
    # A matrix: rows = primes, cols = numbers
    # A[p_idx, i_idx] = 1 if prime p divides number (i_idx+1), else 0
    
    n_primes = len(primes)
    A = [[0.0] * n_vars for _ in range(n_primes)]
    for p_idx, p in enumerate(primes):
        for i in range(1, n_max + 1):
            if i % p == 0:
                A[p_idx][i - 1] = 1.0
    
    # Convert to numpy
    import numpy as np
    A_np = np.array(A)
    
    # Linear constraint: A @ x <= 1 for each row
    lb = np.zeros(n_primes)
    ub = np.ones(n_primes)
    constraint = LinearConstraint(A_np, lb, ub)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Integrality: all variables binary
    integrality = np.ones(n_vars)
    
    result = milp(
        c=c,
        constraints=constraint,
        integrality=integrality,
        bounds=bounds
    )
    
    if not result.success:
        # Fallback: just return 1 and all primes
        return [1] + primes
    
    # Extract solution
    selected = []
    for i in range(n_vars):
        if result.x[i] > 0.5:
            selected.append(i + 1)
    
    # Verify: all pairs must be coprime
    # Quick check
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # Fallback to safe set
                return [1] + primes
    
    # Ensure 1 is included (it should be, but just in case)
    if 1 not in selected:
        selected.append(1)
    
    # Ensure all are Python ints in a list
    return [int(x) for x in selected]

# EVOLVE-BLOCK-END
