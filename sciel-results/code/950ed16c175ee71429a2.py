"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute prime factors for each number
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
                
    # Get all primes <= N
    primes = [i for i in range(2, N + 1) if prime_factors[i] == {i}]
    prime_index = {p: idx for idx, p in enumerate(primes)}
    
    # Objective coefficients
    c = np.array(numbers, dtype=float)
    
    # Constraints: For each prime p, sum of x_i for i divisible by p <= 1
    # We build the constraint matrix A_ub such that A_ub @ x <= b_ub
    # Number of constraints = number of primes
    # Number of variables = N
    
    A_ub = np.zeros((len(primes), N))
    for i, num in enumerate(numbers):
        for p in prime_factors[num]:
            A_ub[prime_index[p], i] = 1
            
    b_ub = np.ones(len(primes))
    
    # Bounds for x_i: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer
    integrality = np.ones(N)
    
    # Define the MILP problem
    # Minimize -c^T x  (since milp minimizes)
    res = milp(
        c=-c,
        constraints=LinearConstraint(A_ub, ub=b_ub),
        integrality=integrality,
        bounds=bounds
    )
    
    if not res.success:
        # Fallback to a simple greedy or prime-only solution if solver fails
        # This should not happen for such a small problem
        return [1] + primes
        
    # Extract solution
    x = res.x
    selected = [num for num, xi in zip(numbers, x) if xi > 0.5]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
