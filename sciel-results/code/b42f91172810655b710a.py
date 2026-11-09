import time
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N = 256
    # 1. Find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Map numbers 2..256 to indices 0..254
    # vars correspond to numbers 2, 3, ..., 256
    nums = list(range(2, N + 1))
    n_vars = len(nums)
    
    # 3. Build constraint matrix
    # A[p, v] = 1 if prime p divides nums[v]
    n_primes = len(primes)
    A = np.zeros((n_primes, n_vars), dtype=int)
    
    # Precompute prime factors for each number for speed, or just check divisibility
    # Divisibility check is fine for 256*54 ~ 13k ops
    for v, n in enumerate(nums):
        for p_idx, p in enumerate(primes):
            if n % p == 0:
                A[p_idx, v] = 1
    
    # 4. Define ILP
    # Maximize sum(n * x_n) => Minimize sum(-n * x_n)
    c = np.array([-n for n in nums], dtype=float)
    
    # Constraints: A @ x <= 1
    constraint = LinearConstraint(A, lb=0, ub=1)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all integer
    integrality = np.ones(n_vars, dtype=int)
    
    # Solve
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        # Fallback to simple greedy if solver fails (unlikely)
        # Greedy: sort by value desc, add if coprime to all selected
        selected = [1]
        for n in sorted(nums, reverse=True):
            if all(np.gcd(n, s) == 1 for s in selected):
                selected.append(n)
        return [int(x) for x in selected]
    
    x = res.x
    # x is numpy array, convert to list of ints
    selected = [1]
    for i, val in enumerate(x):
        if val > 0.5: # binary check
            selected.append(nums[i])
            
    # Ensure types are Python int
    return [int(s) for s in selected]
