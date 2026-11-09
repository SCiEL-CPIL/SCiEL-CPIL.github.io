"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from math import gcd

def construct():
    N = 256
    
    # Generate primes up to N using Sieve of Eratosthenes
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, N + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, N + 1) if sieve[i]]
    
    num_primes = len(primes)
    
    # Build constraint matrix A
    # Rows: primes, Cols: numbers 1..N
    # A[p_idx, n_idx] = 1 if primes[p_idx] divides (n_idx + 1)
    A = np.zeros((num_primes, N), dtype=int)
    for p_idx, p in enumerate(primes):
        for n in range(p, N + 1, p):
            A[p_idx, n - 1] = 1
            
    # Objective: Maximize sum of selected numbers
    # Minimize -sum
    c = -np.arange(1, N + 1, dtype=float)
    
    constraints = LinearConstraint(A, ub=np.ones(num_primes))
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(N)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if not res.success:
            raise RuntimeError("Solver failed")
        
        selected = []
        for i in range(N):
            if res.x[i] > 0.5:
                selected.append(i + 1)
        return [int(x) for x in selected]
        
    except Exception:
        # Fallback to greedy if solver fails
        selected = []
        for k in range(N, 0, -1):
            if all(gcd(k, s) == 1 for s in selected):
                selected.append(k)
        return selected
# EVOLVE-BLOCK-END
