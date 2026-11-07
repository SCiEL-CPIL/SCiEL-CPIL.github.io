"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Returns a list of distinct integers from 1..256 such that every pair is coprime,
    maximizing the sum.
    """
    N = 256
    
    # Step 1: Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
        
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # Step 2: Setup ILP
    # Variables x_n for n in 1..256
    # Maximize sum(n * x_n)
    # Subject to: for each prime p, sum(x_n for n where p|n) <= 1
    # And x_n in {0,1}
    
    # Objective function coefficients (minimize -sum)
    c = np.array([-n for n in range(1, N + 1)], dtype=float)
    
    # Constraint matrix A: num_primes x N
    # A[i, n-1] = 1 if primes[i] divides n
    A = np.zeros((num_primes, N), dtype=float)
    for i, p in enumerate(primes):
        for multiple in range(p, N + 1, p):
            A[i, multiple - 1] = 1.0
            
    # Linear constraints: A @ x <= 1
    lin_cons = LinearConstraint(A, ub=1)
    
    # Bounds: 0 <= x_n <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(N, dtype=int)
    
    # Solve
    try:
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds)
        if res.status == 0: # Optimal
            x = res.x
            # Extract selected integers
            selected = [i + 1 for i in range(N) if x[i] > 0.5]
            return selected
        else:
            # Fallback to greedy/local search if ILP fails
            pass
    except Exception:
        pass
        
    # Fallback: Original logic (improved slightly)
    prime_to_power = {}
    for p in primes:
        power = p
        next_power = p * p
        while next_power <= N:
            power = next_power
            next_power *= p
        prime_to_power[p] = power
        
    current_set = [1]
    current_set.extend(prime_to_power.values())
    
    def get_prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    composites = [n for n in range(N, 1, -1) if not is_prime[n] and n != 1]
    
    for n in composites:
        factors = get_prime_factors(n)
        if not factors:
            continue
            
        powers_to_remove = []
        sum_powers = 0
        valid_swap = True
        
        for p in factors:
            pp = prime_to_power[p]
            if pp in current_set:
                powers_to_remove.append(pp)
                sum_powers += pp
            else:
                valid_swap = False
                break
                
        if not valid_swap:
            continue
            
        gain = n - sum_powers
        if gain <= 0:
            continue
            
        elements_to_check = [x for x in current_set if x not in powers_to_remove]
        
        is_coprime_to_rest = True
        for x in elements_to_check:
            if math.gcd(n, x) != 1:
                is_coprime_to_rest = False
                break
                
        if is_coprime_to_rest:
            for pwr in powers_to_remove:
                current_set.remove(pwr)
            current_set.append(n)
            
    return current_set

# EVOLVE-BLOCK-END
