"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def _primes_up_to(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    primes = _primes_up_to(1000)
    
    # Precompute mapping from (p, r) to index
    # And reverse mapping
    p_to_idx = {}
    idx_to_pr = []
    current_idx = 0
    for p in primes:
        p_to_idx[p] = current_idx
        for r in range(p):
            idx_to_pr.append((p, r))
        current_idx += p
    
    num_vars = current_idx
    
    # Precompute for each k in 1..L_max, which (p,r) cover it
    # We will build the constraint matrix on the fly for each L_target
    # To be efficient, we can precompute the list of indices for each k
    # But k goes up to 1000, and for each k, we need to check all p <= 1000
    # This is 1000 * 168 checks, which is fast.
    
    def solve_for_L(L_target):
        if L_target <= 0:
            return []
        
        # Build coverage constraints
        # A_cov[k, idx] = 1 if idx covers k
        # We need A_cov * x >= 1 for k=1..L_target
        
        # Number of coverage constraints is L_target
        # Number of variable constraints:
        # 1. Distinct primes: sum_r x_{p,r} <= 1 for each p
        # 2. Budget: sum_{p,r} (1/p) x_{p,r} <= 1
        
        num_constraints = L_target + len(primes) + 1
        
        # Use lil_matrix for sparse construction
        A = lil_matrix((num_constraints, num_vars), dtype=float)
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)
        
        # 1. Coverage constraints: A[k, idx] >= 1
        # lb[k] = 1, ub[k] = inf
        for k in range(1, L_target + 1):
            row = k - 1
            lb[row] = 1
            ub[row] = np.inf
            # Find all (p, r) such that k % p == r
            for p in primes:
                if p > L_target:
                    break # Optimization: if p > L_target, r is determined by k%p, but we only have 1 residue per p selected.
                    # Actually, for a fixed p, only ONE r is selected.
                    # The constraint is sum_{r} x_{p,r} * [k % p == r] >= 1? No.
                    # The constraint is: sum over all selected classes (p,r) that cover k >= 1.
                    # For a specific p, if x_{p, r_p} is selected, it covers k if k % p == r_p.
                    # So in the linear combination, the coefficient for x_{p, r} is 1 if r == k % p, else 0.
                    # So for each p, we just need to set A[row, idx(p, k%p)] = 1.
                    r = k % p
                    idx = p_to_idx[p] + r
                    A[row, idx] = 1
        
        # 2. Distinct primes constraints: sum_{r} x_{p,r} <= 1
        # lb = -inf, ub = 1
        for i, p in enumerate(primes):
            row = L_target + i
            lb[row] = -np.inf
            ub[row] = 1
            start_idx = p_to_idx[p]
            for r in range(p):
                A[row, start_idx + r] = 1
        
        # 3. Budget constraint: sum (1/p) x_{p,r} <= 1
        row = L_target + len(primes)
        lb[row] = -np.inf
        ub[row] = 1
        for p in primes:
            start_idx = p_to_idx[p]
            inv_p = 1.0 / p
            for r in range(p):
                A[row, start_idx + r] = inv_p
        
        A = A.tocsr()
        
        # Variables bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
        
        # Objective: Minimize 0 (feasibility)
        c = np.zeros(num_vars)
        
        # Integality: all binary
        integrality = np.ones(num_vars)
        
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=bounds, integrality=integrality)
        
        if res.status == 0:
            # Extract solution
            classes = []
            for p in primes:
                start_idx = p_to_idx[p]
                for r in range(p):
                    if res.x[start_idx + r] > 0.5:
                        classes.append([p, r])
            return classes
        else:
            return None

    # Try to find the largest L
    # We know L=1000 is the target.
    # We can binary search or just iterate down from 1000.
    # Given the structure, if 1000 is infeasible, 999 might be.
    # Let's try 1000 down to 1.
    
    best_classes = []
    for L in range(1000, 0, -1):
        classes = solve_for_L(L)
        if classes is not None:
            best_classes = classes
            break
            
    return best_classes

# EVOLVE-BLOCK-END
