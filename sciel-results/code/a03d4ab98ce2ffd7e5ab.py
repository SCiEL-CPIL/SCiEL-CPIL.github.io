import numpy as np
import time
import random
from fractions import Fraction
from scipy.optimize import least_squares
from itertools import product
import sys

DIMENSIONS = [2, 5, 5]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _verify_exact(n, m, p, U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _solve_linear_system_exact(A, b):
    """
    Solve A x = b exactly using Gaussian elimination with Fractions.
    A is a list of lists, b is a list.
    Returns solution as list of Fractions, or None if no unique solution.
    """
    num_eq = len(b)
    num_var = len(A[0])
    
    # Augmented matrix
    aug = [[Fraction(A[i][j]) for j in range(num_var)] + [Fraction(b[i])] for i in range(num_eq)]
    
    # Gaussian elimination
    row = 0
    for col in range(num_var):
        # Find pivot
        pivot = None
        for r in range(row, num_eq):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        
        # Swap rows
        aug[row], aug[pivot] = aug[pivot], aug[row]
        
        # Eliminate below
        pivot_val = aug[row][col]
        for r in range(row + 1, num_eq):
            factor = aug[r][col] / pivot_val
            if factor != 0:
                for c in range(col, num_var + 1):
                    aug[r][c] -= factor * aug[row][c]
        
        row += 1
        if row == num_eq:
            break
    
    # Check for inconsistency
    for r in range(row, num_eq):
        if aug[r][num_var] != 0:
            return None
    
    # Back substitution
    x = [Fraction(0)] * num_var
    for i in range(row - 1, -1, -1):
        s = aug[i][num_var]
        for j in range(i + 1, num_var):
            s -= aug[i][j] * x[j]
        if aug[i][i] == 0:
            # Underdetermined - set to 0 for now
            x[i] = Fraction(0)
        else:
            x[i] = s / aug[i][i]
    
    return x

def _try_rank_random_sparse(n, m, p, target_rank, max_attempts=50, timeout=8):
    """
    Try to find a rank `target_rank` decomposition by:
    1. Randomly generating sparse integer support patterns for U, V, W
    2. Solving the resulting linear system exactly
    """
    start_time = time.time()
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    num_constraints = n * m * p
    
    # Precompute which (i,j,k) each (u_idx, v_idx, w_idx) triple corresponds to
    # For each constraint (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # Generate random sparse patterns
        # Each factor matrix has some sparsity pattern
        # Strategy: for each column c, randomly assign which entries are nonzero
        
        U_pat = [[0] * target_rank for _ in range(num_u)]
        V_pat = [[0] * target_rank for _ in range(num_v)]
        W_pat = [[0] * target_rank for _ in range(num_w)]
        
        # For each rank dimension c, choose random support
        for c in range(target_rank):
            # Choose subset of rows to be nonzero in U
            u_support = set(random.sample(range(num_u), min(random.randint(1, 3), num_u)))
            v_support = set(random.sample(range(num_v), min(random.randint(1, 3), num_v)))
            w_support = set(random.sample(range(num_w), min(random.randint(1, 3), num_w)))
            
            for idx in u_support:
                U_pat[idx][c] = random.choice([-1, 1])
            for idx in v_support:
                V_pat[idx][c] = random.choice([-1, 1])
            for idx in w_support:
                W_pat[idx][c] = random.choice([-1, 1])
        
        # Now we have a bilinear system: for each (i,j,k),
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is linear in the products, but we can rewrite:
        # Let x_c = V[j*p+k,c]*W[k*n+i,c] for fixed (j,k)
        # But this doesn't work directly because x depends on j,k
        
        # Alternative: Fix U and V patterns, then the system is linear in W
        # Or: treat it as a system where unknowns are the actual values
        
        # Better approach: The constraint is:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # If we fix U and V, this is linear in W entries.
        
        # Let's fix U and V patterns with random values, then solve for W.
        
        # Assign random values to U and V
        U_vals = [[Fraction(0)] * target_rank for _ in range(num_u)]
        V_vals = [[Fraction(0)] * target_rank for _ in range(num_v)]
        
        for c in range(target_rank):
            u_support = set(random.sample(range(num_u), min(random.randint(2, 4), num_u)))
            v_support = set(random.sample(range(num_v), min(random.randint(2, 4), num_v)))
            
            for idx in u_support:
                U_vals[idx][c] = Fraction(random.choice([-2, -1, 1, 2]))
            for idx in v_support:
                V_vals[idx][c] = Fraction(random.choice([-2, -1, 1, 2]))
        
        # Now solve for W: for each (i,j,k):
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in W[k*n+i,c] for each c
        # Unknowns: W[0,0], W[0,1], ..., W[0,r-1], W[1,0], ..., W[p*n-1, r-1]
        # Total unknowns: p*n*target_rank
        # Total equations: n*m*p
        
        num_unknowns = p * n * target_rank
        
        if num_unknowns < num_constraints:
            # Not enough unknowns - skip
            continue
        
        # Build linear system A * x = b
        # x is flattened W
        A = []
        b = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [Fraction(0)] * num_unknowns
                    for c in range(target_rank):
                        coeff = U_vals[i * m + j][c] * V_vals[j * p + k][c]
                        if coeff != 0:
                            w_idx = k * n + i
                            x_idx = w_idx * target_rank + c
                            row[x_idx] = coeff
                    A.append(row)
                    b.append(Fraction(1))
        
        # Solve
        try:
            sol = _solve_linear_system_exact(A, b)
        except Exception:
            continue
        
        if sol is None:
            continue
        
        # Check if solution is valid (all values should be "nice" - but we accept any rational)
        # Reconstruct W
        W_vals = [[Fraction(0)] * target_rank for _ in range(p * n)]
        for idx in range(p * n):
            for c in range(target_rank):
                W_vals[idx][c] = sol[idx * target_rank + c]
        
        # Verify
        if _verify_exact(n, m, p, U_vals, V_vals, W_vals):
            # Convert to lists of floats for return
            U_out = [[float(x) for x in row] for row in U_vals]
            V_out = [[float(x) for x in row] for row in V_vals]
            W_out = [[float(x) for x in row] for row in W_vals]
            return U_out, V_out, W_out
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()

    # Try ranks from 50 down, using random sparse pattern approach
    for target_rank in range(50, 20, -1):
        if time.time() > deadline:
            break
        
        sol = _try_rank_random_sparse(n, m, p, target_rank, max_attempts=100, timeout=10)
        
        if sol is not None:
            best_U, best_V, best_W = sol
            best_rank = target_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
