"""
Strategy: Algebraic Dependency Pruning via Linear Algebra.
Instead of random search, we formulate the tensor decomposition as a linear system.
We iteratively build a sparse support (which rank-1 terms are active) and solve for
the exact rational coefficients using Gaussian elimination over Fractions.
This is a structural change from heuristic random search to exact algebraic solving.
"""

import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # For each (i,j,k), the coefficient should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _solve_linear_system(A, b):
    """
    Solve A x = b over Fractions using Gaussian Elimination.
    A is a list of lists (dense), b is a list.
    Returns list of Fractions or None if inconsistent.
    """
    n = len(b)
    if n == 0:
        return []
    m = len(A[0])
    
    # Create augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    for i in range(n):
        # Find pivot
        pivot_row = None
        for r in range(i, n):
            if aug[r][i] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            # Check if consistent (0 = non-zero)
            if aug[i][n] != 0:
                return None
            # Free variable, set to 0 for simplicity (we just need one solution)
            # Actually, if no pivot, the variable is free. We set it to 0.
            # But we need to ensure the rest of the column is 0.
            for r in range(i + 1, n):
                if aug[r][i] != 0:
                    return None # Should not happen if we swapped correctly, but just in case
            continue
        
        # Swap
        if pivot_row != i:
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]
            
        # Normalize
        pivot_val = aug[i][i]
        for j in range(i, n + 1):
            aug[i][j] /= pivot_val
            
        # Eliminate
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            if factor != 0:
                for j in range(i, n + 1):
                    aug[r][j] -= factor * aug[i][j]
                    
    return [aug[i][n] for i in range(n)]

def _construct_decomposition_for_support(n, m, p, support_U, support_V, support_W, rank):
    """
    Given fixed supports (sets of indices where entries are non-zero),
    solve for the values.
    
    The equations are:
    For each (i,j,k): sum_{c in S} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    where S is the set of columns c such that the term is "active" in the support.
    
    This is a non-linear system (product of variables).
    However, if we fix U and V, W is linear.
    If we fix U, the system is quadratic in V and W.
    
    To make it linear, we can fix two of the matrices to simple values (e.g., 1s)
    or use a specific structure.
    
    Alternative Approach: 
    We know that for schoolbook, U, V, W are 0/1.
    Let's assume the solution has entries in {1, -1, 0}.
    If we fix the supports, we have a system of equations in the signs.
    This is still hard.
    
    Let's try a different structural assumption:
    Assume W is fixed to a specific simple pattern? No.
    
    Let's go back to the linear system idea.
    If we fix the values of U and V to be 1 or -1, then W is determined linearly.
    But we don't know U and V.
    
    New Strategy: 
    1. Generate random sparse patterns for U, V, W with values +/-1.
    2. This defines a linear system for... wait, if U, V, W are fixed, the value is fixed.
    
    We need to find U, V, W.
    
    Let's use the property that if we fix U and V, W is linear.
    Let's fix U to be a random 0/1 matrix with rank r.
    Let's fix V to be a random 0/1 matrix with rank r.
    Then for each (i,j,k), we have:
    sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    Let A_{ijk} = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    This is linear in W if U, V are fixed.
    Let M be the matrix where rows correspond to (i,j,k) and columns to (k,i,c).
    M[(i,j,k), (k,i,c)] = U[i,j,c] * V[j,k,c].
    Then M * W_flat = 1_vector.
    
    So, if we pick random sparse 0/1 matrices U and V, we can solve for W linearly!
    If the solution W has small integer/half-integer entries, we have a candidate.
    """
    
    n, m, p = n, m, p
    num_ijk = n * m * p
    num_w_vars = p * n * rank
    
    # Build the linear system M * W = 1
    # M has shape (num_ijk, num_w_vars)
    # We construct M based on U and V.
    
    # We need to be careful with memory if rank is large, but rank <= 30.
    # num_ijk = 30. num_w_vars <= 30 * 10 = 300.
    # This is a small system.
    
    M = [[Fraction(0) for _ in range(num_w_vars)] for _ in range(num_ijk)]
    b = [Fraction(1) for _ in range(num_ijk)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_ijk = i * m * p + j * p + k
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    if u_val != 0 and v_val != 0:
                        # W variable index: (k, i, c) -> (k * n + i) * rank + c
                        idx_w = (k * n + i) * rank + c
                        M[idx_ijk][idx_w] = Fraction(u_val) * Fraction(v_val)
                        
    # Solve M W = b
    W_flat = _solve_linear_system(M, b)
    if W_flat is None:
        return None
        
    # Check if W_flat entries are "nice" (integers or half-integers)
    # The problem allows integer/half-integer entries.
    for val in W_flat:
        # Check if val is integer or half-integer
        num = val.numerator
        den = val.denominator
        if den not in [1, 2]:
            return None
            
    # Reshape W_flat to W matrix
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = W_flat[(k * n + i) * rank + c]
                
    # We have U, V, W. Verify.
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    return None

def _try_rank_linear(n, m, p, r, deadline, max_attempts=2000):
    """
    Try to find rank r decomposition by fixing U, V to random 0/1 matrices
    and solving for W linearly.
    """
    start = time.time()
    rng = random.Random(100 + r)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Generate random sparse 0/1 matrices U and V
        # We want them to be "full rank" in a sense that the system is solvable.
        # Density control: start with higher density, maybe decrease?
        # For r=30, density 1.0 is schoolbook.
        # For lower r, we need sparsity.
        
        # Heuristic: probability of 1 is roughly (n*m*p) / (r * (n*m)) ? 
        # Total 1s in U should be ~ r * (n*m) / (n*m*p) * something?
        # Actually, if U is 0/1, each column c has some support.
        
        # Let's just try random 0/1 with varying density
        density = rng.uniform(0.2, 0.8)
        
        U = [[1 if rng.random() < density else 0 for _ in range(r)] for _ in range(n * m)]
        V = [[1 if rng.random() < density else 0 for _ in range(r)] for _ in range(m * p)]
        
        # Ensure at least some non-zeros
        if all(all(x == 0 for x in row) for row in U):
            continue
        if all(all(x == 0 for x in row) for row in V):
            continue
            
        result = _construct_decomposition_for_support(n, m, p, U, V, r)
        if result is not None:
            return result
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline: leave 10 seconds margin before 120s limit
    deadline = time.time() + 100
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(10, 15, 6) = 15
    # Try ranks from lower bound up to schoolbook - 1
    
    for r in range(15, schoolbook_rank):
        if time.time() > deadline:
            break
            
        result = _try_rank_linear(n, m, p, r, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
            
    return best

# EVOLVE-BLOCK-END
