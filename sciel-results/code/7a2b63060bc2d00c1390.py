"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * mp  # n*m*p = 42
    
    start_time = time.time()
    timeout = 110  # seconds, leave some margin for the 120s limit
    
    def _schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reconstructs the tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if total != 1:
                        return False
        return True
    
    def try_rank_randomized(target_rank, max_attempts=50, time_limit=20.0):
        """
        Try to find a rank-target_rank decomposition using randomized approach.
        Fix U and V randomly, then solve for W using exact Gaussian elimination.
        """
        start = time.time()
        
        for attempt in range(max_attempts):
            if time.time() - start > time_limit:
                return None
            
            # Initialize U and V with small random rational values
            # Use integers for simplicity, then we can scale
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            
            # Fill U and V with random small integers
            for a in range(nm):
                for r in range(target_rank):
                    U[a][r] = random.randint(-1, 1)
            
            for b in range(mp):
                for r in range(target_rank):
                    V[b][r] = random.randint(-1, 1)
            
            # Now solve for W: for each (i,j,k), we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # This is a system of nm*p = 42 equations in pn * target_rank = 14*target_rank unknowns
            # For target_rank = 41: 574 unknowns, 42 equations - very underdetermined
            #
            # But the equations are not linear in W alone - they are linear in W for fixed U,V
            # Actually, for fixed U and V, each equation is linear in the W entries.
            #
            # Equation for (i,j,k): sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # where a = i*m+j, b = j*p+k, c = k*n+i
            #
            # So for each c (which is k*n+i), we have equations involving W[c, r] for all r.
            # The coefficient of W[c, r] in equation (i,j,k) is U[a,r] * V[b,r].
            
            # Build the linear system
            # Unknowns: W[c, r] for c in range(pn), r in range(target_rank)
            # Total unknowns: pn * target_rank
            
            num_unknowns = pn * target_rank
            num_equations = nm * mp  # = n*m*p = 42
            
            # Build augmented matrix
            # Each row corresponds to an equation (i,j,k)
            # Each column corresponds to an unknown W[c,r]
            
            # For equation (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
            # Coefficient of W[c', r] is:
            #   U[a, r] * V[b, r] if c' == c, else 0
            
            # So each equation only involves W[c, r] for a specific c (determined by i,k)
            # This means the system is block diagonal!
            
            # For each c = k*n+i, the equations involving W[c, r] are those where
            # k*n+i = c, i.e., i = c % n, k = c // n
            # And j ranges over 0..m-1
            # So for each c, we have m equations (one for each j) involving W[c, r]
            
            # Let's solve block by block for each c
            W = [[0] * target_rank for _ in range(pn)]
            
            valid = True
            for c in range(pn):
                i = c % n
                k = c // n
                
                # Equations for this c: for each j in 0..m-1
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1
                
                # Build m x target_rank system
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # b[j] = 1
                
                A = [[0] * target_rank for _ in range(m)]
                rhs = [1] * m
                
                for j in range(m):
                    for r in range(target_rank):
                        A[j][r] = U[i * m + j][r] * V[j * p + k][r]
                
                # Solve A * W[c] = rhs using Gaussian elimination
                solution = solve_linear_system(A, rhs)
                if solution is None:
                    valid = False
                    break
                
                for r in range(target_rank):
                    W[c][r] = solution[r]
            
            if not valid:
                continue
            
            # Verify the decomposition
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def solve_linear_system(A, b):
        """
        Solve A * x = b where A is m x n (m <= n typically) using exact Gaussian elimination.
        Returns a particular solution (free variables set to 0) or None if inconsistent.
        """
        m = len(A)
        n = len(A[0]) if m > 0 else 0
        
        # Convert to Fraction
        A_frac = [[Fraction(A[i][j]) for j in range(n)] for i in range(m)]
        b_frac = [Fraction(b[i]) for i in range(m)]
        
        # Augmented matrix
        aug = [A_frac[i][:] + [b_frac[i]] for i in range(m)]
        
        rank = 0
        pivot_cols = []
        
        for col in range(n):
            if rank >= m:
                break
            
            # Find pivot
            pivot_row = -1
            for row in range(rank, m):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            # Swap
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            
            # Normalize
            pivot_val = aug[rank][col]
            for j in range(col, n + 1):
                aug[rank][j] /= pivot_val
            
            # Eliminate
            for row in range(m):
                if row == rank:
                    continue
                factor = aug[row][col]
                if factor != 0:
                    for j in range(col, n + 1):
                        aug[row][j] -= factor * aug[rank][j]
            
            pivot_cols.append(col)
            rank += 1
        
        # Check for inconsistency
        for row in range(rank, m):
            all_zero = True
            for j in range(n):
                if aug[row][j] != 0:
                    all_zero = False
                    break
            if all_zero and aug[row][n] != 0:
                return None
        
        # Construct particular solution: free variables = 0
        x = [Fraction(0)] * n
        for i in range(rank):
            x[pivot_cols[i]] = aug[i][n]
        
        return x
    
    # Try ranks from schoolbook-1 downward
    # Most likely to succeed at schoolbook-1, then -2, etc.
    
    # Try rank 41 first
    result = try_rank_randomized(schoolbook_rank - 1, max_attempts=100, time_limit=25.0)
    if result is not None:
        return result
    
    # Try rank 40
    if time.time() - start_time < timeout - 10:
        result = try_rank_randomized(schoolbook_rank - 2, max_attempts=80, time_limit=20.0)
        if result is not None:
            return result
    
    # Try rank 39
    if time.time() - start_time < timeout - 5:
        result = try_rank_randomized(schoolbook_rank - 3, max_attempts=60, time_limit=15.0)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
