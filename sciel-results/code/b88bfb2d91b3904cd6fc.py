"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def _schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the tensor."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(r):
                        val += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if val != 1:
                        return False
        return True
    
    def try_rank_r(target_r, max_attempts=100):
        """
        Try to find a rank-target_r decomposition.
        Fix U and V randomly, solve for W.
        """
        n, m, p = DIMENSIONS
        num_constraints = n * m * p  # 42
        dim_W = p * n  # 14
        dim_U = n * m  # 6
        dim_V = m * p  # 21
        
        for attempt in range(max_attempts):
            # Generate random U and V with small integer entries
            U = [[random.choice([-1, 0, 1]) for _ in range(target_r)] for _ in range(dim_U)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_r)] for _ in range(dim_V)]
            
            # Build the system: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is linear in W.
            # Let's build the augmented matrix.
            
            # Number of unknowns: dim_W * target_r = 14 * target_r
            num_unknowns = dim_W * target_r
            
            # Build coefficient matrix A (num_constraints x num_unknowns) and b (num_constraints)
            A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(num_constraints)]
            b = [Fraction(1) for _ in range(num_constraints)]
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = idx
                        for c in range(target_r):
                            coeff = Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c])
                            # W index: k*n + i, column c
                            w_row = k * n + i
                            w_col = c
                            unknown_idx = w_row * target_r + w_col
                            A[row][unknown_idx] = coeff
                        idx += 1
            
            # Solve A * x = b using Gaussian elimination
            # x is a vector of length num_unknowns
            # We'll do exact Gaussian elimination
            
            # Create augmented matrix [A | b]
            aug = [row[:] + [b[i]] for i, row in enumerate(A)]
            
            # Gaussian elimination
            rank_A = 0
            pivot_cols = []
            
            for col in range(num_unknowns):
                # Find pivot
                pivot_row = None
                for row in range(rank_A, num_constraints):
                    if aug[row][col] != 0:
                        pivot_row = row
                        break
                
                if pivot_row is None:
                    continue
                
                # Swap
                aug[rank_A], aug[pivot_row] = aug[pivot_row], aug[rank_A]
                
                # Normalize pivot row
                pivot_val = aug[rank_A][col]
                for j in range(col, num_unknowns + 1):
                    aug[rank_A][j] = aug[rank_A][j] / pivot_val
                
                # Eliminate
                for row in range(num_constraints):
                    if row == rank_A:
                        continue
                    factor = aug[row][col]
                    if factor != 0:
                        for j in range(col, num_unknowns + 1):
                            aug[row][j] -= factor * aug[rank_A][j]
                
                pivot_cols.append(col)
                rank_A += 1
            
            # Check consistency: for any row that is all zeros in A part, b part must be 0
            for row in range(rank_A, num_constraints):
                if aug[row][num_unknowns] != 0:
                    return None  # Inconsistent
            
            # Solution exists. Set free variables to 0.
            x = [Fraction(0) for _ in range(num_unknowns)]
            for i, col in enumerate(pivot_cols):
                x[col] = aug[i][num_unknowns]
            
            # Reconstruct W
            W = [[Fraction(0) for _ in range(target_r)] for _ in range(dim_W)]
            for w_row in range(dim_W):
                for c in range(target_r):
                    W[w_row][c] = x[w_row * target_r + c]
            
            # Verify
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try rank 41 first
    result = try_rank_r(41, max_attempts=50)
    if result is not None:
        return result
    
    # Try rank 40
    result = try_rank_r(40, max_attempts=30)
    if result is not None:
        return result
    
    # Try rank 39
    result = try_rank_r(39, max_attempts=20)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
