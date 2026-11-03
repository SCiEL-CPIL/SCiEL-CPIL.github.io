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
    
    # The matrix multiplication tensor T has indices (i, j, k) -> T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    
    # Schoolbook rank is n*m*p = 42
    # We'll try to find a decomposition with rank less than 42 using algebraic geometry.
    
    def _schoolbook_decomposition(n, m, p):
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
        """Verify that the decomposition exactly represents the matrix multiplication tensor."""
        num_U = len(U)
        num_V = len(V)
        num_W = len(W)
        rank = len(U[0]) if num_U > 0 else 0
        
        # Check dimensions
        if num_U != n * m or num_V != m * p or num_W != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def find_rank_r_decomposition(n, m, p, target_rank, max_attempts=50):
        """
        Try to find a rank-target_rank decomposition using randomized algebraic geometry approach.
        
        Strategy:
        1. Randomly initialize U and V with rational entries.
        2. For each constraint (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        3. This is a system of n*m*p equations in p*n*target_rank unknowns (entries of W).
        4. But the equations are linear in W only if we fix U and V.
        5. So we fix U and V randomly, then solve the linear system for W.
        6. If a solution exists, verify it.
        """
        np.random.seed(int(time.time()) % 1000000)
        
        for attempt in range(max_attempts):
            # Initialize U and V with small random rational values
            # Use integers in {-1, 0, 1} for simplicity, but scale to avoid trivial solutions
            scale = 1
            U = [[random.randint(-scale, scale) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-scale, scale) for _ in range(target_rank)] for _ in range(m * p)]
            
            # Ensure U and V are not all zero
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            
            # Now solve for W: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a linear system in the entries of W.
            # Let's set up the system: for each (i,j,k), we have one equation.
            # Unknowns: W[a,c] for a in range(p*n), c in range(target_rank)
            # Total unknowns: p*n*target_rank
            # Total equations: n*m*p
            
            num_eqs = n * m * p
            num_unk = p * n * target_rank
            
            # Build the linear system A * w_vec = b
            # w_vec is a vector of length num_unk, indexed by (a, c) -> a*target_rank + c
            # For equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # So the coefficient of W[k*n+i, c] in equation (i,j,k) is U[i*m+j,c] * V[j*p+k,c]
            
            A = [[0] * num_unk for _ in range(num_eqs)]
            b = [1] * num_eqs
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = k * n + i
                        for c in range(target_rank):
                            A[eq_idx][a * target_rank + c] = U[i * m + j][c] * V[j * p + k][c]
                        eq_idx += 1
            
            # Solve the linear system using Gaussian elimination over rationals
            # Since the system is underdetermined (num_unk > num_eqs typically), we need to find any solution.
            # Use exact rational arithmetic.
            
            W_solution = _solve_linear_system_rational(A, b)
            
            if W_solution is not None:
                # W_solution is a list of lists, W_solution[a][c]
                # Verify the solution
                if verify_decomposition(U, V, W_solution, n, m, p):
                    return {"U": U, "V": V, "W": W_solution}
        
        return None
    
    def _solve_linear_system_rational(A, b):
        """
        Solve A * x = b over the rationals, where A is num_eqs x num_unk.
        Returns a list of lists x where x is a vector of length num_unk, or None if no solution.
        Since the system is underdetermined, we return one particular solution.
        """
        num_eqs = len(A)
        num_unk = len(A[0]) if num_eqs > 0 else 0
        
        # Convert to Fraction
        A_frac = [[Fraction(x) for x in row] for row in A]
        b_frac = [Fraction(x) for x in b]
        
        # Augmented matrix
        aug = [A_frac[i] + [b_frac[i]] for i in range(num_eqs)]
        
        # Gaussian elimination with partial pivoting
        rank = 0
        for col in range(num_unk):
            # Find pivot
            pivot_row = -1
            for row in range(rank, num_eqs):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            # Swap rows
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            
            # Normalize pivot row
            pivot_val = aug[rank][col]
            for j in range(col, num_unk + 1):
                aug[rank][j] /= pivot_val
            
            # Eliminate other rows
            for row in range(num_eqs):
                if row == rank:
                    continue
                factor = aug[row][col]
                if factor != 0:
                    for j in range(col, num_unk + 1):
                        aug[row][j] -= factor * aug[rank][j]
            
            rank += 1
        
        # Check for inconsistency
        for row in range(rank, num_eqs):
            if any(aug[row][j] != 0 for j in range(num_unk)) and aug[row][num_unk] != 0:
                return None
        
        # Find a particular solution: set free variables to 0
        x = [Fraction(0) for _ in range(num_unk)]
        
        # Determine which columns are pivot columns
        pivot_cols = []
        for row in range(rank):
            for col in range(num_unk):
                if aug[row][col] != 0:
                    pivot_cols.append(col)
                    break
        
        # For each pivot row, solve for the pivot variable
        for row in range(rank):
            # Find the pivot column for this row
            pivot_col = -1
            for col in range(num_unk):
                if aug[row][col] != 0:
                    pivot_col = col
                    break
            if pivot_col == -1:
                continue
            
            # x[pivot_col] = aug[row][num_unk] - sum(aug[row][col] * x[col] for col != pivot_col)
            # Since free variables are 0, x[pivot_col] = aug[row][num_unk]
            x[pivot_col] = aug[row][num_unk]
        
        # Convert to list of lists for W: W[a][c] = x[a*target_rank + c]
        # But we don't know target_rank here, so return as flat list
        return x
    
    # Try to find a decomposition with rank less than 42
    schoolbook_rank = n * m * p
    
    # Try ranks from 41 down to, say, 35
    for target_rank in range(schoolbook_rank - 1, 34, -1):
        result = find_rank_r_decomposition(n, m, p, target_rank, max_attempts=20)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
