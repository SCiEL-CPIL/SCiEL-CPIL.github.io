"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has indices (i, j, k) -> T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    
    # Strategy: Use randomized algebraic geometry approach
    # 1. For a target rank r, set up the system of polynomial equations
    # 2. Use random initialization to break symmetry
    # 3. Solve using exact arithmetic (Fraction)
    # 4. Verify the solution
    
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
    
    def try_rank_decomposition(target_rank, max_iterations=100):
        """
        Try to find a rank-target_rank decomposition using randomized approach.
        Uses random linear combinations to set up solvable systems.
        """
        n, m, p = DIMENSIONS
        num_constraints = n * m * p  # 42 constraints
        
        # For each (i,j,k), we need:
        # sum_{r=1}^{target_rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Strategy: Fix U and V with random values, then solve for W
        # But this is underdetermined for small ranks. Instead, use a smarter approach.
        
        # For rank r, we have:
        # - U: (n*m) x r = 6*r unknowns
        # - V: (m*p) x r = 21*r unknowns  
        # - W: (p*n) x r = 14*r unknowns
        # Total: 41*r unknowns, 42 constraints
        
        # For r >= 2, we have more unknowns than constraints, so solutions exist generically.
        # The challenge is finding them.
        
        # Approach: Use the fact that we can parameterize the solution space.
        # Fix most of U and V to simple values, then solve for the remaining variables.
        
        for iteration in range(max_iterations):
            random.seed(iteration * 12345 + 67890)
            
            # Initialize U and V with random rational values
            # Use small integers to keep numbers manageable
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            
            # Now solve for W: for each (k, n_i) where n_i is the row index in W
            # W[k*n + i, r] appears in constraint (i, j, k) with coefficient U[i*m+j, r] * V[j*p+k, r]
            
            # For a fixed (k, i), W[k*n+i, r] appears in constraints where j varies
            # Actually, let me re-examine the indexing.
            
            # T[a, b, c] where a = i*m + j, b = j*p + k, c = k*n + i
            # So for fixed (i, j, k): a = i*m+j, b = j*p+k, c = k*n+i
            # W[c, r] = W[k*n+i, r]
            
            # For fixed (k, i), W[k*n+i, r] appears in constraints for all j in range(m)
            # The constraint is: sum_{j=0}^{m-1} sum_{r=1}^{target_rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Wait, no. Each (i,j,k) gives one constraint.
            
            # Let me re-derive. For each (i,j,k):
            # sum_{r=1}^{target_rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # For fixed (k, i), the variable W[k*n+i, r] appears in constraints for all j.
            # Specifically, for j = 0, 1, ..., m-1:
            # sum_{r=1}^{target_rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This gives m equations for the r variables W[k*n+i, 0..r-1]
            # But these are coupled across different (k,i) pairs? No, each (k,i) pair is independent.
            
            # Actually, for fixed (k,i), we have m equations (one for each j) and r unknowns (W[k*n+i, 0..r-1]).
            # If r >= m, this is underdetermined. For our case m=3, so if r>=3, it's underdetermined.
            
            # Let's solve this system for each (k,i) pair.
            
            W = [[0] * target_rank for _ in range(p * n)]
            
            success = True
            for k in range(p):
                for i in range(n):
                    # For this (k,i), solve for W[k*n+i, 0..target_rank-1]
                    # Equations: for j in range(m):
                    # sum_{r=0}^{target_rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    
                    # This is a linear system: A * w = b where
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # w[r] = W[k*n+i, r]
                    # b[j] = 1
                    
                    A = [[Fraction(U[i * m + j][r] * V[j * p + k][r]) for r in range(target_rank)] for j in range(m)]
                    b = [Fraction(1) for _ in range(m)]
                    
                    # Solve A * w = b using Gaussian elimination
                    # If the system is inconsistent, this iteration fails.
                    # If it's underdetermined, we need to pick a solution.
                    
                    # Use exact Gaussian elimination
                    augmented = [row[:] + [b[j]] for j, row in enumerate(A)]
                    
                    # Forward elimination
                    rank_A = 0
                    for col in range(target_rank):
                        # Find pivot
                        pivot_row = None
                        for row in range(rank_A, m):
                            if augmented[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row is None:
                            continue
                        
                        # Swap
                        augmented[rank_A], augmented[pivot_row] = augmented[pivot_row], augmented[rank_A]
                        
                        # Normalize
                        pivot_val = augmented[rank_A][col]
                        for c in range(col, target_rank + 1):
                            augmented[rank_A][c] = augmented[rank_A][c] / pivot_val
                        
                        # Eliminate
                        for row in range(m):
                            if row != rank_A and augmented[row][col] != 0:
                                factor = augmented[row][col]
                                for c in range(col, target_rank + 1):
                                    augmented[row][c] -= factor * augmented[rank_A][c]
                        
                        rank_A += 1
                    
                    # Check consistency
                    for row in range(rank_A, m):
                        if augmented[row][target_rank] != 0:
                            success = False
                            break
                    if not success:
                        break
                    
                    # Extract solution - set free variables to 0
                    w_solution = [Fraction(0)] * target_rank
                    for row in range(rank_A):
                        # Find the leading column
                        lead_col = None
                        for c in range(target_rank):
                            if augmented[row][c] != 0:
                                lead_col = c
                                break
                        if lead_col is not None:
                            w_solution[lead_col] = augmented[row][target_rank]
                    
                    W[k * n + i] = w_solution
                    if not success:
                        break
                if not success:
                    break
            
            if not success:
                continue
            
            # Verify the solution
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = Fraction(0)
                        for r in range(target_rank):
                            total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                        if total != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list of Fractions
                U_frac = [[Fraction(x) for x in row] for row in U]
                V_frac = [[Fraction(x) for x in row] for row in V]
                W_frac = [[Fraction(x) for x in row] for row in W]
                return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Schoolbook: n*m*p = 42
    
    # Try to find the minimum rank
    for target_rank in range(21, n * m * p + 1):
        result = try_rank_decomposition(target_rank, max_iterations=50)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
