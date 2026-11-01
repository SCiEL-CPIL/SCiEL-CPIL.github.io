"""
Matrix multiplication 2x4 * 4x6: rank-minimal exact tensor decomposition.

Strategy:
- The schoolbook rank is 2*4*6 = 48.
- We attempt to find a lower-rank decomposition by exploiting the structure of the problem.
- For small matrix sizes, known results and constructive methods can reduce the rank.
- We use a structured search: fix U and V to have a Kronecker-product-like structure
  derived from the Fourier transform of Z_2 x Z_3 (sizes 2 and 3, but our dims are 2,4,6).
- Actually, let's use a more direct approach: try to find a rank-r decomposition where
  r < 48 by solving a linear system.

Given the dimensions n=2, m=4, p=6:
- U is (8, r), V is (24, r), W is (12, r)
- The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
- We need T[i,j,k] = 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])

We'll try ranks from 40 upward (since 48 is schoolbook, and some reductions are known).
For each candidate rank r, we set up the bilinear system and try to solve it.

A practical approach for small sizes: use the fact that we can combine some terms.
For 2x4*4x6, one known approach is to use Strassen-like ideas or exploit the rectangular shape.

Let's try a direct linear algebra approach:
- For a given rank r, we have r "columns" of (u_c, v_c, w_c).
- The constraint is: for all (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1.
- This is a system of 2*4*6 = 48 equations in the variables (entries of U, V, W).
- But it's trilinear, not linear. However, if we fix U and V, it's linear in W.

So: fix U and V (with some structure), then solve for W linearly.
If a solution exists with integer/half-integer entries, we're done.

We'll parameterize U and V with small integer entries and try to solve for W.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # We'll search for rank r from 40 to 47
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # For each candidate rank, try to find a solution
    # We'll use a structured approach:
    # Fix U to be a matrix with entries in {-1, 0, 1} (or small integers)
    # Fix V similarly, then solve for W linearly.
    
    # Actually, let's try a specific known construction for rectangular matrix multiplication.
    # For 2x4 * 4x6, we can think of it as 2 blocks of 1x4 * 4x6, but that's not helpful.
    
    # Let's try rank 44, 45, 46, 47 with a randomized/structured search.
    
    import random
    random.seed(42)
    
    for target_rank in range(40, 48):
        if time.time() > deadline - 10:
            break
        
        # Try several random structured U, V matrices
        attempts = 0
        max_attempts = 50 if target_rank > 42 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            attempts += 1
            
            # Generate U with small integer entries
            U = np.zeros((nm, target_rank), dtype=object)
            V = np.zeros((mp, target_rank), dtype=object)
            
            # Initialize with random small integers
            for c in range(target_rank):
                for i in range(nm):
                    U[i, c] = random.choice([-1, 0, 1])
                for j in range(mp):
                    V[j, c] = random.choice([-1, 0, 1])
            
            # Now solve for W: for each (i,k), we need
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
            # This is a system where for fixed (i,k), we have m equations in r unknowns (W[k*n+i, c] for c)
            # Wait, no. For fixed (i,k), the equation is:
            # sum_j? No, the tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # So for fixed (i,k), we have one equation per j:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for j = 0..m-1
            # This is m equations in r unknowns (the W[k*n+i, c] values).
            
            # Build the linear system for W
            # Variables: W[0..pn-1, 0..r-1] -> pn * r variables
            # Equations: for each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # That's n*m*p = 48 equations.
            
            # Let's set up the matrix A where A[e, var] = coefficient of variable var in equation e
            # Equation e corresponds to (i, j, k)
            # Variable var corresponds to (row_w, c) where row_w = k*n + i, c = 0..r-1
            
            num_eqs = n * m * p
            num_vars = pn * target_rank
            
            A = np.zeros((num_eqs, num_vars), dtype=object)
            b = np.ones(num_eqs, dtype=object)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            var_idx = (k * n + i) * target_rank + c
                            A[eq_idx, var_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            # Solve the linear system
            # Use Gaussian elimination with Fraction for exactness
            try:
                # Convert to Fraction for exact arithmetic
                A_frac = [[Fraction(A[e, v]) for v in range(num_vars)] for e in range(num_eqs)]
                b_frac = [Fraction(1) for _ in range(num_eqs)]
                
                # Gaussian elimination
                augmented = [row[:] + [b_frac[e]] for e, row in enumerate(A_frac)]
                
                rank_aug = 0
                for col in range(num_vars):
                    # Find pivot
                    pivot = None
                    for row in range(rank_aug, num_eqs):
                        if augmented[row][col] != 0:
                            pivot = row
                            break
                    if pivot is None:
                        continue
                    
                    # Swap
                    augmented[rank_aug], augmented[pivot] = augmented[pivot], augmented[rank_aug]
                    
                    # Normalize
                    pivot_val = augmented[rank_aug][col]
                    for j in range(col, num_vars + 1):
                        augmented[rank_aug][j] /= pivot_val
                    
                    # Eliminate
                    for row in range(num_eqs):
                        if row != rank_aug and augmented[row][col] != 0:
                            factor = augmented[row][col]
                            for j in range(col, num_vars + 1):
                                augmented[row][j] -= factor * augmented[rank_aug][j]
                    
                    rank_aug += 1
                
                # Check consistency
                consistent = True
                for row in range(num_eqs):
                    # Check if row is [0, 0, ..., 0 | non-zero]
                    all_zero = all(augmented[row][j] == 0 for j in range(num_vars))
                    if all_zero and augmented[row][num_vars] != 0:
                        consistent = False
                        break
                
                if consistent and rank_aug == num_vars:
                    # Extract solution
                    W_sol = [[Fraction(0)] * target_rank for _ in range(pn)]
                    for var in range(num_vars):
                        row_w = var // target_rank
                        c = var % target_rank
                        # Find the row where this variable is the pivot
                        # In our elimination, the solution is in the last column
                        # Actually, we need to back-substitute or read from the reduced form
                        # Since we did full elimination, the solution for free variables is 0
                        # and pivot variables are in the last column
                        pass
                
                # Simpler: just check if we can find a solution by reading the RREF
                # For now, let's just verify if the system has a solution by checking rank
                if consistent:
                    # Try to extract a solution
                    W_sol = [[Fraction(0)] * target_rank for _ in range(pn)]
                    for var in range(num_vars):
                        row_w = var // target_rank
                        c = var % target_rank
                        # Find which row has this as pivot
                        for r_idx in range(num_eqs):
                            if augmented[r_idx][var] == 1:
                                # Check if this is a pivot column
                                is_pivot = True
                                for other_var in range(var):
                                    if augmented[r_idx][other_var] != 0:
                                        is_pivot = False
                                        break
                                if is_pivot:
                                    W_sol[row_w][c] = augmented[r_idx][num_vars]
                                break
                    
                    # Verify the solution
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = Fraction(0)
                                for c in range(target_rank):
                                    val += U[i * m + j, c] * V[j * p + k, c] * W_sol[k * n + i][c]
                                if val != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Found a valid decomposition
                        U_list = [[int(U[i, c]) if U[i, c] == int(U[i, c]) else float(U[i, c]) for c in range(target_rank)] for i in range(nm)]
                        V_list = [[int(V[j, c]) if V[j, c] == int(V[j, c]) else float(V[j, c]) for c in range(target_rank)] for j in range(mp)]
                        W_list = [[float(W_sol[i][c]) for c in range(target_rank)] for i in range(pn)]
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            except Exception:
                continue
    
    # If no lower rank found, return schoolbook
    return best_result
