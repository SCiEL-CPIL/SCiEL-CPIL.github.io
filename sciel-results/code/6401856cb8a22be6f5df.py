"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try ranks from low to high
    for rank in range(30, 43):
        # Initialize U and V with simple integer patterns
        # U: (nm, rank), V: (mp, rank)
        # Use a structured initialization
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        
        # Fill U and V with a pattern that might allow a solution
        # Strategy: make U and V have 1s in a structured way
        for c in range(rank):
            i = (c // (m * p)) % n
            j = (c // p) % m
            k = c % p
            U[i * m + j][c] = Fraction(1)
            V[j * p + k][c] = Fraction(1)
        
        # Now solve for W: (pn, rank)
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in W
        # We have nm*p = 42 equations, pn*rank = 14*rank unknowns
        
        # Build the linear system A * W_flat = b
        # W_flat is pn * rank vector
        num_eqs = n * m * p
        num_unk = pn * rank
        
        A = [[Fraction(0)] * num_unk for _ in range(num_eqs)]
        b = [Fraction(1)] * num_eqs
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        if coeff != 0:
                            W_idx = k * n + i  # row in W
                            W_col = c  # column in W
                            flat_idx = W_idx * rank + W_col
                            A[eq_idx][flat_idx] += coeff
                    eq_idx += 1
        
        # Solve A * x = b using Gaussian elimination with rationals
        # Add augmented column
        Aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Gaussian elimination
        rank_aug = 0
        for col in range(num_unk):
            # Find pivot
            pivot = -1
            for row in range(rank_aug, num_eqs):
                if Aug[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            
            # Swap
            Aug[rank_aug], Aug[pivot] = Aug[pivot], Aug[rank_aug]
            
            # Normalize
            pivot_val = Aug[rank_aug][col]
            for j in range(col, num_unk + 1):
                Aug[rank_aug][j] /= pivot_val
            
            # Eliminate
            for row in range(num_eqs):
                if row != rank_aug and Aug[row][col] != 0:
                    factor = Aug[row][col]
                    for j in range(col, num_unk + 1):
                        Aug[row][j] -= factor * Aug[rank_aug][j]
            
            rank_aug += 1
        
        # Check for consistency
        consistent = True
        for row in range(rank_aug, num_eqs):
            if Aug[row][num_unk] != 0:
                consistent = False
                break
        
        if not consistent:
            continue
        
        # Extract solution (set free variables to 0)
        W_flat = [Fraction(0)] * num_unk
        for row in range(rank_aug):
            # Find the leading column
            lead_col = -1
            for col in range(num_unk):
                if Aug[row][col] != 0:
                    lead_col = col
                    break
            if lead_col != -1:
                W_flat[lead_col] = Aug[row][num_unk]
        
        # Build W
        W = [[Fraction(0)] * rank for _ in range(pn)]
        for row_idx in range(pn):
            for col_idx in range(rank):
                W[row_idx][col_idx] = W_flat[row_idx * rank + col_idx]
        
        # Verify the solution
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list of lists with float or Fraction
            U_out = [[float(x) if x.denominator == 1 else float(x) for x in row] for row in U]
            V_out = [[float(x) if x.denominator == 1 else float(x) for x in row] for row in V]
            W_out = [[float(x) if x.denominator == 1 else float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
    rank = n * m * p
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
# EVOLVE-BLOCK-END
