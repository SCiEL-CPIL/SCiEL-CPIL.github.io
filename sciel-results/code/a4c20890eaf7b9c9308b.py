"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
from fractions import Fraction
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T_ijk = 1 for all i,j,k
    # We search for rank r
    # Try ranks from low to high
    for rank in range(15, 43):
        # Try multiple random initializations
        for attempt in range(500):
            # Generate random U and V with entries in {-1, 0, 1}
            # Using a fixed seed per attempt for reproducibility? No, just random.
            # But we need it to be fast.
            
            # To increase density, use {-1, 1} or {0, 1}
            # Let's use {-1, 1} for U and V to avoid zero rows/cols
            U = [[Fraction(random.choice([-1, 1])) for _ in range(rank)] for _ in range(nm)]
            V = [[Fraction(random.choice([-1, 1])) for _ in range(rank)] for _ in range(mp)]
            
            # Build linear system for W
            # A * W_flat = b
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
                                W_idx = k * n + i
                                W_col = c
                                flat_idx = W_idx * rank + W_col
                                A[eq_idx][flat_idx] += coeff
                        eq_idx += 1
            
            # Solve A * x = b
            # Gaussian elimination
            Aug = [row[:] + [b[i]] for i, row in enumerate(A)]
            
            rank_aug = 0
            for col in range(num_unk):
                pivot = -1
                for row in range(rank_aug, num_eqs):
                    if Aug[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                
                Aug[rank_aug], Aug[pivot] = Aug[pivot], Aug[rank_aug]
                
                pivot_val = Aug[rank_aug][col]
                for j in range(col, num_unk + 1):
                    Aug[rank_aug][j] /= pivot_val
                
                for row in range(num_eqs):
                    if row != rank_aug and Aug[row][col] != 0:
                        factor = Aug[row][col]
                        for j in range(col, num_unk + 1):
                            Aug[row][j] -= factor * Aug[rank_aug][j]
                
                rank_aug += 1
            
            # Check consistency
            consistent = True
            for row in range(rank_aug, num_eqs):
                if Aug[row][num_unk] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Extract solution
            W_flat = [Fraction(0)] * num_unk
            for row in range(rank_aug):
                lead_col = -1
                for col in range(num_unk):
                    if Aug[row][col] != 0:
                        lead_col = col
                        break
                if lead_col != -1:
                    W_flat[lead_col] = Aug[row][num_unk]
            
            # Check if W entries are half-integers
            valid_half_int = True
            for val in W_flat:
                if val.denominator not in [1, 2]:
                    valid_half_int = False
                    break
            
            if not valid_half_int:
                continue
            
            # Build W
            W = [[Fraction(0)] * rank for _ in range(pn)]
            for row_idx in range(pn):
                for col_idx in range(rank):
                    W[row_idx][col_idx] = W_flat[row_idx * rank + col_idx]
            
            # Verify
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
                U_out = [[float(x) for x in row] for row in U]
                V_out = [[float(x) for x in row] for row in V]
                W_out = [[float(x) for x in row] for row in W]
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
