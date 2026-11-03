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
    
    # Try ranks from low to high
    for rank in range(15, 43):
        # Try several random initializations
        for trial in range(5000):
            # Use a fixed seed for reproducibility in evaluation if needed?
            # Or just random.
            # Generate random U and V with small integer entries
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(mp)]
            
            # Check if for all (i, k), the 3xrank matrix M_ik has rank 3
            valid_rank = True
            for i in range(n):
                for k in range(p):
                    # Build 3 x rank matrix
                    M = [[U[i * m + j][c] * V[j * p + k][c] for c in range(rank)] for j in range(m)]
                    # Check rank of M using Fraction
                    M_frac = [[Fraction(x) for x in row] for row in M]
                    r = 0
                    for col in range(rank):
                        pivot = -1
                        for row in range(r, m):
                            if M_frac[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        M_frac[r], M_frac[pivot] = M_frac[pivot], M_frac[r]
                        pv = M_frac[r][col]
                        for j in range(col, rank):
                            M_frac[r][j] /= pv
                        for row in range(m):
                            if row != r and M_frac[row][col] != 0:
                                factor = M_frac[row][col]
                                for j in range(col, rank):
                                    M_frac[row][j] -= factor * M_frac[r][j]
                        r += 1
                        if r == m:
                            break
                    if r < m:
                        valid_rank = False
                        break
                if not valid_rank:
                    break
            
            if not valid_rank:
                continue
            
            # If all blocks have full rank, solve for W
            W = [[Fraction(0)] * rank for _ in range(pn)]
            valid_solution = True
            for i in range(n):
                for k in range(p):
                    M = [[U[i * m + j][c] * V[j * p + k][c] for c in range(rank)] for j in range(m)]
                    M_frac = [[Fraction(x) for x in row] for row in M]
                    b = [Fraction(1)] * m
                    Aug = [M_frac[i][:] + [b[i]] for i in range(m)]
                    
                    r = 0
                    pivots = []
                    for col in range(rank):
                        pivot = -1
                        for row in range(r, m):
                            if Aug[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        Aug[r], Aug[pivot] = Aug[pivot], Aug[r]
                        pv = Aug[r][col]
                        for j in range(col, rank + 1):
                            Aug[r][j] /= pv
                        for row in range(m):
                            if row != r and Aug[row][col] != 0:
                                factor = Aug[row][col]
                                for j in range(col, rank + 1):
                                    Aug[row][j] -= factor * Aug[r][j]
                        pivots.append(col)
                        r += 1
                        if r == m:
                            break
                    
                    w_vec = [Fraction(0)] * rank
                    for row in range(m):
                        p_col = pivots[row]
                        w_vec[p_col] = Aug[row][rank]
                    
                    for c in range(rank):
                        W[k * n + i][c] = w_vec[c]
                        
                    # Check half-integer constraint
                    for c in range(rank):
                        val = W[k * n + i][c]
                        if val.denominator not in [1, 2]:
                            valid_solution = False
                            break
                    if not valid_solution:
                        break
                if not valid_solution:
                    break
            
            if valid_solution:
                # Final verification
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
