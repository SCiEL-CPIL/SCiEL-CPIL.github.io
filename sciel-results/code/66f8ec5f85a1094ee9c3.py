"""Find rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication."""
from fractions import Fraction
from itertools import combinations
import time

DIMENSIONS = [4, 5, 6]

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook initialization
    rank = n * m * p
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    active = list(range(rank))
    
    # Iteratively remove redundant columns
    # A column c is redundant if the rank-1 tensor it represents
    # is in the span of the others.
    # We check this by seeing if there exist coefficients alpha_d (d != c)
    # such that for all (a,b) where U[a,c]*V[b,c] != 0 or W[...]:
    #   U[a,c] = sum_d alpha_d * U[a,d]
    #   V[b,c] = sum_d alpha_d * V[b,d]
    #   W[e,c] = sum_d alpha_d * W[e,d]
    # This is a system of linear equations.
    
    removed = 0
    while removed < len(active) and time.time() < deadline:
        progress = False
        for ci, c in enumerate(active):
            if time.time() > deadline:
                break
            # Try to remove column c
            others = [d for d in active if d != c]
            if not others:
                break
            
            # Build constraint system: find alpha_d such that
            # U[:,c] = U[:,others] @ alpha
            # V[:,c] = V[:,others] @ alpha  
            # W[:,c] = W[:,others] @ alpha
            
            # Collect all constraints
            # For each row a: U[a,c] = sum_d alpha_d * U[a,d]
            # For each row b: V[b,c] = sum_d alpha_d * V[b,d]
            # For each row e: W[e,c] = sum_d alpha_d * W[e,d]
            
            constraints = []  # list of (coeffs_vector, rhs)
            
            # U constraints
            for a in range(n * m):
                if U[a][c] != 0:
                    coeffs = [U[a][d] for d in others]
                    constraints.append((coeffs, U[a][c]))
            
            # V constraints
            for b in range(m * p):
                if V[b][c] != 0:
                    coeffs = [V[b][d] for d in others]
                    constraints.append((coeffs, V[b][c]))
            
            # W constraints
            for e in range(p * n):
                if W[e][c] != 0:
                    coeffs = [W[e][d] for d in others]
                    constraints.append((coeffs, W[e][c]))
            
            if not constraints:
                # Column c is all zeros, can remove
                active.remove(c)
                removed += 1
                progress = True
                continue
            
            # Solve the linear system using Gaussian elimination with fractions
            num_vars = len(others)
            # Build augmented matrix
            aug = []
            for coeffs, rhs in constraints:
                row = [Fraction(x) for x in coeffs] + [Fraction(rhs)]
                aug.append(row)
            
            # Gaussian elimination
            rank_aug = 0
            for col_i in range(num_vars):
                if rank_aug >= len(aug):
                    break
                # Find pivot
                pivot_row = None
                for row_i in range(rank_aug, len(aug)):
                    if aug[row_i][col_i] != 0:
                        pivot_row = row_i
                        break
                if pivot_row is None:
                    continue
                # Swap
                aug[rank_aug], aug[pivot_row] = aug[pivot_row], aug[rank_aug]
                pivot = aug[rank_aug][col_i]
                # Normalize
                for j in range(col_i, num_vars + 1):
                    aug[rank_aug][j] /= pivot
                # Eliminate
                for row_i in range(len(aug)):
                    if row_i == rank_aug:
                        continue
                    factor = aug[row_i][col_i]
                    if factor != 0:
                        for j in range(col_i, num_vars + 1):
                            aug[row_i][j] -= factor * aug[rank_aug][j]
                rank_aug += 1
            
            # Check consistency: no row with all zeros in coeff part but nonzero rhs
            consistent = True
            for row_i in range(rank_aug, len(aug)):
                if any(aug[row_i][j] != 0 for j in range(num_vars)):
                    consistent = False
                    break
                if aug[row_i][num_vars] != 0:
                    consistent = False
                    break
            
            if consistent:
                # Column c is redundant, remove it
                active.remove(c)
                removed += 1
                progress = True
                break  # Restart since indices changed
        
        if not progress:
            break
    
    # Build final matrices
    final_rank = len(active)
    U_final = [[0] * final_rank for _ in range(n * m)]
    V_final = [[0] * final_rank for _ in range(m * p)]
    W_final = [[0] * final_rank for _ in range(p * n)]
    
    for new_c, old_c in enumerate(active):
        for a in range(n * m):
            U_final[a][new_c] = U[a][old_c]
        for b in range(m * p):
            V_final[b][new_c] = V[b][old_c]
        for e in range(p * n):
            W_final[e][new_c] = W[e][old_c]
    
    # Convert to integers if possible (multiply by common denominator)
    # Check if all entries are integers
    all_int = True
    for mat in [U_final, V_final, W_final]:
        for row in mat:
            for val in row:
                if val != int(val):
                    all_int = False
                    break
            if not all_int:
                break
        if not all_int:
            break
    
    if all_int:
        for mat in [U_final, V_final, W_final]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    mat[i][j] = int(mat[i][j])
    
    return {"U": U_final, "V": V_final, "W": W_final}
