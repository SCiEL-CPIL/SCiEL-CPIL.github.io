import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    def verify(U, V, W, tol=1e-8):
        rank = len(U[0]) if U else 0
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k2*n+i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build constraint matrix
    num_vars = n * m * p  # 48
    A_rows = []
    
    # sum_k c[i,j,k] = 0
    for i in range(n):
        for j in range(m):
            row = [0] * num_vars
            for k in range(p):
                row[i*m*p + j*p + k] = 1
            A_rows.append(row)
    # sum_i c[i,j,k] = 0
    for j in range(m):
        for k in range(p):
            row = [0] * num_vars
            for i in range(n):
                row[i*m*p + j*p + k] = 1
            A_rows.append(row)
    # sum_j c[i,j,k] = 0
    for k in range(p):
        for i in range(n):
            row = [0] * num_vars
            for j in range(m):
                row[i*m*p + j*p + k] = 1
            A_rows.append(row)
    
    # Perform Gaussian elimination with fractions to find null space
    # Convert to Fraction matrix
    A_frac = [[Fraction(x) for x in row] for row in A_rows]
    num_rows = len(A_frac)
    num_cols = num_vars
    
    # RREF with tracking
    pivot_cols = []
    for col in range(num_cols):
        # Find pivot
        pivot_row = None
        for row in range(len(pivot_cols), num_rows):
            if A_frac[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        # Swap
        A_frac[len(pivot_cols)], A_frac[pivot_row] = A_frac[pivot_row], A_frac[len(pivot_cols)]
        pivot_cols.append(col)
        # Normalize
        pivot_val = A_frac[len(pivot_cols)-1][col]
        for c in range(col, num_cols):
            A_frac[len(pivot_cols)-1][c] = A_frac[len(pivot_cols)-1][c] / pivot_val
        # Eliminate
        for row in range(num_rows):
            if row != len(pivot_cols)-1 and A_frac[row][col] != 0:
                factor = A_frac[row][col]
                for c in range(col, num_cols):
                    A_frac[row][c] = A_frac[row][c] - factor * A_frac[len(pivot_cols)-1][c]
    
    # Free variables are those not in pivot_cols
    free_cols = [c for c in range(num_cols) if c not in pivot_cols]
    
    # For each free variable, set it to 1 and others to 0, solve for pivot variables
    null_space_vectors = []
    for fc in free_cols:
        vec = [Fraction(0)] * num_cols
        vec[fc] = Fraction(1)
        for pc in pivot_cols:
            # From RREF: x_pc + sum(free) A[pc_row, fc] * x_fc = 0
            # So x_pc = -A[pc_row, fc]
            pc_row = pivot_cols.index(pc)
            vec[pc] = -A_frac[pc_row][fc]
        null_space_vectors.append(vec)
    
    # Now search for integer combinations of null space vectors that give half-integer entries
    # Try small integer coefficients
    max_coeff = 5
    null_dim = len(null_space_vectors)
    
    # Try individual null space vectors first
    for vec in null_space_vectors:
        if time.time() > deadline - 10:
            break
        # Check if all entries are half-integers
        # An entry is a half-integer if 2*entry is an integer
        valid = True
        for v in vec:
            two_v = 2 * v
            if two_v.denominator != 1:
                valid = False
                break
        if valid:
            # This vector has half-integer entries
            # Construct rank-47 decomposition
            U_new = np.array(U_sb, dtype=float)
            V_new = np.array(V_sb, dtype=float)
            W_new = np.array(W_sb, dtype=float)
            
            for idx in range(num_vars):
                if idx == 0:
                    continue
                c_val = float(vec[idx])
                if abs(c_val) > 1e-10:
                    i = idx // (m*p)
                    j = (idx % (m*p)) // p
                    k = idx % p
                    U_new[i*m+j, idx] += c_val
                    V_new[j*p+k, idx] += c_val
                    W_new[k*n+i, idx] += c_val
            
            U_new[:, 0] = 0
            V_new[:, 0] = 0
            W_new[:, 0] = 0
            
            U_final = np.delete(U_new, 0, axis=1)
            V_final = np.delete(V_new, 0, axis=1)
            W_final = np.delete(W_new, 0, axis=1)
            
            if verify(U_final.tolist(), V_final.tolist(), W_final.tolist()):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Try combinations of two null space vectors
    for i in range(null_dim):
        for j in range(i+1, null_dim):
            if time.time() > deadline - 10:
                break
            for ci in range(-max_coeff, max_coeff+1):
                for cj in range(-max_coeff, max_coeff+1):
                    if ci == 0 and cj == 0:
                        continue
                    vec = [null_space_vectors[i][r]*ci + null_space_vectors[j][r]*cj for r in range(num_cols)]
                    valid = True
                    for v in vec:
                        two_v = 2 * v
                        if two_v.denominator != 1:
                            valid = False
                            break
                    if valid:
                        U_new = np.array(U_sb, dtype=float)
                        V_new = np.array(V_sb, dtype=float)
                        W_new = np.array(W_sb, dtype=float)
                        
                        for idx in range(num_vars):
                            if idx == 0:
                                continue
                            c_val = float(vec[idx])
                            if abs(c_val) > 1e-10:
                                ii = idx // (m*p)
                                jj = (idx % (m*p)) // p
                                kk = idx % p
                                U_new[ii*m+jj, idx] += c_val
                                V_new[jj*p+kk, idx] += c_val
                                W_new[kk*n+ii, idx] += c_val
                        
                        U_new[:, 0] = 0
                        V_new[:, 0] = 0
                        W_new[:, 0] = 0
                        
                        U_final = np.delete(U_new, 0, axis=1)
                        V_final = np.delete(V_new, 0, axis=1)
                        W_final = np.delete(W_new, 0, axis=1)
                        
                        if verify(U_final.tolist(), V_final.tolist(), W_final.tolist()):
                            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
