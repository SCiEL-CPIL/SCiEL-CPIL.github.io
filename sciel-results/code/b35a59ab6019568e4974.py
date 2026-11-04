import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65 decomposition: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1)
    rank = 65
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen Decomposition (Rank 49) ---
    # We handle the first 4 columns of B and first 4 columns of C.
    
    # 2x2 Strassen coefficients
    # A indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # B indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # C indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    
    # u2[l] is the vector of coefficients for A in the l-th product
    # v2[l] is the vector of coefficients for B in the l-th product
    # w2[c][l] is the coefficient of product l in C[c]
    
    u2 = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [1, 0, -1, 0],# M6: c-a (using c-a)
        [0, 1, 0, -1] # M7: b-d
    ]
    
    v2 = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, 0], # M4: g
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    w2 = [
        [1, 0, 0, 1, -1, 0, -1], # C11: M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # C12: M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C21: M2 + M4
        [1, -1, 1, 0, 0, 0, 1]   # C22: M1 - M2 + M3 + M7
    ]
    
    # Block coefficients for 4x4 Strassen
    # A blocks: 0:A11, 1:A12, 2:A21, 3:A22
    # B blocks: 0:B11, 1:B12, 2:B21, 3:B22
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    alpha_a = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [1, 0, -1, 0],# M6 (A21-A11 -> -A11+A21)
        [0, 1, 0, -1] # M7 (A12-A22)
    ]
    
    alpha_b = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, -1, 1, 0],# M4 (B21-B11)
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    col = 0
    for k in range(7):
        for l in range(7):
            # U part
            # U index: i (0..15)
            # A block b (0..3), entry in block e (0..3)
            # U[i][col] = sum_b alpha_a[k][b] * u2[l][e]
            # where i = b*4 + e (row-major: block row, then entry row/col)
            # Wait, A is 4x4. Blocks are 2x2.
            # A11 is rows 0-1, cols 0-1.
            # Index i in A: r, c. Block b = (r//2)*2 + (c//2). Entry e = (r%2)*2 + (c%2).
            
            for i in range(16):
                r = i // 4
                c = i % 4
                b = (r // 2) * 2 + (c // 2)
                e = (r % 2) * 2 + (c % 2)
                coeff = alpha_a[k][b] * u2[l][e]
                if coeff != 0:
                    U[i][col] += coeff
            
            # V part
            # V index: j (0..15) for first 4 cols of B?
            # B is 4x5. We only use first 4 cols for this part.
            # B block b (0..3), entry e (0..3).
            # V[j][col] = sum_b alpha_b[k][b] * v2[l][e]
            # j = b*4 + e
            
            for j in range(16):
                r = j // 4
                c = j % 4
                b = (r // 2) * 2 + (c // 2)
                e = (r % 2) * 2 + (c % 2)
                coeff = alpha_b[k][b] * v2[l][e]
                if coeff != 0:
                    V[j][col] += coeff
            
            # W part
            # W index: o (0..15) for first 4 cols of C.
            # C block b (0..3), entry e (0..3).
            # W[o][col] = sum_{c_block} w2[c_block][l] * (coeff for C_block in M_k?)
            # Wait, M_k is a 2x2 matrix. Its entries are C_blocks.
            # C = sum_k M_k_contrib.
            # Actually, the block Strassen algorithm says:
            # C11 = M1 + M4 - M5 - M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M7
            # So for a given k (which M_k we are computing), the contribution to C_block c_b is w2[c_b][l]?
            # No, w2[c_b][l] is the coefficient of the l-th scalar product in the 2x2 Strassen for M_k?
            # No, w2 is for the 2x2 Strassen algorithm itself.
            # Here, M_k is the result of a 2x2 matrix multiplication.
            # The l-th scalar product in the Strassen algorithm for M_k contributes to the entries of M_k.
            # Let M_k be a 2x2 matrix. Its entries are (M_k)_{r,c}.
            # The contribution of the l-th product to (M_k)_{r,c} is w2[r*2+c][l].
            # So for C_block c_b (which corresponds to (M_k)_{r,c} where c_b = r*2+c), the coefficient is w2[c_b][l].
            
            for o in range(16):
                r = o // 4
                c = o % 4
                c_b = (r // 2) * 2 + (c // 2)
                e = (r % 2) * 2 + (c % 2)
                # w2 index: c_b is the block index in the 2x2 result of M_k.
                # But w2 is defined for the 2x2 result.
                # So coeff = w2[c_b][l].
                coeff = w2[c_b][l]
                if coeff != 0:
                    W[o][col] += coeff
            
            col += 1
    
    # --- Part 2: 4x4x1 Schoolbook Decomposition (Rank 16) ---
    # Uses column 4 (index 4) of B and produces column 4 (index 4) of C.
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    
    for i in range(4):
        for j in range(4):
            # U[i, j] = 1
            U[i * 4 + j][col] = 1
            # V[j, 4] = 1 (j is row of B, 4 is col)
            V[j * 5 + 4][col] = 1
            # W[4, i] = 1 (4 is col of C, i is row)
            W[4 * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Quick verification
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    # Tensor T = sum_c U_c V_c W_c
    # Check T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # i: 0..15 (A), j: 0..19 (B), k: 0..19 (C)
    # A[i] is row i of A? No, A is 4x4. i is index in 4x4.
    # B[j] is row j of B? No, B is 4x5. j is index in 4x5.
    # C[k] is row k of C? No, C is 4x5. k is index in 4x5.
    
    # Let's pick a random A, B and check C = A B.
    A = np.random.randint(0, 10, (4, 4))
    B = np.random.randint(0, 10, (4, 5))
    C_true = A @ B
    
    # Flatten A, B, C
    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = C_true.flatten()
    
    # Compute C_calc
    # T is 4D tensor? No, we have U, V, W.
    # C_flat[k] = sum_i sum_j A_flat[i] B_flat[j] * (sum_c U[i,c] V[j,c] W[k,c])
    # Let M[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # Then C_flat[k] = sum_i sum_j A_flat[i] B_flat[j] M[i,j,k]
    
    # Efficient check:
    # C_calc = (A_flat @ U) * (V.T @ B_flat) ... no.
    # C_calc[k] = sum_c (sum_i A[i] U[i,c]) (sum_j B[j] V[j,c]) W[k,c]
    
    # Let u_c = A_flat @ U[:, c]
    # Let v_c = B_flat @ V[:, c]
    # C_calc = sum_c u_c v_c W[:, c]
    
    u_vals = A_flat @ U
    v_vals = B_flat @ V
    C_calc = np.sum(u_vals[:, None] * v_vals[None, :] * W, axis=1)
    
    if np.allclose(C_calc, C_flat):
        print("Verification Passed")
    else:
        print("Verification Failed")
        print("C_true:", C_true)
        print("C_calc:", C_calc.reshape(4, 5))
