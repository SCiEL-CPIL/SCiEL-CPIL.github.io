import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = 58
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a schoolbook term for a specific (i, j, k)
    def add_schoolbook(i, j, k, col_idx):
        U[i * m + j, col_idx] = 1
        V[j * p + k, col_idx] = 1
        W[k * n + i, col_idx] = 1

    # Helper to add a Strassen block for indices (i1, i2), (j1, j2), (k1, k2)
    def add_strassen(i1, i2, j1, j2, k1, k2, start_col):
        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # U: A[i1,j1] -> +1, A[i2,j2] -> +1
        # V: B[j1,k1] -> +1, B[j2,k2] -> +1
        # W: C[i1,k1] -> +1, C[i1,k2] -> +1, C[i2,k1] -> +1, C[i2,k2] -> +1 ? No, W is output.
        # The product M1 contributes to C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2] with coefficients.
        
        # Let's define the contributions of each M_t to C_ik.
        # M1: C[i1,k1]+=1, C[i1,k2]+=1, C[i2,k1]+=1, C[i2,k2]+=1 (Wait, check Strassen formulas)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5 - M6 - M7
        # C21 = M2 + M4 + M6 + M7
        # C22 = M1 - M2 + M3 + M7
        
        # So for each M_t, we add the corresponding coefficients to W.
        # And we define U and V for M_t.
        
        # M1
        U[i1 * m + j1, start_col + 0] += 1
        U[i2 * m + j2, start_col + 0] += 1
        V[j1 * p + k1, start_col + 0] += 1
        V[j2 * p + k2, start_col + 0] += 1
        W[k1 * n + i1, start_col + 0] += 1
        W[k2 * n + i1, start_col + 0] += 1 # Wait, C12 coeff is 1? No, C12 = M3+M5-M6-M7. M1 is not in C12.
        
        # Let's re-verify Strassen output formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5 - M6 - M7
        # C21 = M2 + M4 + M6 + M7
        # C22 = M1 - M2 + M3 + M7
        
        # So:
        # M1 contributes to C11 (+1) and C22 (+1).
        # M2 contributes to C21 (+1) and C22 (-1).
        # M3 contributes to C12 (+1) and C22 (+1).
        # M4 contributes to C11 (+1) and C21 (+1).
        # M5 contributes to C12 (+1) and C11 (-1).
        # M6 contributes to C12 (-1) and C21 (+1).
        # M7 contributes to C11 (+1), C12 (-1), C21 (+1), C22 (+1).
        
        # Correction:
        # M1: U(A11+A22), V(B11+B22). W: C11+=1, C22+=1.
        U[i1 * m + j1, start_col + 0] += 1
        U[i2 * m + j2, start_col + 0] += 1
        V[j1 * p + k1, start_col + 0] += 1
        V[j2 * p + k2, start_col + 0] += 1
        W[k1 * n + i1, start_col + 0] += 1
        W[k2 * n + i2, start_col + 0] += 1
        
        # M2: U(A21), V(B11-B12). W: C21+=1, C22-=1.
        U[i2 * m + j1, start_col + 1] += 1
        V[j1 * p + k1, start_col + 1] += 1
        V[j1 * p + k2, start_col + 1] -= 1
        W[k1 * n + i2, start_col + 1] += 1
        W[k2 * n + i2, start_col + 1] -= 1
        
        # M3: U(A11+A12), V(B22). W: C12+=1, C22+=1.
        U[i1 * m + j1, start_col + 2] += 1
        U[i1 * m + j2, start_col + 2] += 1
        V[j2 * p + k2, start_col + 2] += 1
        W[k2 * n + i1, start_col + 2] += 1
        W[k2 * n + i2, start_col + 2] += 1
        
        # M4: U(A22), V(B21-B11). W: C11+=1, C21+=1.
        U[i2 * m + j2, start_col + 3] += 1
        V[j2 * p + k1, start_col + 3] += 1
        V[j1 * p + k1, start_col + 3] -= 1
        W[k1 * n + i1, start_col + 3] += 1
        W[k1 * n + i2, start_col + 3] += 1
        
        # M5: U(A12+A22), V(B21). W: C12+=1, C11-=1.
        U[i1 * m + j2, start_col + 4] += 1
        U[i2 * m + j2, start_col + 4] += 1
        V[j2 * p + k1, start_col + 4] += 1
        W[k2 * n + i1, start_col + 4] += 1
        W[k1 * n + i1, start_col + 4] -= 1
        
        # M6: U(A11-A21), V(B12). W: C12-=1, C21+=1.
        U[i1 * m + j1, start_col + 5] += 1
        U[i2 * m + j1, start_col + 5] -= 1
        V[j1 * p + k2, start_col + 5] += 1
        W[k2 * n + i1, start_col + 5] -= 1
        W[k2 * n + i2, start_col + 5] += 1
        
        # M7: U(A22-A11), V(B11+B22). W: C11+=1, C12-=1, C21+=1, C22+=1.
        U[i2 * m + j2, start_col + 6] += 1
        U[i1 * m + j1, start_col + 6] -= 1
        V[j1 * p + k1, start_col + 6] += 1
        V[j2 * p + k2, start_col + 6] += 1
        W[k1 * n + i1, start_col + 6] += 1
        W[k2 * n + i1, start_col + 6] -= 1
        W[k1 * n + i2, start_col + 6] += 1
        W[k2 * n + i2, start_col + 6] += 1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1, 0)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen(0, 1, 2, 3, 2, 3, 7)
    
    # Remaining schoolbook terms
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block1 = (i in [0, 1] and j in [0, 1] and k in [0, 1])
                in_block2 = (i in [0, 1] and j in [2, 3] and k in [2, 3])
                if not in_block1 and not in_block2:
                    add_schoolbook(i, j, k, col)
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
