import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    Indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    Tensor entry T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must equal 1.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                product_sum = np.dot(np.dot(u_val, v_val), w_val)
                if abs(product_sum - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2 + j, r]
    V[j*2 + k, r]
    W[k*2 + i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[2, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_4():
    """
    Construct rank-49 decomposition for 4x4x4 using Strassen recursion.
    View 4x4 as 2x2 blocks of 2x2 matrices.
    """
    U2, V2, W2 = _strassen_2x2_2()
    R2 = 7
    R4 = R2 * R2  # 49
    
    U4 = np.zeros((16, R4))
    V4 = np.zeros((16, R4))
    W4 = np.zeros((16, R4))
    
    # A block coefficients for each Strassen product Mq (q=0..6):
    # Blocks: 0=A00, 1=A01, 2=A10, 3=A11
    A_block_coeff = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    # B block coefficients: 0=B00, 1=B01, 2=B10, 3=B11
    B_block_coeff = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    
    # C block coefficients: C_block_coeff[cb][q] = coeff of Mq in C_cb
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_block_coeff = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    for q in range(7):
        for r_local in range(7):
            r4 = q * 7 + r_local
            
            # Build U4 column r4
            for ab in range(4):
                if A_block_coeff[q][ab] == 0:
                    continue
                block_row = ab // 2
                block_col = ab % 2
                coeff_a = A_block_coeff[q][ab]
                for local_i in range(2):
                    for local_j in range(2):
                        global_i = 2 * block_row + local_i
                        global_j = 2 * block_col + local_j
                        U4[global_i * 4 + global_j, r4] += coeff_a * U2[local_i * 2 + local_j, r_local]
            
            # Build V4 column r4
            for bb in range(4):
                if B_block_coeff[q][bb] == 0:
                    continue
                block_row = bb // 2
                block_col = bb % 2
                coeff_b = B_block_coeff[q][bb]
                for local_j in range(2):
                    for local_k in range(2):
                        global_j = 2 * block_row + local_j
                        global_k = 2 * block_col + local_k
                        V4[global_j * 4 + global_k, r4] += coeff_b * V2[local_j * 2 + local_k, r_local]
            
            # Build W4 column r4
            for cb in range(4):
                if C_block_coeff[cb][q] == 0:
                    continue
                block_row = cb // 2  # k-block
                block_col = cb % 2    # i-block
                coeff_c = C_block_coeff[cb][q]
                for local_k in range(2):
                    for local_i in range(2):
                        global_k = 2 * block_row + local_k
                        global_i = 2 * block_col + local_i
                        W4[global_k * 4 + global_i, r4] += coeff_c * W2[local_k * 2 + local_i, r_local]
    
    return U4, V4, W4

def _strassen_4x8_8():
    """
    Construct rank-98 decomposition for 4x4x8 by combining two 4x4x4 Strassen decompositions.
    A (4x4) * B (4x8) = [A*B0 | A*B1] where B0, B1 are 4x4.
    Dimensions: n=4, m=4, p=8.
    U shape (n*m, rank) = (16, 98)
    V shape (m*p, rank) = (32, 98)
    W shape (p*n, rank) = (32, 98)
    """
    U1, V1, W1 = _strassen_4x4_4()
    if U1 is None:
        return None, None, None
        
    R1 = 49
    R_total = 2 * R1  # 98
    
    U = np.zeros((16, R_total))
    V = np.zeros((32, R_total))
    W = np.zeros((32, R_total))
    
    # Part 1: k in 0..3 (B0)
    for c in range(R1):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, c] = U1[i * 4 + j, c]
                
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, c] = V1[j * 4 + k, c]
                
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, c] = W1[k * 4 + i, c]
                
    # Part 2: k in 4..7 (B1)
    for c in range(R1, 2 * R1):
        c1 = c - R1
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, c] = U1[i * 4 + j, c1]
                
        for j in range(4):
            for k in range(4, 8):
                V[j * 8 + k, c] = V1[j * 4 + (k - 4), c1]
                
        for k in range(4, 8):
            for i in range(4):
                W[k * 4 + i, c] = W1[(k - 4) * 4 + i, c1]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try Strassen
    try:
        U_str, V_str, W_str = _strassen_4x8_8()
        if U_str is not None:
            if _verify_decomposition(U_str, V_str, W_str, n, m, p, tol=1e-9):
                return {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
    except Exception:
        pass
        
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
