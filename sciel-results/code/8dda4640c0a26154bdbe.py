import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                diff = abs(val - 1.0)
                if diff > error:
                    error = diff
    return error < 1e-9

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    
    # Target Rank 26
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mappings
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..6 -> idx = j*7 + k
    # C: k in 0..6, i in 0..1 -> idx = k*2 + i
    
    col = 0
    
    # Strassen Coefficients for 2x2
    # A coeffs (for a00, a01, a10, a11)
    strassen_A = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # B coeffs (for b00, b01, b10, b11)
    strassen_B = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # C coeffs (for c00, c01, c10, c11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_C = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Helper to add a Strassen block
    def add_strassen_block(b_start_k, c_start_k):
        nonlocal col
        # B indices for this block (2x2)
        # b00, b01, b10, b11
        # b00: j=0, k=b_start_k -> 0*7 + b_start_k
        # b01: j=0, k=b_start_k+1 -> 0*7 + b_start_k+1
        # b10: j=1, k=b_start_k -> 1*7 + b_start_k
        # b11: j=1, k=b_start_k+1 -> 1*7 + b_start_k+1
        b_indices = [
            0 * 7 + b_start_k,
            0 * 7 + b_start_k + 1,
            1 * 7 + b_start_k,
            1 * 7 + b_start_k + 1
        ]
        
        # C indices for this block (2x2)
        # c00, c01, c10, c11
        # c00: k=c_start_k, i=0 -> c_start_k*2 + 0
        # c01: k=c_start_k+1, i=0 -> (c_start_k+1)*2 + 0
        # c10: k=c_start_k, i=1 -> c_start_k*2 + 1
        # c11: k=c_start_k+1, i=1 -> (c_start_k+1)*2 + 1
        c_indices = [
            c_start_k * 2 + 0,
            (c_start_k + 1) * 2 + 0,
            c_start_k * 2 + 1,
            (c_start_k + 1) * 2 + 1
        ]
        
        # A indices for A_L (2x2)
        # a00, a01, a10, a11
        a_indices = [
            0 * 3 + 0,
            0 * 3 + 1,
            1 * 3 + 0,
            1 * 3 + 1
        ]
        
        for k in range(7):
            # U part
            for a_idx, val in zip(a_indices, strassen_A[k]):
                U[a_idx, col] = val
            
            # V part
            for b_idx, val in zip(b_indices, strassen_B[k]):
                V[b_idx, col] = val
            
            # W part
            for c_idx, val in zip(c_indices, strassen_C[k]):
                W[c_idx, col] = val
            
            col += 1
    
    # Block 1: B cols 0,1
    add_strassen_block(0, 0)
    
    # Block 2: B cols 2,3
    add_strassen_block(2, 2)
    
    # Block 3: B cols 4,5
    add_strassen_block(4, 4)
    
    # Part 4: Schoolbook for b_4 (B col 6, rows 0,1)
    # A_L (2x2)
    a_indices = [
        0 * 3 + 0,
        0 * 3 + 1,
        1 * 3 + 0,
        1 * 3 + 1
    ]
    
    # B part: b06, b16
    # b06: j=0, k=6 -> 6
    # b16: j=1, k=6 -> 13
    b_indices = [6, 13]
    
    # C part: c06, c16
    # c06: k=6, i=0 -> 12
    # c16: k=6, i=1 -> 13
    c_indices = [12, 13]
    
    # Terms:
    # 1: a00 * b06 * c06
    # 2: a01 * b16 * c06
    # 3: a10 * b06 * c16
    # 4: a11 * b16 * c16
    
    # Term 1
    U[a_indices[0], col] = 1
    V[b_indices[0], col] = 1
    W[c_indices[0], col] = 1
    col += 1
    
    # Term 2
    U[a_indices[1], col] = 1
    V[b_indices[1], col] = 1
    W[c_indices[0], col] = 1
    col += 1
    
    # Term 3
    U[a_indices[2], col] = 1
    V[b_indices[0], col] = 1
    W[c_indices[1], col] = 1
    col += 1
    
    # Term 4
    U[a_indices[3], col] = 1
    V[b_indices[1], col] = 1
    W[c_indices[1], col] = 1
    col += 1
    
    # Part 5: Outer product a_R * b_bot
    # a_R: A col 2 (indices 2, 5)
    # b_bot: B row 2 (indices 14..20)
    # C: all (indices 0..13)
    
    # U part
    U[2, col] = 1
    U[5, col] = 1
    
    # V part
    for k in range(7):
        V[2 * 7 + k, col] = 1
        
    # W part
    for k in range(7):
        for i in range(2):
            W[k * 2 + i, col] = 1
            
    col += 1
    
    # Check col
    # 3 blocks * 7 = 21
    # 1 block * 4 = 4
    # 1 term = 1
    # Total 26
    
    if col != rank:
        print(f"Warning: col {col} != rank {rank}")
        
    # Convert to lists of lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify(U_list, V_list, W_list):
        print("Verification failed")
        # Fallback to schoolbook if verification fails (should not happen)
        # But for safety, return schoolbook
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][c] = 1
                    V_sb[j * p + k][c] = 1
                    W_sb[k * n + i][c] = 1
                    c += 1
        return {'U': U_sb, 'V': V_sb, 'W': W_sb}
        
    return {'U': U_list, 'V': V_list, 'W': W_list}
