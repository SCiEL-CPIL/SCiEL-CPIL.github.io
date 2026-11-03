import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Step 1: Generate 2x2x2 Strassen decomposition (Rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: i*2+j for U, j*2+k for V, k*2+i for W
    
    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[2, 2] = 1
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B11+B01)
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    
    # Step 2: Generate 4x4x4 Strassen decomposition (Rank 49)
    # We split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = A B.
    # Each multiplication of 2x2 blocks uses the 2x2x2 Strassen decomposition.
    # There are 7 block multiplications in Strassen for 4x4.
    # Each block multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Indices for 4x4:
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen 4x4 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # We need to map the 2x2 indices to 4x2 indices.
    # Block (bi, bj) in 4x4 corresponds to 2x2 indices:
    # i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
    
    def add_block_term(U_target, V_target, W_target, col_offset, 
                       A_blocks, B_blocks, C_blocks, 
                       U2, V2, W2):
        # A_blocks: list of (bi, bj, sign)
        # B_blocks: list of (bi, bj, sign)
        # C_blocks: list of (bi, bj, sign)
        
        for col in range(7):
            u_vec = U2[:, col]
            v_vec = V2[:, col]
            w_vec = W2[:, col]
            
            # Apply to A
            for bi, bj, sign in A_blocks:
                for di in range(2):
                    for dj in range(2):
                        i_idx = 2 * bi + di
                        j_idx = 2 * bj + dj
                        u_local_idx = di * 2 + dj
                        U_target[i_idx * 4 + j_idx, col_offset + col] += sign * u_vec[u_local_idx]
                        
            # Apply to B
            for bi, bj, sign in B_blocks:
                for dj in range(2): # j index in 2x2 block
                    for dk in range(2): # k index in 2x2 block
                        j_idx = 2 * bi + dj
                        k_idx = 2 * bj + dk
                        v_local_idx = dj * 2 + dk
                        V_target[j_idx * 4 + k_idx, col_offset + col] += sign * v_vec[v_local_idx]
                        
            # Apply to C
            for bi, bj, sign in C_blocks:
                for dk in range(2):
                    for di in range(2):
                        k_idx = 2 * bi + dk
                        i_idx = 2 * bj + di
                        w_local_idx = dk * 2 + di
                        W_target[k_idx * 4 + i_idx, col_offset + col] += sign * w_vec[w_local_idx]

    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    add_block_term(U4, V4, W4, 0, 
                   [(0,0,1), (1,1,1)], 
                   [(0,0,1), (1,1,1)], 
                   [(0,0,1), (1,1,1)], 
                   U2, V2, W2)
                   
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    add_block_term(U4, V4, W4, 7,
                   [(1,0,1), (1,1,1)],
                   [(0,0,1)],
                   [(1,0,1), (1,1,-1)],
                   U2, V2, W2)
                   
    # M3 = A00(B10-B11)
    # C01 += M3, C10 += M3
    add_block_term(U4, V4, W4, 14,
                   [(0,0,1)],
                   [(1,0,1), (1,1,-1)],
                   [(0,1,1), (1,0,1)],
                   U2, V2, W2)
                   
    # M4 = A11(B01-B00)
    # C00 += M4, C10 += M4
    add_block_term(U4, V4, W4, 21,
                   [(1,1,1)],
                   [(0,1,1), (0,0,-1)],
                   [(0,0,1), (1,0,1)],
                   U2, V2, W2)
                   
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    add_block_term(U4, V4, W4, 28,
                   [(0,0,1), (0,1,1)],
                   [(1,1,1)],
                   [(0,0,-1), (0,1,1)],
                   U2, V2, W2)
                   
    # M6 = (A10-A00)(B00+B10)
    # C11 += M6
    add_block_term(U4, V4, W4, 35,
                   [(1,0,1), (0,0,-1)],
                   [(0,0,1), (1,0,1)],
                   [(1,1,1)],
                   U2, V2, W2)
                   
    # M7 = (A01-A11)(B11+B01)
    # C00 += M7, C01 += M7
    add_block_term(U4, V4, W4, 42,
                   [(0,1,1), (1,1,-1)],
                   [(1,1,1), (0,1,1)],
                   [(0,0,1), (0,1,1)],
                   U2, V2, W2)
                   
    # Step 3: Construct 4x4x8 decomposition
    # Rank = 49 + 49 = 98
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First half (cols 0-3 of B)
    U[:, 0:49] = U4
    # V mapping: V4 is 16x49 (j*4+k). V_full is 32x98 (j*8+k).
    # For k in 0..3, V_full[j*8+k, c] = V4[j*4+k, c]
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
            
    # W mapping: W4 is 16x49 (k*4+i). W_full is 32x98 (k*4+i).
    # For k in 0..3, W_full[k*4+i, c] = W4[k*4+i, c]
    W[0:16, 0:49] = W4
    
    # Second half (cols 4-7 of B)
    U[:, 49:98] = U4
    # V mapping: V4 is 16x49.
    # For k in 0..3 (local), global k = 4+k.
    # V_full[j*8+(4+k), 49+c] = V4[j*4+k, c]
    for j in range(4):
        for k in range(4):
            V[j*8+4+k, 49:98] = V4[j*4+k, 0:49]
            
    # W mapping:
    # For k in 0..3 (local), global k = 4+k.
    # W_full[(4+k)*4+i, 49+c] = W4[k*4+i, c]
    W[16:32, 49:98] = W4
    
    # Convert to lists of lists as required by interface?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The interface example uses lists of lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
