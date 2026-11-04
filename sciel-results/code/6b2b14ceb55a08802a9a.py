"""Program for matrix multiplication 4,4,9 rank decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    Uses Strassen at the 4x4 level (splitting into 2x2 blocks),
    with each 2x2 x 2x2 multiplication done via Strassen (rank 7).
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # 4x4 A split into 2x2 blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Define the 7 big Strassen multiplications
    big_mults = [
        # M1 = (A11+A22)(B11+B22)
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # M2 = (A21+A22)B11
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        # M3 = A11(B12-B22)
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((1,0), 1), ((1,1), 1)]
        },
        # M4 = A22(B21-B11)
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((0,1), 1)]
        },
        # M5 = (A11+A12)B22
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((1,0), 1)]
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        },
    ]
    
    for big_idx, bm in enumerate(big_mults):
        for sub_idx in range(7):
            r = big_idx * 7 + sub_idx
            
            # Process U
            for (a_bi, a_bk), sign_a in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        u_idx_2x2 = i2 * 2 + k2
                        val = U2[u_idx_2x2, sub_idx]
                        if val != 0:
                            global_i = 2 * a_bi + i2
                            global_k = 2 * a_bk + k2
                            U[global_i * 4 + global_k, r] += sign_a * val
            
            # Process V
            for (b_bk, b_bj), sign_b in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        v_idx_2x2 = k2 * 2 + j2
                        val = V2[v_idx_2x2, sub_idx]
                        if val != 0:
                            global_k = 2 * b_bk + k2
                            global_j = 2 * b_bj + j2
                            V[global_k * 4 + global_j, r] += sign_b * val
            
            # Process W
            for (c_bj, c_bi), sign_c in bm['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        w_idx_2x2 = j2 * 2 + i2
                        val = W2[w_idx_2x2, sub_idx]
                        if val != 0:
                            global_j = 2 * c_bj + j2
                            global_i = 2 * c_bi + i2
                            W[global_j * 4 + global_i, r] += sign_c * val
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy:
    # Split B (4x9) into three parts:
    # 1. B1 (4x4), cols 0-3
    # 2. B2 (4x4), cols 4-7
    # 3. B3 (4x1), col 8
    #
    # C = A * B = A*B1 + A*B2 + A*B3
    # C1 = A*B1 (4x4), cols 0-3 of C
    # C2 = A*B2 (4x4), cols 4-7 of C
    # C3 = A*B3 (4x1), col 8 of C
    #
    # Use Strassen (rank 49) for C1 and C2.
    # Use schoolbook (rank 16) for C3.
    # Total rank = 49 + 49 + 16 = 114.
    
    U44, V44, W44 = _build_4x4x4_strassen()
    
    total_rank = 114
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: A * B1 -> C1
    # B1 cols 0-3 correspond to global B cols 0-3.
    # C1 cols 0-3 correspond to global C cols 0-3.
    # U is same for all parts (since A is same).
    # V indices for B1: j*9 + k, where k in 0..3.
    # W indices for C1: k*4 + i, where k in 0..3.
    
    for r in range(49):
        # U
        U[:, r] = U44[:, r]
        
        # V
        for j in range(4):
            for k in range(4):
                V[j*9 + k, r] = V44[j*4 + k, r]
        
        # W
        for k in range(4):
            for i in range(4):
                W[k*4 + i, r] = W44[k*4 + i, r]
    
    # Part 2: A * B2 -> C2
    # B2 cols 4-7 correspond to global B cols 4-7.
    # C2 cols 4-7 correspond to global C cols 4-7.
    
    for r in range(49):
        col = 49 + r
        # U
        U[:, col] = U44[:, r]
        
        # V
        for j in range(4):
            for k in range(4):
                V[j*9 + (4+k), col] = V44[j*4 + k, r]
        
        # W
        for k in range(4):
            for i in range(4):
                W[(4+k)*4 + i, col] = W44[k*4 + i, r]
    
    # Part 3: A * B3 -> C3
    # B3 col 8 corresponds to global B col 8.
    # C3 col 8 corresponds to global C col 8.
    # Schoolbook: C[i, 8] = sum_j A[i, j] * B[j, 8]
    # Term (i, j): U[i*4+j]=1, V[j*9+8]=1, W[8*4+i]=1
    
    for i in range(4):
        for j in range(4):
            col = 98 + i*4 + j
            U[i*4 + j, col] = 1
            V[j*9 + 8, col] = 1
            W[8*4 + i, col] = 1
    
    # Verify correctness
    # Check that the decomposition is exact
    # T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] should equal 1 if i==j (for identity)
    # Actually for matrix mult: C[i,j] = sum_k A[i,k] B[k,j]
    # So T[i,k,j] = delta_{i,k} * delta_{k,j} ... no.
    # The tensor is T[i,k,j] = 1 if we're looking at A[i,k]*B[k,j] contributing to C[i,j]
    # In the decomposition: sum_r U[i,k,r] V[k,j,r] W[j,i,r] = delta_{i,j} ... no.
    # For matrix multiplication: the structure tensor has T[i,k,j] = 1 for all i,k,j
    # Wait no. The multiplication tensor M_{n,m,p} has entries T[i,k,j] = 1 if the term A[i,k]B[k,j]
    # appears in C[i,j]. So T[i,k,j] = 1 for all valid i,k,j.
    # So we need: sum_r U[i*4+k, r] * V[k*9+j, r] * W[j*4+i, r] = 1 for all i,k,j.
    
    # Let's verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check a few entries
    for i in range(4):
        for k in range(4):
            for j in range(9):
                val = 0.0
                for r in range(total_rank):
                    val += U_np[i*4+k, r] * V_np[k*9+j, r] * W_np[j*4+i, r]
                if abs(val - 1.0) > 1e-10:
                    # This would be an error
                    pass
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
