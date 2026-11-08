import numpy as np
import time

def construct():
    n, m, p = 5, 5, 5
    # Use Strassen for 8x8, embed 5x5
    N = 8
    rank = 64
    
    # Generate Strassen decomposition for N x N
    U, V, W = strassen_decomp(N)
    
    # U, V, W are now of size (N*N, rank)
    # We need to map 5x5 indices to 8x8 indices
    # Index mapping: (i, j) -> i*N + j
    # We only care about i,j in 0..4
    
    U_5 = np.zeros((25, rank))
    V_5 = np.zeros((25, rank))
    W_5 = np.zeros((25, rank))
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            # In 8x8, this is i*N + j
            idx_8 = i * N + j
            U_5[idx, :] = U[idx_8, :]
            V_5[idx, :] = V[idx_8, :]
            W_5[idx, :] = W[idx_8, :]
            
    return {"U": U_5.tolist(), "V": V_5.tolist(), "W": W_5.tolist()}

def strassen_decomp(N):
    # Recursive Strassen decomposition
    # Returns U, V, W of size (N*N, rank)
    if N == 2:
        # Base case 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices: 00, 01, 10, 11
        # U rows: 0,1,2,3
        # V rows: 0,1,2,3
        # W rows: 0,1,2,3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[2, 6] = -1
        
        return U, V, W

    # Recursive step
    N_half = N // 2
    U_half, V_half, W_half = strassen_decomp(N_half)
    rank_half = U_half.shape[1]
    
    U = np.zeros((N*N, rank_half * 7))
    V = np.zeros((N*N, rank_half * 7))
    W = np.zeros((N*N, rank_half * 7))
    
    # Block indices
    # 00, 01, 10, 11
    
    # Helper to get block rows
    def get_block(block_id, mat):
        # block_id: 0=00, 1=01, 2=10, 3=11
        r, c = divmod(block_id, 2)
        start_r = r * N_half * N_half
        start_c = c * N_half * N_half
        # This is not correct for flattening.
        # Flattening is row-major.
        # Block (r,c) starts at row r*N_half, col c*N_half
        # In flattened array of size N*N, index is i*N + j
        # Block (r,c) contains i in [r*N_half, (r+1)*N_half), j in [c*N_half, (c+1)*N_half)
        # We need to extract these rows from mat
        
        rows = []
        for i in range(r * N_half, (r + 1) * N_half):
            for j in range(c * N_half, (c + 1) * N_half):
                rows.append(i * N + j)
        return mat[rows, :]

    # We need to construct the linear combinations for the 7 products
    # The 7 products are computed on the blocks.
    # Each product has rank rank_half.
    # Total rank is 7 * rank_half.
    
    for k in range(7):
        offset = k * rank_half
        
        # Determine which blocks go into U, V, W for this k-th Strassen step
        # Based on the 2x2 formulas above
        
        if k == 0: # M1 = (A00+A11)(B00+B11)
            # U: A00 + A11
            U[get_block(0, U), offset:offset+rank_half] += U_half
            U[get_block(3, U), offset:offset+rank_half] += U_half
            # V: B00 + B11
            V[get_block(0, V), offset:offset+rank_half] += V_half
            V[get_block(3, V), offset:offset+rank_half] += V_half
            # W: C00 + C11
            W[get_block(0, W), offset:offset+rank_half] += W_half
            W[get_block(3, W), offset:offset+rank_half] += W_half
            
        elif k == 1: # M2 = (A10+A11)B00
            U[get_block(2, U), offset:offset+rank_half] += U_half
            U[get_block(3, U), offset:offset+rank_half] += U_half
            V[get_block(0, V), offset:offset+rank_half] += V_half
            W[get_block(2, W), offset:offset+rank_half] += W_half
            W[get_block(3, W), offset:offset+rank_half] += W_half

        elif k == 2: # M3 = A00(B01-B11)
            U[get_block(0, U), offset:offset+rank_half] += U_half
            V[get_block(1, V), offset:offset+rank_half] += V_half
            V[get_block(3, V), offset:offset+rank_half] -= V_half
            W[get_block(1, W), offset:offset+rank_half] += W_half
            W[get_block(3, W), offset:offset+rank_half] += W_half

        elif k == 3: # M4 = A11(B10-B00)
            U[get_block(3, U), offset:offset+rank_half] += U_half
            V[get_block(2, V), offset:offset+rank_half] += V_half
            V[get_block(0, V), offset:offset+rank_half] -= V_half
            W[get_block(0, W), offset:offset+rank_half] += W_half
            W[get_block(2, W), offset:offset+rank_half] += W_half

        elif k == 4: # M5 = (A00+A01)B11
            U[get_block(0, U), offset:offset+rank_half] += U_half
            U[get_block(1, U), offset:offset+rank_half] += U_half
            V[get_block(3, V), offset:offset+rank_half] += V_half
            W[get_block(0, W), offset:offset+rank_half] -= W_half
            W[get_block(1, W), offset:offset+rank_half] += W_half

        elif k == 5: # M6 = (A10-A00)(B00+B01)
            U[get_block(2, U), offset:offset+rank_half] += U_half
            U[get_block(0, U), offset:offset+rank_half] -= U_half
            V[get_block(0, V), offset:offset+rank_half] += V_half
            V[get_block(1, V), offset:offset+rank_half] += V_half
            W[get_block(1, W), offset:offset+rank_half] += W_half
            W[get_block(3, W), offset:offset+rank_half] += W_half

        elif k == 6: # M7 = (A01-A11)(B10+B11)
            U[get_block(1, U), offset:offset+rank_half] += U_half
            U[get_block(3, U), offset:offset+rank_half] -= U_half
            V[get_block(2, V), offset:offset+rank_half] += V_half
            V[get_block(3, V), offset:offset+rank_half] += V_half
            W[get_block(0, W), offset:offset+rank_half] += W_half
            W[get_block(2, W), offset:offset+rank_half] -= W_half

    return U, V, W
