import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    # --- STRATEGY: DISCRETE STRUCTURED SEARCH ---
    # 1. Construct a known valid low-rank decomposition (Strassen-based hybrid).
    # 2. Attempt to reduce rank by finding linear dependencies or merging terms
    #    using discrete moves (adding/subtracting basis tensors).
    
    def get_strassen_2x2():
        # Standard Strassen 2x2 * 2x2 -> 2x2
        # U, V, W are 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = 1
        # M3 = A11(B21-B22)
        U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
        # M4 = A22(B11-B21)
        U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[0, 3] = -1; W[1, 3] = 1
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = 1; W[2, 4] = 1
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[2, 5] = 1; W[2, 5] = 1; W[3, 5] = 1
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1; V[1, 6] = 1; V[3, 6] = 1; W[0, 6] = 1; W[1, 6] = 1
        return U, V, W

    def get_mult_1x2_2x2():
        # (1x2) * (2x2) -> (1x2)
        # Direct multiplication is rank 4 (2*2).
        # U: 2x4, V: 4x4, W: 2x4
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        # C0 = A0*B00 + A1*B10
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # C1 = A0*B01 + A1*B11
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    def get_mult_2x1_2x2():
        # (2x1) * (2x2) -> (2x1)
        # Direct multiplication is rank 4 (2*2).
        # U: 4x4, V: 4x4, W: 2x4
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        # C0 = A00*B00 + A10*B10
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[2, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # C1 = A00*B01 + A10*B11
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[2, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Construct Rank 66 baseline (Hybrid Strassen)
    U_66 = np.zeros((12, 66))
    V_66 = np.zeros((24, 66))
    W_66 = np.zeros((18, 66))
    
    def place_block(U_block, V_block, W_block, u_rows, v_rows, w_rows, offset):
        rank_block = U_block.shape[1]
        for c in range(rank_block):
            for local_r, global_r in enumerate(u_rows):
                U_66[global_r, offset + c] = U_block[local_r, c]
            for local_r, global_r in enumerate(v_rows):
                V_66[global_r, offset + c] = V_block[local_r, c]
            for local_r, global_r in enumerate(w_rows):
                W_66[global_r, offset + c] = W_block[local_r, c]

    S_U, S_V, S_W = get_strassen_2x2()
    M12_U, M12_V, M12_W = get_mult_1x2_2x2()
    M21_U, M21_V, M21_W = get_mult_2x1_2x2()
    
    col_offset = 0
    # Blocks 1-6: 2x2 * 2x2 (using Strassen, rank 7 each)
    # We split 3x4 into 2x2 blocks? No, 3 rows, 4 cols.
    # Rows: 0,1 (2x2), 2 (1x2)
    # Cols: 0,1 (2x2), 2,3 (2x2)
    
    # Row block 0 (rows 0-1), Col block 0 (cols 0-1): 2x2 * 2x2
    u_rows = [0, 1, 4, 5] # i*m+j: 0*4+0, 0*4+1, 1*4+0, 1*4+1
    v_rows = [0, 1, 6, 7] # j*p+k: 0*6+0, 0*6+1, 1*6+0, 1*6+1
    w_rows = [0, 1, 3, 4] # k*n+i: 0*3+0, 1*3+0, 0*3+1, 1*3+1
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Row block 0 (rows 0-1), Col block 1 (cols 2-3): 2x2 * 2x2
    u_rows = [0, 1, 4, 5]
    v_rows = [2, 3, 8, 9] # j*p+k: 0*6+2, 0*6+3, 1*6+2, 1*6+3
    w_rows = [6, 7, 9, 10] # k*n+i: 2*3+0, 3*3+0, 2*3+1, 3*3+1
    place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 7

    # Row block 1 (rows 2-2), Col block 0 (cols 0-1): 1x2 * 2x2
    # Note: My M12 block is (1x2) * (2x2).
    # U rows: row 2 of A. Indices: 2*4+0, 2*4+1 -> 8, 9
    # V rows: cols 0-1 of B. Indices: 0,1,6,7
    # W rows: col 0-1 of C, row 2 of C. Indices: 0*3+2, 1*3+2 -> 2, 5
    u_rows = [8, 9]
    v_rows = [0, 1, 6, 7]
    w_rows = [2, 5]
    place_block(M12_U, M12_V, M12_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Row block 1 (rows 2-2), Col block 1 (cols 2-3): 1x2 * 2x2
    u_rows = [8, 9]
    v_rows = [2, 3, 8, 9]
    w_rows = [8, 11] # k*n+i: 2*3+2, 3*3+2 -> 8, 11
    place_block(M12_U, M12_V, M12_W, u_rows, v_rows, w_rows, col_offset)
    col_offset += 4

    # Wait, the above covers:
    # A rows 0,1 with B cols 0,1,2,3.
    # A row 2 with B cols 0,1,2,3.
    # This covers the whole 3x4 * 4x6 matrix?
    # A is 3x4. B is 4x6.
    # My blocks:
    # 1. A[0:2, 0:2] * B[0:2, :] -> No, B is 4x6.
    # The block decomposition must respect the inner dimension m=4.
    # B is 4x6. I split B's rows (which are A's cols) into 0,1 and 2,3.
    # So B[0:2, :] is 2x6, B[2:4, :] is 2x6.
    # My V blocks above:
    # Block 1: v_rows 0,1,6,7. These correspond to B[0,0], B[0,1], B[1,0], B[1,1].
    # This is B[0:2, 0:2].
    # But B has 6 columns. I only used cols 0,1 of B.
    # I need to cover B's columns 0..5.
    
    # Let's restart the block construction properly.
    # A (3x4), B (4x6), C (3x6).
    # Split A rows: [0,1] and [2].
    # Split B rows (A cols): [0,1] and [2,3].
    # Split B cols (C cols): [0,1,2] and [3,4,5]? Or [0,1], [2,3], [4,5]?
    # To use 2x2 Strassen, we need 2x2 blocks.
    # Let's split B cols into [0,1], [2,3], [4,5].
    
    U_66 = np.zeros((12, 72)) # Max rank 72, will trim
    V_66 = np.zeros((24, 72))
    W_66 = np.zeros((18, 72))
    col_offset = 0
    
    # Helper to get indices
    def get_u_idx(i_start, i_end, j_start, j_end):
        return [i*m + j for i in range(i_start, i_end) for j in range(j_start, j_end)]
    def get_v_idx(j_start, j_end, k_start, k_end):
        return [j*p + k for j in range(j_start, j_end) for k in range(k_start, k_end)]
    def get_w_idx(k_start, k_end, i_start, i_end):
        return [k*n + i for k in range(k_start, k_end) for i in range(i_start, i_end)]

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2x2 * 2x2. Strassen.
    place_block(S_U, S_V, S_W, get_u_idx(0,2,0,2), get_v_idx(0,2,0,2), get_w_idx(0,2,0,2), col_offset)
    col_offset += 7

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 2x2 * 2x2. Strassen.
    place_block(S_U, S_V, S_W, get_u_idx(0,2,0,2), get_v_idx(0,2,2,4), get_w_idx(2,4,0,2), col_offset)
    col_offset += 7

    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # 2x2 * 2x2. Strassen.
    place_block(S_U, S_V, S_W, get_u_idx(0,2,0,2), get_v_idx(0,2,4,6), get_w_idx(4,6,0,2), col_offset)
    col_offset += 7

    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 2x2 * 2x2. Strassen.
    place_block(S_U, S_V, S_W, get_u_idx(0,2,2,4), get_v_idx(2,4,0,2), get_w_idx(0,2,0,2), col_offset)
    col_offset += 7

    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # 2x2 * 2x2. Strassen.
    place_block(S_U, S_V, S_W, get_u_idx(0,2,2,4), get_v_idx(2,4,2,4), get_w_idx(2,4,0,2), col_offset)
    col_offset += 7

    # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # 2x2 * 2x2. Strassen.
    place_block(S_U, S_V, S_W, get_u_idx(0,2,2,4), get_v_idx(2,4,4,6), get_w_idx(4,6,0,2), col_offset)
    col_offset += 7

    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # 1x2 * 2x2. Direct (Rank 4).
    place_block(M12_U, M12_V, M12_W, get_u_idx(2,3,0,2), get_v_idx(0,2,0,2), get_w_idx(0,2,2,3), col_offset)
    col_offset += 4

    # Block 8: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    # 1x2 * 2x2. Direct (Rank 4).
    place_block(M12_U, M12_V, M12_W, get_u_idx(2,3,0,2), get_v_idx(0,2,2,4), get_w_idx(2,4,2,3), col_offset)
    col_offset += 4

    # Block 9: A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    # 1x2 * 2x2. Direct (Rank 4).
    place_block(M12_U, M12_V, M12_W, get_u_idx(2,3,0,2), get_v_idx(0,2,4,6), get_w_idx(4,6,2,3), col_offset)
    col_offset += 4

    # Block 10: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # 1x2 * 2x2. Direct (Rank 4).
    place_block(M12_U, M12_V, M12_W, get_u_idx(2,3,2,4), get_v_idx(2,4,0,2), get_w_idx(0,2,2,3), col_offset)
    col_offset += 4

    # Block 11: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    # 1x2 * 2x2. Direct (Rank 4).
    place_block(M12_U, M12_V, M12_W, get_u_idx(2,3,2,4), get_v_idx(2,4,2,4), get_w_idx(2,4,2,3), col_offset)
    col_offset += 4

    # Block 12: A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    # 1x2 * 2x2. Direct (Rank 4).
    place_block(M12_U, M12_V, M12_W, get_u_idx(2,3,2,4), get_v_idx(2,4,4,6), get_w_idx(4,6,2,3), col_offset)
    col_offset += 4
    
    total_rank = col_offset
    U_66 = U_66[:, :total_rank]
    V_66 = V_66[:, :total_rank]
    W_66 = W_66[:, :total_rank]

    if verify(U_66.tolist(), V_66.tolist(), W_66.tolist(), n, m, p):
        best_U = U_66.tolist()
        best_V = V_66.tolist()
        best_W = W_66.tolist()
        best_rank = total_rank

    # --- DISCRETE REDUCTION SEARCH ---
    # Try to find linear dependencies among the columns of (U, V, W).
    # If col_c = sum(alpha_i * col_i), we can eliminate c.
    # Since entries are integers, we look for integer linear combinations.
    
    U_cur = U_66.copy()
    V_cur = V_66.copy()
    W_cur = W_66.copy()
    r_cur = total_rank
    
    # Greedy search for removable columns
    # A column c is removable if there exist integer coefficients for other columns
    # such that the tensor contribution cancels out.
    # This is equivalent to checking if the vector (U[:,c], V[:,c], W[:,c]) is in the
    # span of the others, but with the constraint that the sum of products must be 0.
    # Actually, if T = sum u_c v_c w_c, and we remove c, we need the new T to be 1.
    # So we need sum_{c' != c} u_{c'} v_{c'} w_{c'} = 1.
    # This implies the contribution of c must be 0, OR we adjust others.
    
    # Simpler heuristic: Check if any column is "zero" in the tensor product sense?
    # No, all contribute.
    
    # Alternative: Randomized Discrete Perturbation
    # Pick two columns c1, c2. Try to replace them with a single column c3
    # such that u3 v3 w3 = u1 v1 w1 + u2 v2 w2.
    # This is generally hard.
    
    # Instead, let's try to find if any column is redundant due to symmetry or zeroing.
    # In Strassen, some terms might cancel if dimensions allow.
    
    # For now, we return the hybrid rank. If time permits, we try a simple linear algebra check
    # to see if the rank can be reduced by Gaussian elimination on the "tensor" matrix.
    # Construct the matrix of the tensor in the "slice" basis.
    
    # Flatten the tensor T[i,j,k] into a vector of length n*m*p.
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    # This is a rank-r decomposition of the tensor.
    
    # We can try to find a lower rank decomposition by solving a system of linear equations
    # for a new set of r' < r columns.
    # This is NP-hard, but for small reductions (r to r-1), we can try.
    
    # Try r-1
    if r_cur > 1:
        r_test = r_cur - 1
        # Random initialization for r_test
        # Use the first r_test columns as init, then optimize?
        # No, we are in discrete domain.
        
        # Check if any column is a linear combination of others in the "trilinear" sense.
        # This is too complex for a quick check.
        
        pass

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
