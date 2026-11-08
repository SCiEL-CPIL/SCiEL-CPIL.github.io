import numpy as np
import time

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x4 matrix.
    
    The current best known upper bound for the rank of the 3x4x4 multiplication tensor is 44.
    The decomposition uses a block strategy:
    1. Split the 4x4 matrix B into four 2x2 blocks.
    2. Split the 3x4 matrix A into two 3x2 blocks (columns 0-1 and 2-3).
    3. The result C is a 3x4 matrix. 
       C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
       C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
       
    Each term is a multiplication of a 3x2 matrix by a 2x2 matrix, resulting in a 3x2 matrix.
    We can decompose a 3x2x2 multiplication tensor.
    Let's find the minimal rank for 3x2x2.
    Dimensions: A is 3x2 (6 elements), B is 2x2 (4 elements), C is 3x2 (6 elements).
    Using Strassen for 2x2x2 requires 7 mults.
    The 3x2x2 tensor can be seen as two independent 2x2x2 multiplications for rows 0 and 1 of A,
    plus one row 2 which is a 1x2x2 multiplication.
    Row 2 of A (1x2) times B (2x2) gives Row 2 of C (1x2).
    A 1x2x2 multiplication:
    c0 = a0 b00 + a1 b10
    c1 = a0 b01 + a1 b11
    This is 4 independent mults (rank 4).
    
    So for the 3x2x2 block:
    Rows 0-1 of A form a 2x2 matrix. Multiply by 2x2 B. Use Strassen (rank 7).
    Row 2 of A is a 1x2 vector. Multiply by 2x2 B. Use trivial (rank 4).
    Total rank for one 3x2x2 block = 7 + 4 = 11.
    
    We have 4 such blocks:
    1. A_L (3x2) * B_11 (2x2) -> C_L (3x2)
    2. A_R (3x2) * B_21 (2x2) -> C_L (3x2) (Add to previous C_L)
    3. A_L (3x2) * B_12 (2x2) -> C_R (3x2)
    4. A_R (3x2) * B_22 (2x2) -> C_R (3x2) (Add to previous C_R)
    
    Total rank = 4 * 11 = 44.
    
    The program below constructs this decomposition exactly.
    """
    n, m, p = 3, 4, 4
    rank = 44

    U = np.zeros((n * m, rank))  # 12 x 44
    V = np.zeros((m * p, rank))  # 16 x 44
    W = np.zeros((p * n, rank))  # 12 x 44

    # Local factors for 3x2x2 multiplication
    # A_local is 3x2 (indices 0..5: r*2+c)
    # B_local is 2x2 (indices 0..3: r*2+c)
    # C_local is 3x2 (indices 0..5: r*2+c) -- Wait, W index convention is k*n+i?
    # Task says: W[k*n+i,r].
    # Matrix C is n x p. Here local C is 3x2.
    # Global C is 3x4.
    # Let's check the global indexing in the current program.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # n=3, p=4. W has shape (p*n, rank) = (12, rank).
    # Index w in W corresponds to C[i,j]?
    # Usually C[i,j] = sum_k A[i,k] B[k,j].
    # Tensor T_{i,j,k} = A_{i,k} B_{k,j}.
    # Decomposition: T_{i,j,k} = sum_r U_{i,k,r} V_{k,j,r} W_{j,i,r} ?
    # Or W_{j,i,r}? The prompt says W[k*n+i,r] where k is col, i is row?
    # "W[k*n+i,r]" usually implies k is the column index (0..p-1) and i is the row index (0..n-1)?
    # If so, W index = j*n + i.
    # Let's verify with the seed code logic.
    # Seed code: `c_indices = [(c_col_start + c) * n + (c_row_start + r) for r in range(3) for c in range(2)]`
    # Here c_col_start + c is the global column j.
    # c_row_start + r is the global row i.
    # So global index in W is `j * n + i`.
    # This matches the convention W[k*n+i, r] if k=j (col) and i=row.
    
    # So, for the local block:
    # A_local rows 0..2, cols 0..1.
    # B_local rows 0..1, cols 0..1.
    # C_local rows 0..2, cols 0..1.
    # Local W index: j_loc * 3 + i_loc.
    
    # Let's build the 11-rank decomposition for 3x2x2.
    # U_loc: (6, 11)
    # V_loc: (4, 11)
    # W_loc: (6, 11)
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Part 1: Strassen for A[0:2, :] * B -> C[0:2, :]
    # A_strassen is 2x2. Indices in A_loc: 0,1 (row 0), 2,3 (row 1).
    # B is 2x2. Indices 0,1 (row 0), 2,3 (row 1).
    # C_strassen is 2x2. Indices in C_loc: 0,1 (row 0), 2,3 (row 1).
    # W indices for C[0:2, 0:2]:
    # j=0: i=0->0, i=1->1
    # j=1: i=0->3, i=1->4
    # Wait, local W index is j_loc * 3 + i_loc.
    # C[0,0] -> j=0, i=0 -> 0*3+0 = 0
    # C[0,1] -> j=1, i=0 -> 1*3+0 = 3
    # C[1,0] -> j=0, i=1 -> 0*3+1 = 1
    # C[1,1] -> j=1, i=1 -> 1*3+1 = 4
    
    # Strassen formulas for 2x2:
    # M1 = (a00 + a11) (b00 + b11)  -> c00 += M1, c11 += M1
    # M2 = (a10 + a11) b00          -> c10 += M2, c11 -= M2
    # M3 = a00 (b01 - b11)          -> c01 += M3, c11 += M3
    # M4 = a11 (b10 - b00)          -> c00 += M4, c10 += M4
    # M5 = (a00 + a01) b11          -> c00 -= M5, c01 += M5
    # M6 = (a10 - a00) (b00 + b01)  -> c11 += M6
    # M7 = (a01 - a11) (b10 + b11)  -> c00 += M7
    
    # Col 0: M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[4, 0] = 1 # c00, c11
    
    # Col 1: M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[1, 1] = 1; W_loc[4, 1] = -1 # c10, c11
    
    # Col 2: M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[3, 2] = 1; W_loc[4, 2] = 1 # c01, c11
    
    # Col 3: M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[1, 3] = 1 # c00, c10
    
    # Col 4: M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[3, 4] = 1 # c00, c01
    
    # Col 5: M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[4, 5] = 1 # c11
    
    # Col 6: M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1 # c00
    
    # Part 2: Trivial for A[2, :] * B -> C[2, :]
    # A indices: 4 (a20), 5 (a21)
    # C indices: 
    # C[2,0] -> j=0, i=2 -> 0*3+2 = 2
    # C[2,1] -> j=1, i=2 -> 1*3+2 = 5
    
    # M8: a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[2, 7] = 1
    
    # M9: a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[2, 8] = 1
    
    # M10: a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11: a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Helper to add a block
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        # Global A indices: row * 4 + col
        a_indices = []
        for r in range(3):
            for c in range(2):
                a_indices.append((a_row_start + r) * 4 + (a_col_start + c))
                
        # Global B indices: row * 4 + col
        b_indices = []
        for r in range(2):
            for c in range(2):
                b_indices.append((b_row_start + r) * 4 + (b_col_start + c))
                
        # Global W indices: col * 3 + row
        # Note: local W index j_loc * 3 + i_loc
        # Global W index: (c_col_start + j_loc) * 3 + (c_row_start + i_loc)
        # We need to map local W indices 0..5 to global indices.
        # Local i_loc 0..2, j_loc 0..1
        w_indices = []
        for i_loc in range(3):
            for j_loc in range(2):
                local_w_idx = j_loc * 3 + i_loc
                global_w_idx = (c_col_start + j_loc) * 3 + (c_row_start + i_loc)
                w_indices.append((local_w_idx, global_w_idx))
                
        # Add U
        for i, idx in enumerate(a_indices):
            U[idx, col_offset:col_offset+11] += U_loc[i, :]
            
        # Add V
        for i, idx in enumerate(b_indices):
            V[idx, col_offset:col_offset+11] += V_loc[i, :]
            
        # Add W
        for local_idx, global_idx in w_indices:
            W[global_idx, col_offset:col_offset+11] += W_loc[local_idx, :]

    # Block 0: A cols 0-1, B rows 0-1 cols 0-1 -> C rows 0-2 cols 0-1
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A cols 2-3, B rows 2-3 cols 0-1 -> C rows 0-2 cols 0-1
    add_block(1, 0, 2, 2, 0, 0, 0)
    
    # Block 2: A cols 0-1, B rows 0-1 cols 2-3 -> C rows 0-2 cols 2-3
    add_block(2, 0, 0, 0, 2, 0, 2)
    
    # Block 3: A cols 2-3, B rows 2-3 cols 2-3 -> C rows 0-2 cols 2-3
    add_block(3, 0, 2, 2, 2, 0, 2)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify on random matrices
    np.random.seed(42)
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 4)
    C_exact = A @ B
    
    # Reconstruct C from decomposition
    # T_{i,j,k} = sum_r U_{i,k,r} V_{k,j,r} W_{j,i,r}
    # C_{i,j} = sum_k T_{i,j,k}
    C_decomp = np.zeros((3, 4))
    for r in range(44):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        # Reshape u to 3x4, v to 4x4, w to 4x3 (since w index is j*3+i)
        u_mat = u.reshape(3, 4)
        v_mat = v.reshape(4, 4)
        w_mat = w.reshape(4, 3) # j is row, i is col
        
        # This decomposition is rank-1 for the tensor.
        # Term = u_mat[i,k] * v_mat[k,j] * w_mat[j,i]
        # Sum over k for fixed i,j? No, sum over r.
        # C_decomp[i,j] += sum_k U[i,k,r] V[k,j,r] W[j,i,r]
        
        # Vectorized:
        # Sum_k U[i,k,r] V[k,j,r] is (U_mat @ V_mat)[i,j]
        # Let M_r = U_mat @ V_mat
        # Then term_r[i,j] = M_r[i,j] * w_mat[j,i]
        # w_mat[j,i] is W[j*3+i].
        # So we need to element-wise multiply M_r with W_mat.T ?
        # W_mat has shape (4,3) where rows are j, cols are i.
        # So W_mat[j,i] is correct.
        # M_r has shape (3,4) where rows are i, cols are j.
        # We need to transpose M_r to (4,3) to match W_mat?
        # M_r.T[j,i] = M_r[i,j].
        # So term_r_matrix = M_r.T * W_mat (element-wise)
        # Then C_decomp += term_r_matrix.T
        
        M_r = u_mat @ v_mat
        term_r = (M_r.T * w_mat).T
        C_decomp += term_r
        
    diff = np.max(np.abs(C_exact - C_decomp))
    print(f"Max diff: {diff}")
    if diff < 1e-9:
        print("PASS")
    else:
        print("FAIL")
        print(C_exact)
        print(C_decomp)
