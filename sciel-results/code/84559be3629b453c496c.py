"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen: C = A*B where A,B,C are 2x2
    # Indexing: A[i,j], B[j,k], C[i,k]
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)  
    # W has shape (2*2, 7) = (4, 7)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1                 # C00
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1                # B00
    W[2, 1] = 1                 # C10
    
    # M3
    U[0, 2] = 1                # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, -B11
    W[1, 2] = 1                 # C01
    
    # M4
    U[3, 3] = 1                # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, -B00
    W[0, 3] = 1                 # C00
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1                # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1   # B00, B01
    W[3, 5] = 1                 # C11
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1   # B10, B11
    W[0, 6] = 1                 # C00
    
    return U, V, W

def _block_multiply(n1, m1, n2, m2, p1, p2):
    """
    Multiply (n1+m1) x (m1+p1) ... wait, let me think about this differently.
    
    Actually, for block decomposition of matrix multiplication:
    We have A (n x m) * B (m x p) = C (n x p)
    
    Partition A into blocks, B into blocks, and use efficient multiplication for each block pair.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Try block decomposition
    # Partition: n=4=2+2, m=5=2+2+1, p=6=2+2+2
    
    # For each pair of row-blocks of A and column-blocks of B,
    # we need to multiply the corresponding sub-matrices.
    
    # Let's define the block structure:
    # A is 4x5, partition rows as [2,2], cols as [2,2,1]
    # B is 5x6, partition rows as [2,2,1], cols as [2,2,2]
    # C is 4x6, partition rows as [2,2], cols as [2,2,2]
    
    a_row_blocks = [2, 2]  # sum = 4
    a_col_blocks = [2, 2, 1]  # sum = 5
    b_row_blocks = [2, 2, 1]  # sum = 5 (must match a_col_blocks)
    b_col_blocks = [2, 2, 2]  # sum = 6
    
    # For each (i_block, j_block, k_block) where i_block indexes A rows,
    # j_block indexes A cols / B rows, k_block indexes B cols:
    # We multiply A[i_block, j_block] (size a_row_blocks[i_block] x a_col_blocks[j_block])
    # by B[j_block, k_block] (size b_row_blocks[j_block] x b_col_blocks[k_block])
    # Result contributes to C[i_block, k_block]
    
    # For each such sub-multiplication, we use either schoolbook or Strassen (if 2x2x2)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomposition(u_local, v_local, w_local, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, a_rows, a_cols, b_cols, c_rows, c_cols, n_total, m_total, p_total):
        """Add a local decomposition to the global U, V, W."""
        rank_local = u_local.shape[1]
        
        # Map local indices to global
        # A global index: (c_row_off + local_i) * m_total + (a_col_off + local_j)
        # But we need to map to the U matrix which has n*m rows
        # U row index for A[i,j] is i*m + j where i is global row, j is global col
        
        u_global = np.zeros((n * m, rank_local))
        v_global = np.zeros((m * p, rank_local))
        w_global = np.zeros((p * n, rank_local))
        
        for local_i in range(a_rows):
            for local_j in range(a_cols):
                global_i = c_row_off + local_i  # row in A/C
                global_j = a_col_off + local_j  # col in A / row in B
                u_idx = global_i * m + global_j
                u_global[u_idx, :] = u_local[local_i * a_cols + local_j, :]
        
        for local_j in range(a_cols):
            for local_k in range(b_cols):
                global_j = a_col_off + local_j  # row in B
                global_k = b_col_off + local_k  # col in B
                v_idx = global_j * p + global_k
                v_global[v_idx, :] = v_local[local_j * b_cols + local_k, :]
        
        for local_k in range(b_cols):
            for local_i in range(c_rows):
                global_k = b_col_off + local_k  # col in B / col in C
                global_i = c_row_off + local_i  # row in C
                w_idx = global_k * n + global_i  # W[k*n+i]
                w_global[w_idx, :] = w_local[local_k * c_rows + local_i, :]
        
        U_cols.append(u_global)
        V_cols.append(v_global)
        W_cols.append(w_global)
    
    # Precompute offsets
    a_row_offsets = [0]
    for s in a_row_blocks[:-1]:
        a_row_offsets.append(a_row_offsets[-1] + s)
    
    a_col_offsets = [0]
    for s in a_col_blocks[:-1]:
        a_col_offsets.append(a_col_offsets[-1] + s)
    
    b_col_offsets = [0]
    for s in b_col_blocks[:-1]:
        b_col_offsets.append(b_col_offsets[-1] + s)
    
    for i_b in range(len(a_row_blocks)):
        for j_b in range(len(a_col_blocks)):
            for k_b in range(len(b_col_blocks)):
                a_rows = a_row_blocks[i_b]
                a_cols = a_col_blocks[j_b]
                b_cols = b_col_blocks[k_b]
                
                a_row_off = a_row_offsets[i_b]
                a_col_off = a_col_offsets[j_b]
                b_row_off = a_col_offsets[j_b]  # same as a_col_offsets since they must match
                b_col_off = b_col_offsets[k_b]
                c_row_off = a_row_offsets[i_b]
                c_col_off = b_col_offsets[k_b]
                
                # Get decomposition for this sub-problem
                if a_rows == 2 and a_cols == 2 and b_cols == 2:
                    u_loc, v_loc, w_loc = _strassen_2x2()
                else:
                    # Schoolbook
                    rank = a_rows * a_cols * b_cols
                    u_loc = np.zeros((a_rows * a_cols, rank))
                    v_loc = np.zeros((a_cols * b_cols, rank))
                    w_loc = np.zeros((b_cols * a_rows, rank))
                    col = 0
                    for ii in range(a_rows):
                        for jj in range(a_cols):
                            for kk in range(b_cols):
                                u_loc[ii * a_cols + jj, col] = 1
                                v_loc[jj * b_cols + kk, col] = 1
                                w_loc[kk * a_rows + ii, col] = 1
                                col += 1
                
                add_decomposition(u_loc, v_loc, w_loc,
                                  a_row_off, a_col_off, b_row_off, b_col_off,
                                  c_row_off, c_col_off,
                                  a_rows, a_cols, b_cols, a_rows, b_cols,
                                  n, m, p)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Combine all columns
    if not U_cols:
        # Fallback to schoolbook
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
    else:
        U = np.hstack(U_cols)
        V = np.hstack(V_cols)
        W = np.hstack(W_cols)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
