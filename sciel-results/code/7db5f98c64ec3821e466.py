"""Rank-minimal tensor decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = -1
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = -1; W[2,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m times m x p. Rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    assert V.shape[1] == rank and W.shape[1] == rank
    assert U.shape[0] == n * m
    assert V.shape[0] == m * p
    assert W.shape[0] == p * n
    
    rng = np.random.RandomState(seed)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C = A @ B
    
    # Compute C from decomposition: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # More efficiently: C = sum_c (U[:,c] reshaped as n x m) * (V[:,c] reshaped as m x p) * ... 
    # Actually: C[i,k] = sum_j sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c]
    # Let's compute: for each c, contribution to C is (A * U_c_col) @ (V_c_col * B) ... 
    # Standard: C = sum_c (A diag(U_c) ) @ (diag(V_c) B) ... no.
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
    # So C[i,k] = sum_j T[i,j,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    C_comp = np.zeros((n, p))
    for c in range(rank):
        u_c = U[:, c]  # length n*m
        v_c = V[:, c]  # length m*p
        w_c = W[:, c]  # length p*n
        # u_c[i*m+j] = U[i,j,c], v_c[j*p+k] = V[j,k,c], w_c[k*n+i] = W[i,k,c]
        # C[i,k] += sum_j u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        # = w_c[k*n+i] * sum_j u_c[i*m+j] * v_c[j*p+k]
        # Let M_c[i,k] = sum_j u_c[i*m+j] * v_c[j*p+k]
        # Then C += M_c * w_c (broadcast: w_c indexed by [k*n+i])
        u_mat = u_c.reshape(n, m)
        v_mat = v_c.reshape(m, p)
        M_c = u_mat @ v_mat  # n x p, M_c[i,k] = sum_j u_c[i*m+j]*v_c[j*p+k]
        w_mat = w_c.reshape(p, n)  # w_mat[k, i] = w_c[k*n+i]
        C_comp += M_c * w_mat.T  # w_mat.T[i,k] = w_mat[k,i] = w_c[k*n+i]
    
    if not np.allclose(C, C_comp, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook fallback
    def schoolbook():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Block decomposition:
    # A (4x5) split into A_top (2x5), A_bot (2x5)
    # B (5x5) split into B_left (5x2), B_right (5x3)
    # C_top_left (2x2): A_top * B_left -> Strassen rank 7
    # C_top_right (2x3): A_top * B_right -> schoolbook rank 12
    # C_bot_left (2x2): A_bot * B_left -> Strassen rank 7
    # C_bot_right (2x3): A_bot * B_right -> schoolbook rank 12
    # Total rank: 7 + 12 + 7 + 12 = 38
    
    # Subproblem parameters:
    # A_top: rows 0-1 of A, all 5 cols -> indices in U: i*m+j for i in [0,1], j in [0,4]
    # A_bot: rows 2-3 of A, all 5 cols -> indices in U: i*m+j for i in [2,3], j in [0,4]
    # B_left: all 5 rows, cols 0-1 of B -> indices in V: j*p+k for j in [0,4], k in [0,1]
    # B_right: all 5 rows, cols 2-4 of B -> indices in V: j*p+k for j in [0,4], k in [2,4]
    # C_top_left: rows 0-1, cols 0-1 of C -> indices in W: k*n+i for k in [0,1], i in [0,1]
    # C_top_right: rows 0-1, cols 2-4 of C -> indices in W: k*n+i for k in [2,4], i in [0,1]
    # C_bot_left: rows 2-3, cols 0-1 of C -> indices in W: k*n+i for k in [0,1], i in [2,3]
    # C_bot_right: rows 2-3, cols 2-4 of C -> indices in W: k*n+i for k in [2,4], i in [2,3]
    
    # Build the decomposition
    # Subproblem 1: C_top_left = A_top * B_left (2x2x2, Strassen)
    U1, V1, W1 = _strassen_2x2()
    
    # Subproblem 2: C_top_right = A_top * B_right (2x3x2, schoolbook)
    U2, V2, W2 = _schoolbook(2, 3, 2)
    
    # Subproblem 3: C_bot_left = A_bot * B_left (2x2x2, Strassen)
    U3, V3, W3 = _strassen_2x2()
    
    # Subproblem 4: C_bot_right = A_bot * B_right (2x3x2, schoolbook)
    U4, V4, W4 = _schoolbook(2, 3, 2)
    
    total_rank = U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1]
    # = 7 + 12 + 7 + 12 = 38
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Subproblem 1: A_top (2x5) * B_left (5x2) -> C_top_left (2x2)
    # U1[i*2+j, c] corresponds to A_top[i,j] = A[i,j] for i in [0,1], j in [0,4]
    # In full U: index = i*m + j = i*5 + j
    # V1[j*2+k, c] corresponds to B_left[j,k] = B[j,k] for j in [0,4], k in [0,1]
    # In full V: index = j*p + k = j*5 + k
    # W1[k*2+i, c] corresponds to C_top_left[k,i] = C[k,i] for k in [0,1], i in [0,1]
    # In full W: index = k*n + i = k*4 + i
    
    col_offset = 0
    for c in range(U1.shape[1]):
        # Map U1 indices to full U
        for idx in range(U1.shape[0]):
            if U1[idx, c] != 0:
                i_sub = idx // 2
                j_sub = idx % 2
                full_idx = i_sub * m + j_sub  # i_sub in [0,1], j_sub in [0,4]
                U[full_idx, col_offset + c] = U1[idx, c]
        
        # Map V1 indices to full V
        for idx in range(V1.shape[0]):
            if V1[idx, c] != 0:
                j_sub = idx // 2
                k_sub = idx % 2
                full_idx = j_sub * p + k_sub  # j_sub in [0,4], k_sub in [0,1]
                V[full_idx, col_offset + c] = V1[idx, c]
        
        # Map W1 indices to full W
        for idx in range(W1.shape[0]):
            if W1[idx, c] != 0:
                k_sub = idx // 2
                i_sub = idx % 2
                full_idx = k_sub * n + i_sub  # k_sub in [0,1], i_sub in [0,1]
                W[full_idx, col_offset + c] = W1[idx, c]
    
    col_offset += U1.shape[1]
    
    # Subproblem 2: A_top (2x5) * B_right (5x3) -> C_top_right (2x3)
    # U2[i*3+j, c] corresponds to A_top[i,j] = A[i,j] for i in [0,1], j in [0,4]
    # Wait, U2 is for 2x3 schoolbook: U2 has shape (2*3, 12)
    # U2[i*3+j, c] where i in [0,1], j in [0,2] -> but A_top has 5 cols, not 3!
    # I need to be more careful.
    
    # Actually, for the schoolbook of 2x3 (n_sub=2, m_sub=3, p_sub=2):
    # U2[i*m_sub + j, c] = U2[i*3+j, c] for i in [0,1], j in [0,2]
    # This represents A_sub[i,j] where A_sub is 2x3
    # But A_top is 2x5. The "3" here is the number of columns of B_right, not A_top.
    # 
    # Let me reconsider. For C = A * B where A is n x m and B is m x p:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # For the subproblem C_top_right = A_top * B_right:
    # A_top is 2x5, B_right is 5x3, C_top_right is 2x3
    # So n_sub=2, m_sub=5, p_sub=3
    # U2 should have shape (2*5, rank2), V2 shape (5*3, rank2), W2 shape (3*2, rank2)
    
    # I made an error. Let me redo.
    # For 2x5x3: schoolbook rank = 2*5*3 = 30. That's worse than I thought.
    # 
    # Let me reconsider the block decomposition more carefully.
    # 
    # A is 4x5, B is 5x5, C is 4x5
    # Split A into A_top (2x5) and A_bot (2x5)
    # Split B into B_top (5x2) and B_bot (5x3) -- splitting columns of B
    # 
    # C_top_left = A_top * B_top: (2x5)(5x2) = 2x2, n=2,m=5,p=2, schoolbook rank=20
    # C_top_right = A_top * B_bot: (2x5)(5x3) = 2x3, n=2,m=5,p=3, schoolbook rank=30
    # C_bot_left = A_bot * B_top: (2x5)(5x2) = 2x2, n=2,m=5,p=2, schoolbook rank=20
    # C_bot_right = A_bot * B_bot: (2x5)(5x3) = 2x3, n=2,m=5,p=3, schoolbook rank=30
    # Total: 20+30+20+30 = 100. Same as schoolbook!
    
    # The issue is that the middle dimension (5) is shared. I need to split that too.
    # 
    # Better approach: Split the middle dimension as well.
    # A is 4x5, B is 5x5
    # Split into blocks:
    # A = [A11 A12] where A11 is 4x2, A12 is 4x3
    # B = [B11; B21] where B11 is 2x5, B21 is 3x5
    # 
    # C = A11*B11 + A12*B21
    # 
    # This doesn't help directly. Let me think differently.
    
    # Alternative: Use the fact that 5 = 2 + 3 and 4 = 2 + 2
    # Split A into 4 blocks of size (2x2) and (2x3):
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # Split B into 4 blocks:
    # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # 
    # C = A*B has blocks:
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (2x2)
    # C22 = A21*B12 + A22*B22 (2x3)
    
    # This is getting complex. Let me just use a different strategy.
    
    # Key insight: The rank of matrix multiplication tensor [n,m,p] satisfies:
    # R[n,m,p] <= R[n1,m,p] + R[n-n1,m,p] (splitting rows of A)
    # R[n,m,p] <= R[n,m1,p] + R[n,m-m1,p] (splitting cols of A / rows of B)
    # R[n,m,p] <= R[n,m,p1] + R[n,m,p-p1] (splitting cols of B)
    # 
    # But these are additive, so they don't reduce below n*m*p unless we have
    # subproblems with known lower rank.
    # 
    # The only known subproblem with lower rank than schoolbook is 2x2x2 (rank 7 vs 8).
    # 
    # So the best I can do is:
    # Split 4x5x5 into pieces where some are 2x2x2 (rank 7) and others are schoolbook.
    # 
    # Let me split A into two 2x5 blocks, and B into a 5x2 and 5x3 block.
    # But as I showed, the middle dimension is 5, so subproblems are 2x5x2 and 2x5x3.
    # 
    # For 2x5x2: Can I do better than schoolbook rank 20?
    # 2x5x2 = (2x2x2) + (2x3x2). Rank <= 7 + 12 = 19.
    # For 2x5x3: Can I do better than schoolbook rank 30?
    # 2x5x3 = (2x2x3) + (2x3x3). 
    # 2x2x3: rank <= 2*R[2,2,3]... no, that's not right.
    # 2x2x3 schoolbook rank = 12. Can we do better?
    # 2x3x3 schoolbook rank = 18.
    # 
    # This is getting complicated. Let me just try to get a valid decomposition
    # with rank 38 using a different blocking strategy.
    
    # New approach: Direct construction with rank 38
    # 
    # Actually, let me just return the schoolbook decomposition which is valid
    # and has rank 100. The score is seed/value, and since the seed program
    # also returns rank 100, the score is 1.0. To improve, I need rank < 100.
    # 
    # Let me try a simpler block approach:
    # Split the 4x5x5 into:
    # - One 2x2x2 block (rank 7)
    # - Remaining parts as schoolbook
    # 
    # If I isolate a 2x2x2 subproblem and handle the rest with schoolbook:
    # The 2x2x2 contributes 7 instead of 8, saving 1.
    # Total: 100 - 1 = 99.
    
    # Let me construct this:
    # Use the first 2 rows of A, first 2 cols of B (which gives first 2 rows of C)
    # Wait, that's not quite right either.
    
    # Let me just do a clean implementation:
    # Take the full 4x5x5 schoolbook, but replace one 2x2x2 sub-block with Strassen.
    
    # In the schoolbook, each (i,j,k) triple gets its own column.
    # For the 2x2x2 block at i in [0,1], j in [0,1], k in [0,1]:
    # There are 8 columns. Replace with 7 Strassen columns.
    # Net saving: 1 column.
    
    # Total rank: 100 - 8 + 7 = 99.
    
    U_full = np.zeros((n * m, n * m * p), dtype=float)
    V_full = np.zeros((m * p, n * m * p), dtype=float)
    W_full = np.zeros((p * n, n * m * p), dtype=float)
    
    # First, set up all schoolbook entries
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_full[i * m + j, col] = 1
                V_full[j * p + k, col] = 1
                W_full[k * n + i, col] = 1
                col += 1
    
    # Now, replace the 2x2x2 block (i in [0,1], j in [0,1], k in [0,1]) with Strassen
    # The 8 columns corresponding to this block are at positions:
    # col(i,j,k) = i*m*p + j*p + k = i*25 + j*5 + k
    # For i in [0,1], j in [0,1], k in [0,1]:
    # cols: 0,1,2,3, 25,26,27,28
    
    block_cols = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                block_cols.append(i * m * p + j * p + k)
    
    # Zero out these columns
    for c in block_cols:
        U_full[:, c] = 0
        V_full[:, c] = 0
        W_full[:, c] = 0
    
    # Insert Strassen at the first block column position
    U_s, V_s, W_s = _strassen_2x2()
    base_col = block_cols[0]  # column 0
    
    for c in range(7):
        # Map Strassen indices to full indices
        # U_s[i*2+j, c] -> U_full[i*5+j, base_col+c] for i in [0,1], j in [0,1]
        for idx in range(4):
            i_sub = idx // 2
            j_sub = idx % 2
            U_full[i_sub * m + j_sub, base_col + c] = U_s[idx, c]
        
        # V_s[j*2+k, c] -> V_full[j*5+k, base_col+c] for j in [0,1], k in [0,1]
        for idx in range(4):
            j_sub = idx // 2
            k_sub = idx % 2
            V_full[j_sub * p + k_sub, base_col + c] = V_s[idx, c]
        
        # W_s[k*2+i, c] -> W_full[k*4+i, base_col+c] for k in [0,1], i in [0,1]
        for idx in range(4):
            k_sub = idx // 2
            i_sub = idx % 2
            W_full[k_sub * n + i_sub, base_col + c] = W_s[idx, c]
    
    # Now we have rank 100 but 7 of the columns are Strassen and the 8th block column is zero.
    # We can remove the zero column to get rank 99.
    # Actually, the zero column is at position block_cols[7] = 28.
    # Let's just keep rank 100 for simplicity, or remove the zero column.
    
    # Remove zero columns
    nonzero_cols = [c for c in range(n * m * p) if 
                    np.any(U_full[:, c] != 0) or 
                    np.any(V_full[:, c] != 0) or 
                    np.any(W_full[:, c] != 0)]
    
    U_final = U_full[:, nonzero_cols]
    V_final = V_full[:, nonzero_cols]
    W_final = W_full[:, nonzero_cols]
    
    final_rank = U_final.shape[1]
    
    # Verify
    if _verify(U_final, V_final, W_final, n, m, p):
        result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback to pure schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return result
# EVOLVE-BLOCK-END
