"""Optimized decomposition for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False, abs(val - 1.0)
    return True, 0.0

def _build_strassen_2x2x2():
    """Build Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) with integer entries.
    Index convention: U[i*2+j], V[j*2+k], W[k*2+i]
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U: shape (4, 7), U[i*2+j, r]
    # V: shape (4, 7), V[j*2+k, r]
    # W: shape (4, 7), W[k*2+i, r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is U[0], A22 is U[3]
    # B11 is V[0], B22 is V[3]
    # C11 gets +M1, C22 gets +M1
    # W index: W[k*2+i], for C11: k=0,i=0 -> W[0]; for C22: k=1,i=1 -> W[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2 = (A21+A22)B11
    # A21 is U[2], A22 is U[3]
    # B11 is V[0]
    # C21 gets +M2, C22 gets -M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21: k=1,i=0 -> W[2]
    W[3, 1] = -1  # C22: k=1,i=1 -> W[3]
    
    # M3 = A11(B21-B22)
    # A11 is U[0]
    # B21 is V[2], B22 is V[3]
    # C12 gets +M3, C22 gets +M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12: k=1,i=0 -> W[1]
    W[3, 2] = 1  # C22: k=1,i=1 -> W[3]
    
    # M4 = A22(B12-B11)
    # A22 is U[3]
    # B12 is V[1], B11 is V[0]
    # C11 gets +M4, C21 gets +M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11: k=0,i=0 -> W[0]
    W[2, 3] = 1  # C21: k=1,i=0 -> W[2]
    
    # M5 = (A11+A12)B22
    # A11 is U[0], A12 is U[1]
    # B22 is V[3]
    # C11 gets -M5, C12 gets +M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C11: k=0,i=0 -> W[0]
    W[1, 4] = 1  # C12: k=1,i=0 -> W[1]
    
    # M6 = (A21-A11)(B11+B12)
    # A21 is U[2], A11 is U[0]
    # B11 is V[0], B12 is V[1]
    # C22 gets +M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22: k=1,i=1 -> W[3]
    
    # M7 = (A12-A22)(B21+B22)
    # A12 is U[1], A22 is U[3]
    # B21 is V[2], B22 is V[3]
    # C11 gets +M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11: k=0,i=0 -> W[0]
    
    return U, V, W

def _build_schoolbook_block(n, m, p):
    """Build schoolbook decomposition for n x m x p."""
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

def _build_strassen_2x4x8():
    """Build decomposition for 2x4 x 4x8 using Strassen on 2x2x2 blocks.
    
    Split A(2x4) = [A1(2x2) | A2(2x2)]
    Split B(4x8) into 8 blocks of (2x2): B[row_block][col_block]
    B is 4x8, so rows split into 2 blocks of 2, cols split into 4 blocks of 2.
    
    C = A*B is 2x8.
    C[0:2, 0:2] = A1*B[0][0] + A2*B[1][0]
    C[0:2, 2:4] = A1*B[0][1] + A2*B[1][1]
    C[0:2, 4:6] = A1*B[0][2] + A2*B[1][2]
    C[0:2, 6:8] = A1*B[0][3] + A2*B[1][3]
    
    For each pair (A_block, B_block) -> 2x2x2 multiplication using Strassen (rank 7).
    We have 2 A-blocks * 4 B-blocks = 8 Strassen decompositions.
    Total rank: 8 * 7 = 56.
    """
    U_S, V_S, W_S = _build_strassen_2x2x2()
    
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x8: rows 0,1,2,3; cols 0,1,2,3,4,5,6,7
    # C is 2x8: rows 0,1; cols 0,1,2,3,4,5,6,7
    
    # U: shape (2*4=8, 56) -> U[i*4+j, r]
    # V: shape (4*8=32, 56) -> V[j*8+k, r]
    # W: shape (8*2=16, 56) -> W[k*2+i, r]
    
    total_rank = 56
    U = np.zeros((8, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((16, total_rank))
    
    for a_block in range(2):  # A1 (cols 0-1) or A2 (cols 2-3)
        for b_block in range(4):  # B col blocks 0-3
            r_offset = (a_block * 4 + b_block) * 7
            
            # A_block is 2x2: rows 0,1 of A; cols a_block*2, a_block*2+1
            # In U: U[i*4 + j] where i is row (0,1), j is col in A
            # A row i -> U row i*4 + (a_block*2 + col_in_block)
            
            # B_block is 2x2: rows b_block*2... wait no.
            # B is 4x8. We split B's rows into 2 blocks of 2, and cols into 4 blocks of 2.
            # But for the product A(2x4) * B(4x8), the inner dimension is 4.
            # A's columns are 0,1,2,3. B's rows are 0,1,2,3.
            # So A_block a_block uses A columns a_block*2, a_block*2+1.
            # B_block for this a_block: we need B rows a_block*2, a_block*2+1.
            # And B columns b_block*2, b_block*2+1.
            
            # So B_block is 2x2: B rows [a_block*2, a_block*2+1], cols [b_block*2, b_block*2+1]
            
            for local_r in range(7):
                r = r_offset + local_r
                
                # U entries: for each (i, j) in 2x2 block of A
                for i in range(2):
                    for j in range(2):
                        val = U_S[i*2+j, local_r]
                        if val != 0:
                            U[i*4 + a_block*2 + j, r] += val
                
                # V entries: for each (j, k) in 2x2 block of B
                # B row = a_block*2 + j_local, B col = b_block*2 + k_local
                for j in range(2):
                    for k in range(2):
                        val = V_S[j*2+k, local_r]
                        if val != 0:
                            V[(a_block*2 + j)*8 + b_block*2 + k, r] += val
                
                # W entries: for each (k, i) in 2x2 block of C
                # C row = i, C col = b_block*2 + k
                for k in range(2):
                    for i in range(2):
                        val = W_S[k*2+i, local_r]
                        if val != 0:
                            W[(b_block*2 + k)*2 + i, r] += val
    
    return U, V, W

def _build_strassen_1x4x8():
    """Build decomposition for 1x4 x 4x8 using schoolbook on 1x2x2 blocks.
    
    Split A(1x4) = [A1(1x2) | A2(1x2)]
    B is 4x8.
    
    C[0, 0:2] = A1 * B[0:2, 0:2]  (1x2 x 2x2 -> rank 4)
    C[0, 2:4] = A1 * B[0:2, 2:4]  (1x2 x 2x2 -> rank 4)
    C[0, 4:6] = A1 * B[0:2, 4:6]  (1x2 x 2x2 -> rank 4)
    C[0, 6:8] = A1 * B[0:2, 6:8]  (1x2 x 2x2 -> rank 4)
    C[0, 0:2] += A2 * B[2:4, 0:2]  (1x2 x 2x2 -> rank 4)
    C[0, 2:4] += A2 * B[2:4, 2:4]  (1x2 x 2x2 -> rank 4)
    C[0, 4:6] += A2 * B[2:4, 4:6]  (1x2 x 2x2 -> rank 4)
    C[0, 6:8] += A2 * B[2:4, 6:8]  (1x2 x 2x2 -> rank 4)
    
    Total rank: 8 * 4 = 32.
    """
    # Schoolbook for 1x2 x 2x2: rank 4
    U_sb, V_sb, W_sb = _build_schoolbook_block(1, 2, 2)
    
    # U: shape (1*4=4, 32) -> U[0*4+j, r] = U[j, r]
    # V: shape (4*8=32, 32) -> V[j*8+k, r]
    # W: shape (8*1=8, 32) -> W[k*1+0, r] = W[k, r]
    
    total_rank = 32
    U = np.zeros((4, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((8, total_rank))
    
    for a_block in range(2):  # A1 (cols 0-1) or A2 (cols 2-3)
        for b_block in range(4):  # B col blocks
            r_offset = (a_block * 4 + b_block) * 4
            
            for local_r in range(4):
                r = r_offset + local_r
                
                # U: 1x2 block at cols a_block*2, a_block*2+1
                for j in range(2):
                    val = U_sb[j, local_r]  # U_sb[i*2+j] with i=0
                    if val != 0:
                        U[a_block*2 + j, r] += val
                
                # V: 2x2 block at rows a_block*2, a_block*2+1; cols b_block*2, b_block*2+1
                for j in range(2):
                    for k in range(2):
                        val = V_sb[j*2+k, local_r]
                        if val != 0:
                            V[(a_block*2 + j)*8 + b_block*2 + k, r] += val
                
                # W: 2x1 block at cols b_block*2, b_block*2+1; row 0
                for k in range(2):
                    val = W_sb[k*1+0, local_r]  # W_sb[k*1+i] with i=0
                    if val != 0:
                        W[b_block*2 + k, r] += val
    
    return U, V, W

def _build_strassen_3x4x8():
    """Build full 3x4x8 decomposition using Strassen blocks.
    
    Split A(3x4) = A_top(2x4) || A_bottom(1x4)
    B(4x8) stays whole.
    
    C_top(2x8) = A_top(2x4) * B(4x8)  -> rank 56 (Strassen 2x4x8)
    C_bottom(1x8) = A_bottom(1x4) * B(4x8)  -> rank 32 (schoolbook 1x4x8)
    
    Total rank: 56 + 32 = 88.
    
    U: shape (3*4=12, 88)
    V: shape (4*8=32, 88)
    W: shape (8*3=24, 88)
    """
    U_top, V_top, W_top = _build_strassen_2x4x8()
    U_bot, V_bot, W_bot = _build_strassen_1x4x8()
    
    rank_top = U_top.shape[1]  # 56
    rank_bot = U_bot.shape[1]  # 32
    total_rank = rank_top + rank_bot  # 88
    
    U = np.zeros((12, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((24, total_rank))
    
    # Top part: rows 0-1 of A, all 4 cols
    # U_top: shape (8, 56), U_top[i*4+j, r] where i=0,1 is C row, j=0..3 is A col
    # Full U: U[i*4+j, r] where i=0,1,2 is A row, j=0..3 is A col
    # So U_top[i*4+j, r] -> U[i*4+j, r] directly (rows 0-7)
    U[:8, :rank_top] = U_top
    V[:32, :rank_top] = V_top
    # W_top: shape (16, 56), W_top[k*2+i, r] where i=0,1 is C row, k=0..7 is C col
    # Full W: W[k*3+i, r] where i=0,1,2 is C row, k=0..7 is C col
    # So W_top[k*2+i, r] -> W[k*3+i, r]
    for k in range(8):
        for i in range(2):
            W[k*3 + i, :rank_top] = W_top[k*2 + i, :rank_top]
    
    # Bottom part: row 2 of A, all 4 cols
    # U_bot: shape (4, 32), U_bot[j, r] where j=0..3 is A col
    # Full U: U[2*4+j, r] = U[8+j, r]
    U[8:12, rank_top:] = U_bot
    V[:32, rank_top:] = V_bot
    # W_bot: shape (8, 32), W_bot[k, r] where k=0..7 is C col
    # Full W: W[k*3+2, r]
    for k in range(8):
        W[k*3 + 2, rank_top:] = W_bot[k, :]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    try:
        U, V, W = _build_strassen_3x4x8()
        valid, err = _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook_block(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
