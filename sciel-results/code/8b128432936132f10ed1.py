import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 5
    
    # Schoolbook initialization
    rank0 = n * m * p
    U = np.zeros((n*m, rank0), dtype=float)
    V = np.zeros((m*p, rank0), dtype=float)
    W = np.zeros((p*n, rank0), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank0
    
    # Try to reduce rank by removing one term at a time and checking
    # if the remaining terms can still decompose the tensor.
    # This is equivalent to checking if the removed term is in the span
    # of the others, which for tensor rank is non-trivial.
    #
    # A simpler check: try setting one column to zero and see if we can
    # adjust other columns to compensate. This is a nonlinear problem.
    #
    # Alternative structural change: use a different factorization strategy.
    # For 4x5x5, use the fact that 4 = 2*2 to do a Kronecker product
    # based decomposition.
    #
    # Let me try: C = A*B, split A = [A1; A2] where A1, A2 are 2x5,
    # B = [B1 B2] where B1 is 5x2, B2 is 5x3.
    # C11 = A1*B1 (2x2), C12 = A1*B2 (2x3), C21 = A2*B1 (2x2), C22 = A2*B2 (2x3)
    #
    # For each 2x2 block, use schoolbook (4 terms each): 4*4 = 16
    # For each 2x3 block, use schoolbook (6 terms each): 6*2 = 12
    # Total: 16 + 12 = 28 for the 4x5x2 part
    #
    # Wait, that's for 4x5x2, not 4x5x5. Let me redo.
    #
    # 4x5x5: split B's 5 cols into 2+3.
    # C[:,0:2] = A*B[:,0:2] (4x5x2)
    # C[:,2:5] = A*B[:,2:5] (4x5x3)
    #
    # 4x5x2: split A's 4 rows into 2+2.
    # C[0:2,0:2] = A[0:2]*B[:,0:2] (2x5x2)
    # C[2:4,0:2] = A[2:4]*B[:,0:2] (2x5x2)
    # Each 2x5x2: split B's 5 rows into 2+3.
    # 2x5x2 = A(2x2)*B(2x2) + A(2x3)*B(3x2) = 4 + 12 = 16
    # So 4x5x2 = 16 + 16 = 32
    #
    # 4x5x3: split A's 4 rows into 2+2.
    # C[0:2,2:5] = A[0:2]*B[:,2:5] (2x5x3)
    # C[2:4,2:5] = A[2:4]*B[:,2:5] (2x5x3)
    # Each 2x5x3: split B's 5 rows into 2+3.
    # 2x5x3 = A(2x2)*B(2x3) + A(2x3)*B(3x3) = 12 + 18 = 30
    # So 4x5x3 = 30 + 30 = 60
    #
    # Total: 32 + 60 = 92!
    #
    # This matches the inspiration score of 1.1765 = 100/85... wait, 100/92 = 1.087.
    # The inspirations mention score 1.1765 which would be rank 85.
    #
    # Let me verify: 4x5x2 = 32, 4x5x3 = 60, total 92. 100/92 ≈ 1.087.
    #
    # But the inspirations claim 1.1765 = 100/85, so they found rank 85.
    # My calculation gives 92. The difference must be in how the blocks are split.
    #
    # Let me try a different split for 4x5x3:
    # Split A's 4 rows into 3+1 instead of 2+2.
    # 4x5x3 = 3x5x3 + 1x5x3
    # 3x5x3: split B's 5 rows into 3+2.
    #   3x5x3 = A(3x3)*B(3x3) + A(3x2)*B(2x3) = 27 + 18 = 45
    # 1x5x3: split B's 5 rows into 3+2.
    #   1x5x3 = A(1x3)*B(3x3) + A(1x2)*B(2x3) = 9 + 6 = 15
    # Total: 45 + 15 = 60. Same as before.
    #
    # Try splitting B's cols differently for 4x5x5:
    # 4x5x5 = 4x5x2 + 4x5x3 (as before) = 32 + 60 = 92
    #
    # Or: 4x5x5 = 4x3x5 + 4x2x5
    # 4x3x5: split A's 4 rows into 2+2.
    #   4x3x5 = 2x3x5 + 2x3x5
    #   2x3x5: split B's 5 cols into 2+3.
    #     2x3x5 = A(2x3)*B1(3x2) + A(2x3)*B2(3x3) = 12 + 18 = 30
    #   So 4x3x5 = 30 + 30 = 60
    # 4x2x5: split A's 4 rows into 2+2.
    #   4x2x5 = 2x2x5 + 2x2x5
    #   2x2x5: split B's 5 cols into 2+3.
    #     2x2x5 = A(2x2)*B1(2x2) + A(2x2)*B2(2x3) = 4 + 12 = 16
    #   So 4x2x5 = 16 + 16 = 32
    # Total: 60 + 32 = 92. Same.
    #
    # It seems 92 is the natural schoolbook block decomposition rank.
    # To get below 92, we need actual low-rank sub-decompositions.
    #
    # Let me implement the 92-rank decomposition.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_schoolbook_block(n_block, m_block, p_block, i_off, j_off, k_off):
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    gi = i_off + i
                    gj = j_off + j
                    gk = k_off + k
                    U_list.append((gi * m + gj, 1))
                    V_list.append((gj * p + gk, 1))
                    W_list.append((gk * n + gi, 1))
    
    # 4x5x2: split A into 2+2 rows
    # Block 1: A[0:2, :] * B[:, 0:2] -> 2x5x2
    #   Split B's 5 rows into 2+3
    #   Sub-block 1a: A[0:2, 0:2] * B[0:2, 0:2] -> 2x2x2
    add_schoolbook_block(2, 2, 2, 0, 0, 0)
    #   Sub-block 1b: A[0:2, 2:5] * B[2:5, 0:2] -> 2x3x2
    add_schoolbook_block(2, 3, 2, 0, 2, 0)
    # Block 2: A[2:4, :] * B[:, 0:2] -> 2x5x2
    #   Split B's 5 rows into 2+3
    #   Sub-block 2a: A[2:4, 0:2] * B[0:2, 0:2] -> 2x2x2
    add_schoolbook_block(2, 2, 2, 2, 0, 0)
    #   Sub-block 2b: A[2:4, 2:5] * B[2:5, 0:2] -> 2x3x2
    add_schoolbook_block(2, 3, 2, 2, 2, 0)
    
    # 4x5x3: split A into 2+2 rows
    # Block 3: A[0:2, :] * B[:, 2:5] -> 2x5x3
    #   Split B's 5 rows into 2+3
    #   Sub-block 3a: A[0:2, 0:2] * B[0:2, 2:5] -> 2x2x3
    add_schoolbook_block(2, 2, 3, 0, 0, 2)
    #   Sub-block 3b: A[0:2, 2:5] * B[2:5, 2:5] -> 2x3x3
    add_schoolbook_block(2, 3, 3, 0, 2, 2)
    # Block 4: A[2:4, :] * B[:, 2:5] -> 2x5x3
    #   Split B's 5 rows into 2+3
    #   Sub-block 4a: A[2:4, 0:2] * B[0:2, 2:5] -> 2x2x3
    add_schoolbook_block(2, 2, 3, 2, 0, 2)
    #   Sub-block 4b: A[2:4, 2:5] * B[2:5, 2:5] -> 2x3x3
    add_schoolbook_block(2, 3, 3, 2, 2, 2)
    
    rank = len(U_list)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r, (idx, val) in enumerate(U_list):
        U[idx, r] = val
    for r, (idx, val) in enumerate(V_list):
        V[idx, r] = val
    for r, (idx, val) in enumerate(W_list):
        W[idx, r] = val
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U_sb = np.zeros((n*m, rank0), dtype=int)
    V_sb = np.zeros((m*p, rank0), dtype=int)
    W_sb = np.zeros((p*n, rank0), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
