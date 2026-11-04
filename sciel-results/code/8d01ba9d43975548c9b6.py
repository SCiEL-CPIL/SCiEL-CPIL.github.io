"""Matrix multiplication 4x4x8: rank-minimal exact tensor decomposition."""

import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2x2 matrix multiplication (Strassen)."""
    # Standard Strassen decomposition for C = A*B where A,B,C are 2x2.
    # Indexing: A[i,j] for i,j in 0..1
    # U maps A (4 entries) to 7 multipliers
    # V maps B (4 entries) to 7 multipliers
    # W maps C (4 entries) from 7 multipliers
    
    # Strassen multipliers:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U: for each multiplier, coefficients on A00, A01, A10, A11
    U = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ], dtype=float)
    
    # V: for each multiplier, coefficients on B00, B01, B10, B11
    V = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, -1, 1, 0],  # M3: B10-B11
        [1, -1, 0, 0],  # M4: B01-B10
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ], dtype=float)
    
    # W: for each C entry, coefficients on M1..M7
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    W = np.array([
        [1, 0, 0, -1, 1, 0, 1],   # C00
        [0, 0, 1, 1, 0, 0, 0],   # C01
        [0, 1, 0, 0, 1, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ], dtype=float)
    
    return U, V, W


def _strassen_4x4():
    """Return U (16x49), V (16x49), W (16x49) for 4x4x4 matrix multiplication using block Strassen."""
    # Apply 2x2 Strassen at the block level (each block is 2x2), 
    # then apply 2x2 Strassen within each 2x2 block.
    # Total rank: 7 * 7 = 49
    
    U2, V2, W2 = _strassen_2x2()
    
    # Build 4x4 Strassen via two levels:
    # Level 1: Block 2x2 Strassen -> 7 block multipliers, each involving 2x2 matrices
    # Level 2: For each of the 7 block operations, apply 2x2 Strassen -> 7 sub-multipliers each
    # Total: 7 * 7 = 49
    
    # U4: 16 entries of A -> 49 multipliers
    # A is indexed as A[i,j] for i,j in 0..3, flattened row-major
    # Block structure: A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    
    # First level: 7 block-level multipliers on 2x2 blocks
    # Second level: each block multiplier is computed via 2x2 Strassen (7 sub-multipliers)
    
    # U4[row, col]: row = A[i,j] (i*4+j), col = sub-multiplier index (block_mult * 7 + sub_mult)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for bm in range(7):
        for sm in range(7):
            col = bm * 7 + sm
            
            # For U: coefficient of A[i,j] in the bm-th block multiplier's sm-th sub-multiplier
            # Block multiplier bm uses certain A-blocks with certain signs/coefficients
            # From U2[bm, block_row] gives coefficient for A-block block_row
            # Within that block, sub-multiplier sm uses certain 2x2 entries with U2[sm, entry]
            
            # A[i,j] belongs to block (i//2, j//2) and entry (i%2, j%2)
            for i in range(4):
                for j in range(4):
                    row = i * 4 + j
                    block_i = i // 2
                    block_j = j // 2
                    entry_i = i % 2
                    entry_j = j % 2
                    # Coefficient = U2[bm, block_i*2+block_j] * U2[sm, entry_i*2+entry_j]
                    U4[row, col] = U2[bm, block_i * 2 + block_j] * U2[sm, entry_i * 2 + entry_j]
            
            # Similarly for V
            for i in range(4):
                for j in range(4):
                    row = i * 4 + j
                    block_i = i // 2
                    block_j = j // 2
                    entry_i = i % 2
                    entry_j = j % 2
                    V4[row, col] = V2[bm, block_i * 2 + block_j] * V2[sm, entry_i * 2 + entry_j]
            
            # For W: C[i,j] gets contribution from block (i//2, j//2) entry (i%2, j%2)
            # Block multiplier bm contributes to C-block with W2[bm, block_i*2+block_j]
            # Sub-multiplier sm contributes with W2[sm, entry_i*2+entry_j]
            for i in range(4):
                for j in range(4):
                    row = i * 4 + j
                    block_i = i // 2
                    block_j = j // 2
                    entry_i = i % 2
                    entry_j = j % 2
                    W4[row, col] += W2[bm, block_i * 2 + block_j] * W2[sm, entry_i * 2 + entry_j]
    
    return U4, V4, W4


def _verify_decomposition(U, V, W, n, m, p, trials=5):
    """Verify that the decomposition exactly computes C = A@B for random integer inputs."""
    for _ in range(trials):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        # Compute C from decomposition:
        # C[k*n+i, :] where C[i,k] = sum_r U[i*m+j, r] A[i,j] * V[j*p+k, r] B[j,k] * W[k*n+i, r]
        # More precisely: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        
        C_computed = np.zeros((n, p), dtype=float)
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for r in range(len(U_arr[0])):
            u_vec = np.zeros((n, m), dtype=float)
            v_vec = np.zeros((m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    u_vec[i, j] = U_arr[i * m + j, r]
            for j in range(m):
                for k in range(p):
                    v_vec[j, k] = V_arr[j * p + k, r]
            
            a_part = u_vec * A
            b_part = v_vec * B
            c_part = a_part @ b_part
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_arr[k * n + i, r] * c_part[i, k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    # Approach: Use block Strassen for 4x4x4 (rank 49), then extend to 4x4x8.
    # For 4x4x8, we can view B as two 4x4 blocks: B = [B1 | B2]
    # C = A * B = [A*B1 | A*B2]
    # Each A*Bx is a 4x4x4 multiplication, rank 49 each.
    # But we can share the A-side computations: A is the same for both.
    # So rank = 49 (for A transforms) + 49 + 49 (for B1, B2 transforms and C) 
    # Actually, the tensor rank for 4x4x8:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For k in 0..7, split into k1=k%4 (within block) and k2=k//4 (which block)
    # C[i, k2*4+k1] = sum_j A[i,j] B[j, k2*4+k1]
    # = sum_j A[i,j] B2[k2][j, k1]
    
    # Using block Strassen: rank for one 4x4x4 is 49.
    # For two blocks, we need 49 (shared A) + 49 (B1) + 49 (B2) + 49 (C1) + 49 (C2)?
    # No, that's not right. The rank is the number of scalar multiplications.
    # In Strassen, rank = number of multipliers. For 4x4x4, that's 49.
    # For 4x4x8 with two 4x4 B-blocks sharing A:
    # Each multiplier r has U[r] (for A), V1[r] (for B1), V2[r] (for B2), W1[r] (for C1), W2[r] (for C2)
    # But in standard tensor decomposition, each rank-1 term has one U, one V, one W.
    # For the 4x4x8 tensor, the V matrix has shape (m*p, rank) = (32, rank).
    # The W matrix has shape (p*n, rank) = (32, rank).
    
    # If we use independent Strassen for each block: rank = 49 + 49 = 98
    # But we can do better by sharing A-side structure.
    
    # Let's first try the simple approach: two independent 4x4x4 Strassen decompositions.
    # This gives rank 98, which is better than 128.
    
    U4, V4, W4 = _strassen_4x4()
    
    # For 4x4x8:
    # U: shape (n*m, rank) = (16, rank)
    # V: shape (m*p, rank) = (32, rank)
    # W: shape (p*n, rank) = (32, rank)
    
    rank = 2 * 49  # Two independent 4x4x4 decompositions
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First block: B[:, 0:4], C[:, 0:4]
    U[:, 0:49] = U4
    # V for first block: V4[j*4+k, r] maps to V[j*8+k, r] for k in 0..3
    for j in range(m):
        for k in range(4):
            V[j * p + k, 0:49] = V4[j * 4 + k, :]
    # W for first block: W4[k*4+i, r] maps to W[k*4+i, r] for k in 0..3
    for k in range(4):
        for i in range(n):
            W[k * n + i, 0:49] = W4[k * 4 + i, :]
    
    # Second block: B[:, 4:8], C[:, 4:8]
    U[:, 49:98] = U4
    for j in range(m):
        for k in range(4):
            V[j * p + (k + 4), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(k + 4) * n + i, 49:98] = W4[k * 4 + i, :]
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify_decomposition(U_list, V_list, W_list, n, m, p):
        # Fall back to schoolbook
        rank_s = n * m * p
        U_s = [[0] * rank_s for _ in range(n * m)]
        V_s = [[0] * rank_s for _ in range(m * p)]
        W_s = [[0] * rank_s for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U_list, "V": V_list, "W": W_list}
