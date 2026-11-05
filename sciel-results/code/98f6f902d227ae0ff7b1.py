import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS

    # Target rank: 46
    rank_init = 46
    U = np.zeros((n * m, rank_init), dtype=np.float64)
    V = np.zeros((m * p, rank_init), dtype=np.float64)
    W = np.zeros((p * n, rank_init), dtype=np.float64)

    col = 0

    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col
        U[u_idx, col] = coeff
        V[v_idx, col] = coeff
        W[w_idx, col] = coeff
        col += 1

    def emit_strassen_block(i_offset, j_offset, k_offset):
        """
        Emits 7 terms for a 2x2x2 Strassen block.
        i_offset: offset for i index (0 or 1)
        j_offset: offset for j index (0, 1, 2, or 3)
        k_offset: offset for k index (0, 2, 4)
        """
        nonlocal col
        
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Global indices:
        # I = i + i_offset
        # J = j + j_offset
        # K = k + k_offset
        
        # M1: (A00 + A11)(B00 + B11)
        U[(0 + i_offset)*m + (0 + j_offset), col] = 1.0
        U[(1 + i_offset)*m + (1 + j_offset), col] = 1.0
        V[(0 + j_offset)*p + (0 + k_offset), col] = 1.0
        V[(1 + j_offset)*p + (1 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (0 + i_offset), col] = 1.0
        W[(1 + k_offset)*n + (1 + i_offset), col] = 1.0
        col += 1
        
        # M2: (A10 + A11)B00
        U[(1 + i_offset)*m + (0 + j_offset), col] = 1.0
        U[(1 + i_offset)*m + (1 + j_offset), col] = 1.0
        V[(0 + j_offset)*p + (0 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (1 + i_offset), col] = 1.0
        W[(1 + k_offset)*n + (1 + i_offset), col] = -1.0
        col += 1
        
        # M3: A00(B10 - B00)
        U[(0 + i_offset)*m + (0 + j_offset), col] = 1.0
        V[(0 + j_offset)*p + (1 + k_offset), col] = 1.0
        V[(1 + j_offset)*p + (1 + k_offset), col] = -1.0
        W[(1 + k_offset)*n + (0 + i_offset), col] = 1.0
        W[(1 + k_offset)*n + (1 + i_offset), col] = 1.0
        col += 1
        
        # M4: A11(B01 - B00)
        U[(1 + i_offset)*m + (1 + j_offset), col] = 1.0
        V[(1 + j_offset)*p + (0 + k_offset), col] = 1.0
        V[(0 + j_offset)*p + (0 + k_offset), col] = -1.0
        W[(0 + k_offset)*n + (0 + i_offset), col] = 1.0
        W[(0 + k_offset)*n + (1 + i_offset), col] = 1.0
        col += 1
        
        # M5: (A00 + A01)(B11 - B10)
        U[(0 + i_offset)*m + (0 + j_offset), col] = 1.0
        U[(0 + i_offset)*m + (1 + j_offset), col] = 1.0
        V[(1 + j_offset)*p + (1 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (0 + i_offset), col] = -1.0
        W[(1 + k_offset)*n + (0 + i_offset), col] = 1.0
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[(1 + i_offset)*m + (0 + j_offset), col] = 1.0
        U[(0 + i_offset)*m + (0 + j_offset), col] = -1.0
        V[(0 + j_offset)*p + (0 + k_offset), col] = 1.0
        V[(0 + j_offset)*p + (1 + k_offset), col] = 1.0
        W[(1 + k_offset)*n + (1 + i_offset), col] = 1.0
        col += 1
        
        # M7: (A01 - A11)(B10 + B11)
        U[(0 + i_offset)*m + (1 + j_offset), col] = 1.0
        U[(1 + i_offset)*m + (1 + j_offset), col] = -1.0
        V[(1 + j_offset)*p + (0 + k_offset), col] = 1.0
        V[(1 + j_offset)*p + (1 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (0 + i_offset), col] = 1.0
        col += 1

    def emit_direct_block(i_offset, j_offset, k_offset):
        """
        Emits 2 terms for a 2x2x1 direct multiplication.
        i_offset: offset for i index (0 or 1)
        j_offset: offset for j index (0, 1, 2, or 3)
        k_offset: offset for k index (6)
        """
        nonlocal col
        
        # C_i0 = A_i0 B_00 + A_i1 B_10
        # Term 1: A_i0 B_00 -> C_i0
        U[(0 + i_offset)*m + (0 + j_offset), col] = 1.0
        V[(0 + j_offset)*p + (0 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (0 + i_offset), col] = 1.0
        col += 1
        
        U[(1 + i_offset)*m + (0 + j_offset), col] = 1.0
        V[(0 + j_offset)*p + (0 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (1 + i_offset), col] = 1.0
        col += 1
        
        # Term 2: A_i1 B_10 -> C_i0
        U[(0 + i_offset)*m + (1 + j_offset), col] = 1.0
        V[(1 + j_offset)*p + (0 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (0 + i_offset), col] = 1.0
        col += 1
        
        U[(1 + i_offset)*m + (1 + j_offset), col] = 1.0
        V[(1 + j_offset)*p + (0 + k_offset), col] = 1.0
        W[(0 + k_offset)*n + (1 + i_offset), col] = 1.0
        col += 1

    # Left part: A[:, 0:2] * B[0:2, :]
    # i_offset = 0 (i is 0, 1)
    # j_offset = 0 (j is 0, 1)
    # k offsets: 0, 2, 4 for Strassen (2x2x2), 6 for Direct (2x2x1)
    
    # Strassen blocks for k=0,1; k=2,3; k=4,5
    emit_strassen_block(0, 0, 0)
    emit_strassen_block(0, 0, 2)
    emit_strassen_block(0, 0, 4)
    
    # Direct block for k=6
    emit_direct_block(0, 0, 6)
    
    # Right part: A[:, 2:4] * B[2:4, :]
    # i_offset = 0
    # j_offset = 2 (j is 2, 3)
    # k offsets: 0, 2, 4 for Strassen, 6 for Direct
    
    emit_strassen_block(0, 2, 0)
    emit_strassen_block(0, 2, 2)
    emit_strassen_block(0, 2, 4)
    
    emit_direct_block(0, 2, 6)
    
    # Verify
    if col != rank_init:
        print(f"Error: Expected rank {rank_init}, got {col}")
        return None

    U_3d = U.reshape(n, m, rank_init)
    V_3d = V.reshape(m, p, rank_init)
    W_3d = W.reshape(p, n, rank_init)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # The tensor T should be 1.0 everywhere
    if not np.allclose(T, 1.0, atol=1e-9):
        print("Verification failed")
        return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
