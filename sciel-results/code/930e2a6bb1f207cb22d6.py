import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        # Vectorized verification for speed
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)
        # W shape: (p*n, rank)
        
        # We need to check T[i,j,k] = 1 for all i,j,k
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Iterate over i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    # The sum is element-wise product then sum
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def strassen_2x2():
        """
        Return U, V, W for 2x2x2 Strassen with shapes (4,7), (4,7), (4,7).
        """
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
        W[2, 1] = 1; W[3, 1] = -1
        
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
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def make_blocked_strassen():
        """
        Block structure for 3x4x6:
        A: 3x4, B: 4x6, C: 3x6
        
        Splits:
        I (rows of A, cols of C): [0:2], [2:3]
        J (cols of A, rows of B): [0:2], [2:4]
        K (cols of B, rows of C): [0:2], [2:4], [4:6]
        
        Total rank = 6 * 7 (Strassen) + 6 * 4 (Schoolbook) = 42 + 24 = 66
        """
        sU, sV, sW = strassen_2x2()
        
        total_rank = 6 * 7 + 6 * 4  # 66
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col_offset = 0
        
        a_row_blocks = [(0, 2), (2, 3)]
        a_col_blocks = [(0, 2), (2, 4)]
        b_col_blocks = [(0, 2), (2, 4), (4, 6)]
        
        for a_row_start, a_row_end in a_row_blocks:
            a_row_size = a_row_end - a_row_start
            
            for a_col_start, a_col_end in a_col_blocks:
                b_row_start = a_col_start
                b_row_end = a_col_end
                
                for b_col_start, b_col_end in b_col_blocks:
                    if a_row_size == 2:
                        # Use Strassen for (2,2,2) block
                        for r_local in range(7):
                            # U
                            for i_local in range(2):
                                for j_local in range(2):
                                    val = sU[i_local * 2 + j_local, r_local]
                                    if val != 0:
                                        U[(a_row_start + i_local) * m + (a_col_start + j_local), col_offset] = val
                            
                            # V
                            for j_local in range(2):
                                for k_local in range(2):
                                    val = sV[j_local * 2 + k_local, r_local]
                                    if val != 0:
                                        V[(b_row_start + j_local) * p + (b_col_start + k_local), col_offset] = val
                            
                            # W
                            for k_local in range(2):
                                for i_local in range(2):
                                    val = sW[k_local * 2 + i_local, r_local]
                                    if val != 0:
                                        W[(b_col_start + k_local) * n + (a_row_start + i_local), col_offset] = val
                            
                            col_offset += 1
                    else:
                        # a_row_size == 1, use schoolbook for (1,2,2) block
                        # Rank = 4
                        for j_local in range(2):
                            for k_local in range(2):
                                U[a_row_start * m + (a_col_start + j_local), col_offset] = 1.0
                                V[(b_row_start + j_local) * p + (b_col_start + k_local), col_offset] = 1.0
                                W[(b_col_start + k_local) * n + a_row_start, col_offset] = 1.0
                                col_offset += 1
        
        return U, V, W, total_rank

    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - start_time > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try blocked Strassen
    try:
        bU, bV, bW, b_rank = make_blocked_strassen()
        if verify_decomposition(bU, bV, bW, b_rank, n, m, p):
            if b_rank < best_rank:
                best_U, best_V, best_W, best_rank = bU, bV, bW, b_rank
        else:
            pass
    except Exception:
        pass
    
    if time.time() - start_time > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
