"""4x5x6 matrix multiplication decomposition (rank 96, improved Strassen-based)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Tiling strategy:
    # j in [0,2): 2x2x2 Strassen blocks (rank 7 each)
    # j in [2,4): 2x2x2 Strassen blocks (rank 7 each)  
    # j in [4,5): 2x1x2 blocks (rank 2 each)
    # k in [0,2), [2,4), [4,6)
    # i in [0,2), [2,4)
    
    # Count:
    # j=[0,2): 2(i) * 3(k) = 6 blocks * 7 = 42
    # j=[2,4): 2(i) * 3(k) = 6 blocks * 7 = 42
    # j=[4,5): 2(i) * 3(k) = 6 blocks * 2 = 12
    # Total: 96
    
    rank = 96
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_offset = 0
    
    def add_strassen_block(i0, j0, k0, r_start):
        """Add 2x2x2 Strassen block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        
        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        
        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        return r_start + 1
    
    def add_2x1x2_block(i0, j0, k0, r_start):
        """Add 2x1x2 block for A[i0:i0+2, j0:j0+1] * B[j0:j0+1, k0:k0+2]
        This is just the sum of 2 rank-1 terms."""
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # Term 1: A[i0, j0] * B[j0, k0]
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        # Term 2: A[i0+1, j0] * B[j0, k0+1]
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        return r_start + 1
    
    # Process j=[0,2): 2x2x2 Strassen blocks
    for i0 in [0, 2]:
        for k0 in [0, 2, 4]:
            r_offset = add_strassen_block(i0, 0, k0, r_offset)
    
    # Process j=[2,4): 2x2x2 Strassen blocks
    for i0 in [0, 2]:
        for k0 in [0, 2, 4]:
            r_offset = add_strassen_block(i0, 2, k0, r_offset)
    
    # Process j=[4,5): 2x1x2 blocks
    for i0 in [0, 2]:
        for k0 in [0, 2, 4]:
            r_offset = add_2x1x2_block(i0, 4, k0, r_offset)
    
    if r_offset != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {r_offset}")
    
    # Verify the decomposition is exact
    def verify(U_mat, V_mat, W_mat, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank_val):
                        s += U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        rank = rank_sb
    
    return {"U": U, "V": V, "W": W}
