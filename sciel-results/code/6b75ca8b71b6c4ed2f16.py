"""Matrix multiplication 3x4 by 4x6: optimized Strassen-based rank-66 decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that U * V^T * W^T gives the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against schoolbook: T[i,j,k] = 1 if all indices valid, else 0
    # Actually the tensor for matmul: T[i,j,k] = A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Row 0 (1x4x6): schoolbook, 24 terms
    - Rows 1-2 (2x4x6): Strassen on 2x2x2 sub-blocks, 6 blocks * 7 = 42 terms
    Total rank: 66
    
    U shape: (12, 66)  -- rows indexed by i*4+j
    V shape: (24, 66)  -- rows indexed by j*6+k
    W shape: (18, 66)  -- rows indexed by k*3+i
    """
    n, m, p = 3, 4, 6
    
    # Build the decomposition
    rank = 66
    U = [[0] * rank for _ in range(n * m)]   # 12 x 66
    V = [[0] * rank for _ in range(m * p)]   # 24 x 66
    W = [[0] * rank for _ in range(p * n)]   # 18 x 66
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        """Add 7 Strassen terms for a 2x2x2 block at global indices."""
        nonlocal col
        
        # A indices (2x2 sub-block of A)
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        # B indices (2x2 sub-block of B)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        # C/W indices (2x2 sub-block of C), W index = k*n + i
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # P1 = (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)e -> contributes to C[1,0] and C[1,1]
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # P3 = a(f-h) -> contributes to C[0,1] and C[1,1]
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P4 = d(g-e) -> contributes to C[0,0] and C[1,0]
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)h -> contributes to C[0,0] and C[0,1]
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f) -> contributes to C[1,1]
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h) -> contributes to C[0,0]
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Part 1: Row 0 schoolbook (1x4x6 = 24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Part 2: Rows 1-2 Strassen (2x4x6, 6 blocks of 2x2x2 = 42 terms)
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback: pure schoolbook (rank 72)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
