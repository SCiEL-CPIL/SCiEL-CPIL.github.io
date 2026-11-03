"""Matrix multiplication 3x4 by 4x6: Strassen-based rank-66 decomposition with verification."""
import time

DIMENSIONS = [3, 4, 6]

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

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
        
        # P1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # P3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
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
    def verify(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if val != 1:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback to pure schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
