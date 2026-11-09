import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 130

    def build_strassen_2x2x2():
        """
        Build Strassen's 7-term decomposition for 2x2 * 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        U index: i*2 + j
        V index: j*2 + k
        W index: k*2 + i
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # A00: U[0]=1, A11: U[3]=1
        # B00: V[0]=1, B11: V[3]=1
        # C00: W[0]=1, C11: W[3]=1
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        
        # Term 2: M2 = (A10+A11)B00
        # A10: U[2]=1, A11: U[3]=1
        # B00: V[0]=1
        # C10: W[1]=1 (k=0, i=1 -> 0*2+1=1)
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0
        
        # Term 3: M3 = A00(B10-B11)
        # A00: U[0]=1
        # B10: V[2]=1, B11: V[3]=-1
        # C01: W[2]=1 (k=1, i=0 -> 1*2+0=2)
        # C11: W[3]=1
        U[0, 2] = 1.0
        V[2, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0
        
        # Term 4: M4 = A11(B01-B10)
        # A11: U[3]=1
        # B01: V[1]=1, B10: V[2]=-1
        # C00: W[0]=1
        # C10: W[1]=1
        U[3, 3] = 1.0
        V[1, 3] = 1.0; V[2, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0
        
        # Term 5: M5 = (A00+A01)B11
        # A00: U[0]=1, A01: U[1]=1
        # B11: V[3]=1
        # C00: W[0]=-1
        # C01: W[2]=1
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        # A10: U[2]=1, A00: U[0]=-1
        # B00: V[0]=1, B10: V[2]=1
        # C11: W[3]=1
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[2, 5] = 1.0
        W[3, 5] = 1.0
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        # A01: U[1]=1, A11: U[3]=-1
        # B01: V[1]=1, B11: V[3]=1
        # C00: W[0]=1
        # C01: W[2]=1
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[1, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0; W[2, 6] = 1.0
        
        return U, V, W

    def build_schoolbook(n, m, p):
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb), dtype=float)
        V = np.zeros((m * p, rank_sb), dtype=float)
        W = np.zeros((p * n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def combine_decompositions(blocks, n, m, p):
        total_rank = sum(len(block) for block in blocks)
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (i_start, j_start, k_start, i_end, j_end, k_end, U_block, V_block, W_block) in blocks:
            rank_block = U_block.shape[1]
            n_b = i_end - i_start
            m_b = j_end - j_start
            p_b = k_end - k_start
            
            for li in range(n_b):
                for lj in range(m_b):
                    gi = i_start + li
                    gj = j_start + lj
                    U[gi * m + gj, col:col+rank_block] = U_block[li * m_b + lj, :]
            for lj in range(m_b):
                for lk in range(p_b):
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk, col:col+rank_block] = V_block[lj * p_b + lk, :]
            for lk in range(p_b):
                for li in range(n_b):
                    gk = k_start + lk
                    gi = i_start + li
                    W[gk * n + gi, col:col+rank_block] = W_block[lk * n_b + li, :]
            col += rank_block
        
        return U, V, W

    def schoolbook_block(n_b, m_b, p_b):
        rank_b = n_b * m_b * p_b
        U_b = np.zeros((n_b * m_b, rank_b), dtype=float)
        V_b = np.zeros((m_b * p_b, rank_b), dtype=float)
        W_b = np.zeros((p_b * n_b, rank_b), dtype=float)
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j, col] = 1.0
                    V_b[j * p_b + k, col] = 1.0
                    W_b[k * n_b + i, col] = 1.0
                    col += 1
        return U_b, V_b, W_b

    # Pre-compute Strassen block
    U_s, V_s, W_s = build_strassen_2x2x2()
    
    # Partition
    i_parts = [(0, 2), (2, 4)]
    j_parts = [(0, 2), (2, 4), (4, 5)]
    k_parts = [(0, 2), (2, 4), (4, 5)]
    
    blocks = []
    
    for i_start, i_end in i_parts:
        for j_start, j_end in j_parts:
            for k_start, k_end in k_parts:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    U_b, V_b, W_b = U_s.copy(), V_s.copy(), W_s.copy()
                else:
                    U_b, V_b, W_b = schoolbook_block(n_b, m_b, p_b)
                
                blocks.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
    
    U, V, W = combine_decompositions(blocks, n, m, p)
    
    if not verify(U, V, W, n, m, p, tol=1e-9):
        U, V, W = build_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
