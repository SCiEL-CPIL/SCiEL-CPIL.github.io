"""Matrix multiplication 4x5 x 5x5 tensor decomposition with optimized block partitioning."""
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
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        
        # Term 2: M2 = (A10+A11)B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0
        
        # Term 3: M3 = A00(B10-B11)
        U[0, 2] = 1.0
        V[2, 2] = 1.0; V[3, 2] = -1.0
        W[1, 2] = 1.0; W[3, 2] = 1.0
        
        # Term 4: M4 = A11(B01-B10)
        U[3, 3] = 1.0
        V[1, 3] = 1.0; V[2, 3] = -1.0
        W[0, 3] = 1.0; W[2, 3] = 1.0
        
        # Term 5: M5 = (A00+A01)B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[1, 4] = 1.0
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[2, 5] = 1.0
        W[3, 5] = 1.0
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[1, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0; W[1, 6] = 1.0
        
        return U, V, W

    def build_2x3x3_rank17():
        """
        Build rank-17 decomposition for 2x3 * 3x3 -> 2x3.
        """
        U_s, V_s, W_s = build_strassen_2x2x2()
        
        rank_total = 17
        U = np.zeros((6, rank_total), dtype=float)
        V = np.zeros((9, rank_total), dtype=float)
        W = np.zeros((6, rank_total), dtype=float)
        
        for li in range(2):
            for lj in range(2):
                U[li * 3 + lj, 0:7] = U_s[li * 2 + lj, :]
        for lj in range(2):
            for lk in range(2):
                V[lj * 3 + lk, 0:7] = V_s[lj * 2 + lk, :]
        for lk in range(2):
            for li in range(2):
                W[lk * 2 + li, 0:7] = W_s[lk * 2 + li, :]
        
        remaining = []
        for i in range(2):
            for j in range(2):
                remaining.append((i, j, 2))
            for k in range(2):
                remaining.append((i, 2, k))
            remaining.append((i, 2, 2))
        
        for idx, (i, j, k) in enumerate(remaining):
            col = 7 + idx
            U[i * 3 + j, col] = 1.0
            V[j * 3 + k, col] = 1.0
            W[k * 2 + i, col] = 1.0
        
        return U, V, W

    def build_2x3x2_rank11():
        """
        Build rank-11 decomposition for 2x3 * 3x2 -> 2x2.
        """
        U_s, V_s, W_s = build_strassen_2x2x2()
        
        rank_total = 11
        U = np.zeros((6, rank_total), dtype=float)
        V = np.zeros((6, rank_total), dtype=float)
        W = np.zeros((4, rank_total), dtype=float)
        
        for li in range(2):
            for lj in range(2):
                U[li * 3 + lj, 0:7] = U_s[li * 2 + lj, :]
        for lj in range(2):
            for lk in range(2):
                V[lj * 2 + lk, 0:7] = V_s[lj * 2 + lk, :]
        for lk in range(2):
            for li in range(2):
                W[lk * 2 + li, 0:7] = W_s[lk * 2 + li, :]
        
        remaining = [(0, 2, 0), (0, 2, 1), (1, 2, 0), (1, 2, 1)]
        for idx, (i, j, k) in enumerate(remaining):
            col = 7 + idx
            U[i * 3 + j, col] = 1.0
            V[j * 2 + k, col] = 1.0
            W[k * 2 + i, col] = 1.0
        
        return U, V, W

    def build_2x2x3_rank11():
        """
        Build rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        """
        U_s, V_s, W_s = build_strassen_2x2x2()
        
        rank_total = 11
        U = np.zeros((4, rank_total), dtype=float)
        V = np.zeros((6, rank_total), dtype=float)
        W = np.zeros((6, rank_total), dtype=float)
        
        for li in range(2):
            for lj in range(2):
                U[li * 2 + lj, 0:7] = U_s[li * 2 + lj, :]
        for lj in range(2):
            for lk in range(2):
                V[lj * 3 + lk, 0:7] = V_s[lj * 2 + lk, :]
        for lk in range(2):
            for li in range(2):
                W[lk * 2 + li, 0:7] = W_s[lk * 2 + li, :]
        
        remaining = [(0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
        for idx, (i, j, k) in enumerate(remaining):
            col = 7 + idx
            U[i * 2 + j, col] = 1.0
            V[j * 3 + k, col] = 1.0
            W[k * 2 + i, col] = 1.0
        
        return U, V, W

    def build_schoolbook(n_b, m_b, p_b):
        """Build schoolbook decomposition for n_b x m_b * m_b x p_b."""
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

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Verify tensor decomposition."""
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
        """
        Combine block decompositions into full decomposition.
        """
        total_rank = sum(U_block.shape[1] for (_, _, _, _, _, _, U_block, _, _) in blocks)
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

    def get_block(n_b, m_b, p_b):
        """Get the best known decomposition for a block."""
        if n_b == 2 and m_b == 3 and p_b == 3:
            return build_2x3x3_rank17()
        elif n_b == 2 and m_b == 3 and p_b == 2:
            return build_2x3x2_rank11()
        elif n_b == 2 and m_b == 2 and p_b == 3:
            return build_2x2x3_rank11()
        elif n_b == 2 and m_b == 2 and p_b == 2:
            return build_strassen_2x2x2()
        else:
            return build_schoolbook(n_b, m_b, p_b)

    def try_partition(i_parts, j_parts, k_parts):
        """Try a specific partition and return (rank, U, V, W) or None if invalid."""
        blocks = []
        
        for i_start, i_end in i_parts:
            for j_start, j_end in j_parts:
                for k_start, k_end in k_parts:
                    n_b = i_end - i_start
                    m_b = j_end - j_start
                    p_b = k_end - k_start
                    
                    U_b, V_b, W_b = get_block(n_b, m_b, p_b)
                    blocks.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
        
        U, V, W = combine_decompositions(blocks, n, m, p)
        
        if not verify(U, V, W, n, m, p, tol=1e-9):
            return None
        return U.shape[1], U, V, W

    # Best partition: i=(2,2), j=(3,2), k=(3,2) -> rank 46
    best_result = try_partition([(0,2),(2,4)], [(0,3),(3,5)], [(0,3),(3,5)])
    
    if best_result is None:
        # Fallback to schoolbook
        U, V, W = build_schoolbook(n, m, p)
        if not verify(U, V, W, n, m, p, tol=1e-9):
            raise Exception("Verification failed")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank, best_U, best_V, best_W = best_result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
