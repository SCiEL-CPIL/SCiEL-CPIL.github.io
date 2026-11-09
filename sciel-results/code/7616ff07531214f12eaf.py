"""Matrix multiplication 4x5 x 5x5 tensor decomposition with Strassen optimization."""
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill


def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 130  # Safety margin

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

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Verify tensor decomposition using vectorized operations."""
        rank = U_mat.shape[1]
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, 1.0, atol=tol)

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

    def build_full_decomposition(i_parts, j_parts, k_parts):
        """
        Build full decomposition from partition.
        Uses Strassen for 2x2x2 blocks, schoolbook for all others.
        """
        U_s, V_s, W_s = build_strassen_2x2x2()
        
        total_rank = 0
        block_list = []
        
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
                    
                    total_rank += U_b.shape[1]
                    block_list.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b) in block_list:
            rank_block = U_b.shape[1]
            nb_i = i_end - i_start
            nb_j = j_end - j_start
            nb_k = k_end - k_start
            for li in range(nb_i):
                for lj in range(nb_j):
                    gi = i_start + li
                    gj = j_start + lj
                    U[gi * m + gj, col:col+rank_block] = U_b[li * nb_j + lj, :]
            for lj in range(nb_j):
                for lk in range(nb_k):
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk, col:col+rank_block] = V_b[lj * nb_k + lk, :]
            for lk in range(nb_k):
                for li in range(nb_i):
                    gk = k_start + lk
                    gi = i_start + li
                    W[gk * n + gi, col:col+rank_block] = W_b[lk * nb_i + li, :]
            col += rank_block
        
        return U, V, W

    # Generate all reasonable partitions
    def gen_parts(dim):
        """Generate all ways to partition [0, dim] into contiguous segments of size 1 or 2."""
        parts = []
        def backtrack(pos, current):
            if pos == dim:
                parts.append(list(current))
                return
            if pos + 1 <= dim:
                current.append((pos, pos + 1))
                backtrack(pos + 1, current)
                current.pop()
            if pos + 2 <= dim:
                current.append((pos, pos + 2))
                backtrack(pos + 2, current)
                current.pop()
        backtrack(0, [])
        return parts

    i_parts_list = gen_parts(n)  # partitions of [0,4]
    j_parts_list = gen_parts(m)  # partitions of [0,5]
    k_parts_list = gen_parts(p)  # partitions of [0,5]

    # Pre-compute block ranks for quick evaluation
    def block_rank(n_b, m_b, p_b):
        if n_b == 2 and m_b == 2 and p_b == 2:
            return 7
        return n_b * m_b * p_b

    # Score partitions by estimated rank (without building)
    def estimate_rank(i_parts, j_parts, k_parts):
        total = 0
        for i_start, i_end in i_parts:
            for j_start, j_end in j_parts:
                for k_start, k_end in k_parts:
                    total += block_rank(i_end - i_start, j_end - j_start, k_end - k_start)
        return total

    # Sort all partition combinations by estimated rank
    all_combos = []
    for ip in i_parts_list:
        for jp in j_parts_list:
            for kp in k_parts_list:
                r = estimate_rank(ip, jp, kp)
                all_combos.append((r, ip, jp, kp))
    
    all_combos.sort(key=lambda x: x[0])

    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')

    # Try top partitions
    for est_rank, ip, jp, kp in all_combos:
        if est_rank >= best_rank:
            break
        if time.time() > deadline - 5:
            break
        
        U, V, W = build_full_decomposition(ip, jp, kp)
        if verify(U, V, W, n, m, p, tol=1e-9):
            r = U.shape[1]
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if best_U is None:
        # Fallback: full schoolbook
        best_U, best_V, best_W = schoolbook_block(n, m, p)
        if not verify(best_U, best_V, best_W, n, m, p, tol=1e-9):
            raise Exception("Verification failed")

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
