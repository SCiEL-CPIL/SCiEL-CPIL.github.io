"""Tensor rank minimization for 2x5x6 matrix multiplication.

Strategy: hybrid block decomposition with iterative rank reduction.
1. Build an initial decomposition using Strassen (rank 7) for 2x2x2 blocks
   and standard rank for remaining blocks.
2. Iteratively try to remove columns by solving a linear system for the
   correction terms.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # ============================================================
    # Phase 1: Build initial decomposition
    # ============================================================

    # Partition indices
    # J (size 5) into: J1={0,1}, J2={2,3}, J3={4}
    # K (size 6) into: K1={0,1}, K2={2,3}, K3={4,5}

    # Block structure:
    # 6 blocks of 2x2x2: (J1,J2) x (K1,K2,K3) -> each uses Strassen rank 7
    # 3 blocks of 1x2x2: J3 x (K1,K2,K3) -> each uses standard rank 4
    # Total initial rank = 6*7 + 3*4 = 42 + 12 = 54

    initial_rank = 54
    U = np.zeros((n * m, initial_rank), dtype=float)
    V = np.zeros((m * p, initial_rank), dtype=float)
    W = np.zeros((p * n, initial_rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= initial_rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    def strassen_block(ja, jb, ka, kb):
        """Add Strassen 7-term decomposition for 2x2x2 block."""
        u0a = 0 * m + ja
        u0b = 0 * m + jb
        u1a = 1 * m + ja
        u1b = 1 * m + jb

        v_a_ka = ja * p + ka
        v_a_kb = ja * p + kb
        v_b_ka = jb * p + ka
        v_b_kb = jb * p + kb

        w_ka_0 = ka * n + 0
        w_ka_1 = ka * n + 1
        w_kb_0 = kb * n + 0
        w_kb_1 = kb * n + 1

        # M1: (A00+A11)(B00+B11)
        add_term([(u0a, 1), (u1b, 1)],
                 [(v_a_ka, 1), (v_b_kb, 1)],
                 [(w_ka_0, 1), (w_kb_1, 1)])

        # M2: (A10+A11)(B00)
        add_term([(u1a, 1), (u1b, 1)],
                 [(v_a_ka, 1)],
                 [(w_ka_1, 1), (w_kb_1, -1)])

        # M3: (A00)(B01-B11)
        add_term([(u0a, 1)],
                 [(v_a_kb, 1), (v_b_kb, -1)],
                 [(w_kb_0, 1), (w_kb_1, 1)])

        # M4: (A11)(B10-B00)
        add_term([(u1b, 1)],
                 [(v_b_ka, 1), (v_a_ka, -1)],
                 [(w_ka_0, 1), (w_ka_1, 1)])

        # M5: (A00+A01)(B11)
        add_term([(u0a, 1), (u0b, 1)],
                 [(v_b_kb, 1)],
                 [(w_ka_0, -1), (w_kb_0, 1)])

        # M6: (A10-A00)(B00+B01)
        add_term([(u1a, 1), (u0a, -1)],
                 [(v_a_ka, 1), (v_a_kb, 1)],
                 [(w_kb_1, 1)])

        # M7: (A01-A11)(B10+B11)
        add_term([(u0b, 1), (u1b, -1)],
                 [(v_b_ka, 1), (v_b_kb, 1)],
                 [(w_ka_0, 1)])

    # Add 6 Strassen blocks
    strassen_block(0, 1, 0, 1)
    strassen_block(0, 1, 2, 3)
    strassen_block(0, 1, 4, 5)
    strassen_block(2, 3, 0, 1)
    strassen_block(2, 3, 2, 3)
    strassen_block(2, 3, 4, 5)

    # Add 3 standard blocks for J3={4}
    for k_block in [(0, 1), (2, 3), (4, 5)]:
        for k in k_block:
            for i in range(n):
                u_idx = i * m + 4
                v_idx = 4 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    U_init = U[:, :col].copy()
    V_init = V[:, :col].copy()
    W_init = W[:, :col].copy()
    current_rank = col

    # ============================================================
    # Phase 2: Iterative rank reduction
    # ============================================================

    def try_reduce_column(U_cur, V_cur, W_cur, rank_cur, col_to_remove):
        """
        Try to remove column col_to_remove by finding corrections to other columns.
        
        We need: sum_l U[:,l] * V[:,l] * W[:,l] (with col l != col_to_remove)
                 = original sum - contribution from col_to_remove
        
        This is a linear system in the unknown corrections.
        """
        n_rm, m_rm, p_rm = n, m, p
        target_vec = np.zeros(n_rm * m_rm * p_rm)
        
        # Compute current full tensor
        for i in range(n_rm):
            for j in range(m_rm):
                for k in range(p_rm):
                    val = 0.0
                    for l in range(rank_cur):
                        val += U_cur[i * m_rm + j, l] * V_cur[j * p_rm + k, l] * W_cur[k * n_rm + i, l]
                    # The tensor entry should be 1 if i,j,k form a valid (i,j,k) index
                    # Actually the tensor is T[i,j,k] = 1 for all valid (i,j,k)
                    target_vec[i * m_rm * p_rm + j * p_rm + k] = 1.0
        
        # Remove the contribution of col_to_remove
        for i in range(n_rm):
            for j in range(m_rm):
                for k in range(p_rm):
                    target_vec[i * m_rm * p_rm + j * p_rm + k] -= \
                        U_cur[i * m_rm + j, col_to_remove] * V_cur[j * p_rm + k, col_to_remove] * W_cur[k * n_rm + i, col_to_remove]
        
        # Now we need to find corrections delta_l for l != col_to_remove such that:
        # sum_{l != col_to_remove} delta_l * (U[:,l] * V[:,l] * W[:,l]) = target_vec
        # 
        # This is a linear system. Let's set up the matrix.
        # Unknowns: delta_l for l in [0, rank_cur) \ {col_to_remove}
        num_unknowns = rank_cur - 1
        rhs = target_vec.copy()
        
        # Build the linear system
        # A[r, x] where r indexes (i,j,k) and x indexes the unknowns
        # We'll solve using least squares
        
        # For each unknown x (corresponding to column l_x != col_to_remove):
        # A[r, x] = U[r_u, l_x] * V[r_v, l_x] * W[r_w, l_x]
        # where r = i*m*p + j*p + k, r_u = i*m+j, r_v = j*p+k, r_w = k*n+i
        
        idx_map = []
        for idx in range(rank_cur):
            if idx != col_to_remove:
                idx_map.append(idx)
        
        A = np.zeros((n_rm * m_rm * p_rm, num_unknowns))
        for x, l_x in enumerate(idx_map):
            for i in range(n_rm):
                for j in range(m_rm):
                    for k in range(p_rm):
                        r = i * m_rm * p_rm + j * p_rm + k
                        r_u = i * m_rm + j
                        r_v = j * p_rm + k
                        r_w = k * n_rm + i
                        A[r, x] = U_cur[r_u, l_x] * V_cur[r_v, l_x] * W_cur[r_w, l_x]
        
        # Solve least squares
        try:
            delta, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None, None, None, None
        
        # Check if the solution is good enough
        A_delta = A @ delta
        error = np.linalg.norm(A_delta - rhs) / max(np.linalg.norm(rhs), 1e-15)
        
        if error > 1e-8:
            return None, None, None, None
        
        # Build new matrices
        U_new = np.zeros((n_rm * m_rm, num_unknowns))
        V_new = np.zeros((m_rm * p_rm, num_unknowns))
        W_new = np.zeros((p_rm * n_rm, num_unknowns))
        
        for x, l_x in enumerate(idx_map):
            U_new[:, x] = U_cur[:, l_x] + delta[x]
            V_new[:, x] = V_cur[:, l_x]
            W_new[:, x] = W_cur[:, l_x]
        
        # Verify
        valid = True
        for i in range(n_rm):
            for j in range(m_rm):
                for k in range(p_rm):
                    val = 0.0
                    for l in range(num_unknowns):
                        val += U_new[i * m_rm + j, l] * V_new[j * p_rm + k, l] * W_new[k * n_rm + i, l]
                    if abs(val - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_new, V_new, W_new, delta
        
        return None, None, None, None

    # Try to reduce rank by removing columns one at a time
    # Strategy: try removing each column and see if the linear system has a solution
    U_cur = U_init.copy()
    V_cur = V_init.copy()
    W_cur = W_init.copy()
    rank_cur = current_rank

    # Order of columns to try removing: prefer columns from standard blocks (last 12)
    # as they are more likely to be redundant
    col_order = list(range(rank_cur - 1, -1, -1))  # try from last to first
    
    removed_count = 0
    for col_idx in col_order:
        if time.time() > deadline - 5.0:
            break
        if rank_cur <= 1:
            break
        
        U_new, V_new, W_new, delta = try_reduce_column(U_cur, V_cur, W_cur, rank_cur, col_idx)
        
        if U_new is not None:
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            rank_cur -= 1
            removed_count += 1
            # After removing a column, the column indices shift, so restart
            col_order = list(range(rank_cur - 1, -1, -1))
    
    final_rank = rank_cur
    U_final = U_cur[:, :final_rank]
    V_final = V_cur[:, :final_rank]
    W_final = W_cur[:, :final_rank]

    # ============================================================
    # Phase 3: Verification
    # ============================================================

    def verify(U_v, V_v, W_v):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_v[i * m + j, :] * V_v[j * p + k, :] * W_v[k * n + i, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if not verify(U_final, V_final, W_final):
        # Fallback to trivial rank 60
        rank_fallback = n * m * p
        U_fb = np.zeros((n * m, rank_fallback), dtype=float)
        V_fb = np.zeros((m * p, rank_fallback), dtype=float)
        W_fb = np.zeros((p * n, rank_fallback), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
