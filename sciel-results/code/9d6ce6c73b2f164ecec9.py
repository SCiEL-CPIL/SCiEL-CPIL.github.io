"""Rewritten program for matrix multiplication 3,5,6 rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Build a Strassen-based decomposition
    # Partition: n=[2,1], m=[2,2,1], p=[2,2,2]
    # This gives blocks: 2x2x2 (Strassen rank 7), 2x2x1 (rank 4), 2x1x2 (rank 4),
    # 2x1x1 (rank 2), 1x2x2 (rank 4), 1x2x1 (rank 2), 1x1x2 (rank 2), 1x1x1 (rank 1)
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    # Collect all terms as (u_indices, v_indices, w_indices, coeffs)
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """7 Strassen terms for 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U_idx(la, lc): return a_rows[la]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        terms.append((
            [(U_idx(0,0), 1), (U_idx(1,1), 1)],
            [(V_idx(0,0), 1), (V_idx(1,1), 1)],
            [(W_idx(0,0), 1), (W_idx(1,1), 1)]
        ))
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        terms.append((
            [(U_idx(1,0), 1), (U_idx(1,1), 1)],
            [(V_idx(0,0), 1)],
            [(W_idx(0,1), 1), (W_idx(1,1), -1)]
        ))
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        terms.append((
            [(U_idx(0,0), 1)],
            [(V_idx(0,1), 1), (V_idx(1,1), -1)],
            [(W_idx(1,0), 1), (W_idx(1,1), 1)]
        ))
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        terms.append((
            [(U_idx(1,1), 1)],
            [(V_idx(1,0), 1), (V_idx(0,0), -1)],
            [(W_idx(0,0), 1), (W_idx(0,1), 1)]
        ))
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        terms.append((
            [(U_idx(0,0), 1), (U_idx(0,1), 1)],
            [(V_idx(1,1), 1)],
            [(W_idx(0,0), -1), (W_idx(1,0), 1)]
        ))
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        terms.append((
            [(U_idx(1,0), 1), (U_idx(0,0), -1)],
            [(V_idx(0,0), 1), (V_idx(0,1), 1)],
            [(W_idx(1,1), 1)]
        ))
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        terms.append((
            [(U_idx(0,1), 1), (U_idx(1,1), -1)],
            [(V_idx(1,0), 1), (V_idx(1,1), 1)],
            [(W_idx(0,0), 1)]
        ))
    
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[la]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[la]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,0),1)], [(W_idx(0,1),1)]))
        terms.append(([(U_idx(1,1),1)], [(V_idx(1,0),1)], [(W_idx(0,1),1)]))
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U_idx(la, lc): return a_rows[la]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[la]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,1),1)], [(W_idx(1,0),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,0),1)], [(W_idx(0,1),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,1),1)], [(W_idx(1,1),1)]))
    
    def add_schoolbook_2x1x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[la]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[la]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(1,0),1)], [(V_idx(0,0),1)], [(W_idx(0,1),1)]))
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        def U_idx(la, lc): return a_rows[0]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,1),1)], [(W_idx(1,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,1),1)], [(W_idx(1,0),1)]))
    
    def add_schoolbook_1x2x1(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[0]*m + c_cols[lc]
        def V_idx(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,1),1)], [(V_idx(1,0),1)], [(W_idx(0,0),1)]))
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U_idx(la, lc): return a_rows[0]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W_idx(ld, la): return d_cols[ld]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,1),1)], [(W_idx(1,0),1)]))
    
    def add_schoolbook_1x1x1(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0 = d_cols[0]
        def U_idx(la, lc): return a_rows[0]*m + c_cols[0]
        def V_idx(lc, ld): return c_cols[0]*p + d_cols[0]
        def W_idx(ld, la): return d_cols[0]*n + a_rows[0]
        terms.append(([(U_idx(0,0),1)], [(V_idx(0,0),1)], [(W_idx(0,0),1)]))
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    add_schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    add_schoolbook_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_schoolbook_1x1x1(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row, ci] += coeff
        for row, coeff in v_entries:
            V[row, ci] += coeff
        for row, coeff in w_entries:
            W[row, ci] += coeff
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        # Check: sum_r U[:,r] * V[:,r]^T * W[:,r]^T == matmul tensor
        # Equivalent: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # We check a sample of random matrices
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            C_decomp = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                Ar = np.outer(U_mat[:n*m, r].reshape(n, m), np.ones(1))  # Actually need proper reconstruction
            # Better: reconstruct using the tensor decomposition
            # C[i,k] = sum_r sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for r in range(U_mat.shape[1]):
                        for j in range(m):
                            s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - C_expected[i, k]) > 1e-9:
                        return False
            # Also verify with specific A, B
        # Verify with identity-like test
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U_mat.shape[1]):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linearly dependent columns
    def try_reduce(U_mat, V_mat, W_mat):
        """Try to find and remove linearly dependent columns."""
        current_rank = U_mat.shape[1]
        keep = list(range(current_rank))
        
        # Try removing one column at a time, checking if the remaining still spans
        # Use a greedy approach: try to remove each column and verify
        for r in range(current_rank):
            if r not in keep:
                continue
            # Try removing column r
            new_keep = [k for k in keep if k != r]
            if not new_keep:
                continue
            U_new = U_mat[:, new_keep]
            V_new = V_mat[:, new_keep]
            W_new = W_mat[:, new_keep]
            if verify(U_new, V_new, W_new):
                keep = new_keep
                U_mat = U_new
                V_mat = V_new
                W_mat = W_new
        
        return U_mat, V_mat, W_mat
    
    U, V, W = try_reduce(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
