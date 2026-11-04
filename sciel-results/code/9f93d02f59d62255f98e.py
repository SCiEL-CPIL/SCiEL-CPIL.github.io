"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

Block decomposition using Strassen for 2x2 sub-blocks:
- Split rows: 2+1, columns: 3+4
- 2x4 * 4x3: split middle into 2+2, each 2x2*2x3 uses Strassen(7)+schoolbook(4)=11, total 22
- 2x4 * 4x4: split middle into 2+2, each 2x2*2x4 uses Strassen(7)+Strassen(7)=14, total 28
- 1x4 * 4x7: schoolbook = 28
Total: 22 + 28 + 28 = 78
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
            
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_schoolbook(n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate schoolbook decomposition for a block."""
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    def gen_strassen_2x2x2(i_off, j_off, k_off):
        """Strassen decomposition for 2x2 * 2x2."""
        i0 = i_off; i1 = i_off+1
        j0 = j_off; j1 = j_off+1
        k0 = k_off; k1 = k_off+1
        
        # M1 = (A11+A22)(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0, k1*n+i1], [1, 1])
        
        # M2 = (A21+A22)B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0], [1],
                      [k0*n+i1, k1*n+i1], [1, -1])
        
        # M3 = A11(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                      [j0*p+k1, j1*p+k1], [1, -1],
                      [k1*n+i0, k1*n+i1], [1, 1])
        
        # M4 = A22(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                      [j1*p+k0, j0*p+k0], [1, -1],
                      [k0*n+i0, k0*n+i1], [1, 1])
        
        # M5 = (A11+A12)B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                      [j1*p+k1], [1],
                      [k0*n+i0, k1*n+i0], [-1, 1])
        
        # M6 = (A21-A11)(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                      [j0*p+k0, j0*p+k1], [1, 1],
                      [k1*n+i1], [1])
        
        # M7 = (A12-A22)(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                      [j1*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0], [1])

    def gen_2x2x2x3(i_off, j_off, k_off):
        """2x2 * 2x3: Strassen for first 2 cols + schoolbook for last col. Rank 7+4=11."""
        gen_strassen_2x2x2(i_off, j_off, k_off)
        # Schoolbook for the 3rd column (k_off+2)
        for i in range(2):
            for j in range(2):
                ii = i_off + i
                jj = j_off + j
                kk = k_off + 2
                add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    def gen_2x2x2x4(i_off, j_off, k_off):
        """2x2 * 2x4: Strassen for cols 0-1 + Strassen for cols 2-3. Rank 7+7=14."""
        gen_strassen_2x2x2(i_off, j_off, k_off)
        gen_strassen_2x2x2(i_off, j_off, k_off + 2)

    # Build the decomposition:
    # Rows split: 2+1 (i_off=0 for first 2 rows, i_off=2 for last row)
    # Cols split: 3+4 (k_off=0 for first 3, k_off=3 for last 4)
    
    # Block: 2x4 * 4x3 (rows 0-1, cols 0-2)
    # Split middle (j) into 2+2
    gen_2x2x2x3(0, 0, 0)   # rows 0-1, middle 0-1, cols 0-2
    gen_2x2x2x3(0, 2, 0)   # rows 0-1, middle 2-3, cols 0-2
    
    # Block: 2x4 * 4x4 (rows 0-1, cols 3-6)
    # Split middle (j) into 2+2
    gen_2x2x2x4(0, 0, 3)   # rows 0-1, middle 0-1, cols 3-6
    gen_2x2x2x4(0, 2, 3)   # rows 0-1, middle 2-3, cols 3-6
    
    # Block: 1x4 * 4x7 (row 2, all cols)
    gen_schoolbook(1, 4, 7, 2, 0, 0)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T
    
    rank = U.shape[1]
    
    # Verify correctness using the tensor contraction
    ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    U_idx = ii * m + jj
    V_idx = jj * p + kk
    W_idx = kk * n + ii
    
    U_sel = U[U_idx]
    V_sel = V[V_idx]
    W_sel = W[W_idx]
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    if not np.allclose(vals, 1.0, atol=1e-9):
        # Fallback: schoolbook rank 84
        U_vecs_fb = []
        V_vecs_fb = []
        W_vecs_fb = []
        
        def add_rank1_term_fb(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            u = np.zeros(n * m, dtype=float)
            v = np.zeros(m * p, dtype=float)
            w = np.zeros(p * n, dtype=float)
            for idx, val in zip(u_indices, u_vals):
                u[idx] += val
            for idx, val in zip(v_indices, v_vals):
                v[idx] += val
            for idx, val in zip(w_indices, w_vals):
                w[idx] += val
            U_vecs_fb.append(u)
            V_vecs_fb.append(v)
            W_vecs_fb.append(w)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_rank1_term_fb([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
        
        U_fb = np.array(U_vecs_fb).T
        V_fb = np.array(V_vecs_fb).T
        W_fb = np.array(W_vecs_fb).T
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Check half-integer property
    all_half_int = np.allclose(U * 2, np.round(U * 2)) and \
                   np.allclose(V * 2, np.round(V * 2)) and \
                   np.allclose(W * 2, np.round(W * 2))
    
    if all_half_int:
        return {"U": (np.round(U * 2) / 2).tolist(), 
                "V": (np.round(V * 2) / 2).tolist(), 
                "W": (np.round(W * 2) / 2).tolist()}
    
    # Return as-is if not half-integer (shouldn't happen with our construction)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
