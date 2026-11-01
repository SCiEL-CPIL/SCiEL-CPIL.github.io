"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, use 145s as safety margin
    
    n, m, p = 2, 4, 4
    
    # Build the target tensor T
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                T_full[a_idx, b_idx, c_idx] = 1
    
    rank_28 = 28
    U = np.zeros((n*m, rank_28))
    V = np.zeros((m*p, rank_28))
    W = np.zeros((p*n, rank_28))
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1
        U[a00_u, offset + 0] += 1
        U[a11_u, offset + 0] += 1
        V[b00_v, offset + 0] += 1
        V[b11_v, offset + 0] += 1
        W[c00_w, offset + 0] += 1
        W[c11_w, offset + 0] += 1
        
        # M2
        U[a10_u, offset + 1] += 1
        U[a11_u, offset + 1] += 1
        V[b00_v, offset + 1] += 1
        W[c10_w, offset + 1] += 1
        W[c11_w, offset + 1] -= 1
        
        # M3
        U[a00_u, offset + 2] += 1
        V[b01_v, offset + 2] += 1
        V[b11_v, offset + 2] -= 1
        W[c01_w, offset + 2] += 1
        W[c11_w, offset + 2] += 1
        
        # M4
        U[a11_u, offset + 3] += 1
        V[b10_v, offset + 3] += 1
        V[b00_v, offset + 3] -= 1
        W[c00_w, offset + 3] += 1
        W[c10_w, offset + 3] += 1
        
        # M5
        U[a00_u, offset + 4] += 1
        U[a01_u, offset + 4] += 1
        V[b11_v, offset + 4] += 1
        W[c00_w, offset + 4] -= 1
        W[c01_w, offset + 4] += 1
        
        # M6
        U[a10_u, offset + 5] += 1
        U[a00_u, offset + 5] -= 1
        V[b00_v, offset + 5] += 1
        V[b01_v, offset + 5] += 1
        W[c11_w, offset + 5] += 1
        
        # M7
        U[a01_u, offset + 6] += 1
        U[a11_u, offset + 6] -= 1
        V[b10_v, offset + 6] += 1
        V[b11_v, offset + 6] += 1
        W[c00_w, offset + 6] += 1
    
    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 0, 0, 2, 0, 2)
    add_strassen_block(14, 0, 2, 2, 0, 0, 0)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Verify
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        rng = np.random.RandomState(42)
        for _ in range(10):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for r in range(U_mat.shape[1]):
                        u_val = np.dot(U_mat[:, r], A_flat)
                        v_val = np.dot(V_mat[:, r], B_flat)
                        s += u_val * v_val * W_mat[j * n + i, r]
                    C_computed[i, j] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check for linear dependencies to potentially reduce rank
    # This is a fast check. If rank < 28, we can eliminate a term.
    vecs = []
    for r in range(rank_28):
        vec = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]).flatten()
        vecs.append(vec)
    mat = np.array(vecs).T
    
    try:
        s_vals = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
        rank_mat = np.sum(s_vals > 1e-10)
        
        if rank_mat < 28:
            _, s_vals, Vt = np.linalg.svd(mat, full_matrices=False)
            null_vecs = Vt[rank_mat:]
            
            for nv in null_vecs:
                if time.time() > deadline - 5:
                    break
                max_idx = np.argmax(np.abs(nv))
                if abs(nv[max_idx]) > 1e-10:
                    U_sub = np.delete(U, max_idx, axis=1)
                    V_sub = np.delete(V, max_idx, axis=1)
                    W_sub = np.delete(W, max_idx, axis=1)
                    
                    T_sub = np.zeros((n*m, m*p, p*n))
                    for r in range(27):
                        T_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                    
                    if np.allclose(T_sub, T_full, atol=1e-10):
                        return {"U": U_sub.tolist(), "V": V_sub.tolist(), "W": W_sub.tolist()}
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
