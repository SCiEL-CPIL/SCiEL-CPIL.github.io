import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The decomposition uses 4 independent Strassen (rank-7) blocks:
    - A is 2x4, B is 4x4, C is 2x4
    - Split A into A_L (2x2) and A_R (2x2)
    - Split B into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR
    - C_L = A_L @ B_TL + A_R @ B_BL (2x2 result)
    - C_R = A_L @ B_TR + A_R @ B_BR (2x2 result)
    - Each 2x2 block product uses Strassen's rank-7 algorithm
    """
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_dict, v_dict, w_dict, col_idx):
        """Add a single rank-1 term to the decomposition matrices."""
        for idx, coeff in u_dict.items():
            if coeff != 0:
                U[idx, col_idx] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                V[idx, col_idx] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                W[idx, col_idx] += coeff
            
    # Strassen decomposition terms for 2x2 matrix multiplication P = X * Y
    # X = [[X00, X01], [X10, X11]], Y = [[Y00, Y01], [Y10, Y11]]
    # P = X @ Y
    strassen_terms = [
        # M1 = (X00 + X11) * (Y00 + Y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (X10 + X11) * Y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = X00 * (Y01 - Y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = X11 * (Y10 - Y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (X00 + X01) * Y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (X10 - X00) * (Y00 + Y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (X01 - X11) * (Y10 + Y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        Apply Strassen terms to compute one 2x2 block of the product.
        
        a_col_offset: which 2-column block of A (0 or 1)
        b_row_offset: which 2-row block of B (0 or 1)  
        b_col_offset: which 2-col block of B (0 or 1)
        c_col_offset: which 2-col block of C (0 or 1)
        """
        for t_idx, (u_local, v_local, w_local) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            # Map local A indices to global U indices
            # A is 2x4, flattened as A[i*4+j]
            u_global = {}
            for loc_idx, coeff in u_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = a_col_offset * 2 + c
                g_idx = r * 4 + g_col
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            # Map local B indices to global V indices
            # B is 4x4, flattened as B[i*4+j]
            v_global = {}
            for loc_idx, coeff in v_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * 4 + g_col
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            # Map local C indices to global W indices
            # C is 2x4, output factor uses trace convention W[k*n+i, r]
            # W index: k*2 + i where k is col of C, i is row of C
            w_global = {}
            for loc_idx, coeff in w_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = c_col_offset * 2 + c
                g_idx = g_col * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(u_global, v_global, w_global, col)
    
    # Four 2x2 block products:
    # C_L = A_L @ B_TL + A_R @ B_BL
    # C_R = A_L @ B_TR + A_R @ B_BR
    apply_strassen_block(0, 0, 0, 0, 0)   # A_L @ B_TL -> C_L
    apply_strassen_block(1, 1, 0, 0, 7)   # A_R @ B_BL -> C_L
    apply_strassen_block(0, 0, 1, 1, 14)  # A_L @ B_TR -> C_R
    apply_strassen_block(1, 1, 1, 1, 21)  # A_R @ B_BR -> C_R
    
    # Verify within time budget
    if time.time() - t_start < DEADLINE - 5:
        Uf = U.astype(float)
        Vf = V.astype(float)
        Wf = W.astype(float)
        np.random.seed(12345)
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C = A @ B
            Cf = np.zeros((n, p))
            for ii in range(n):
                for kk in range(p):
                    acc = 0.0
                    for cc in range(rank):
                        for jj in range(m):
                            acc += Uf[ii * m + jj, cc] * Vf[jj * p + kk, cc] * Wf[kk * n + ii, cc]
                    Cf[ii, kk] = acc
            if not np.allclose(C, Cf, atol=1e-10):
                # If verification fails, we still return (shouldn't happen)
                pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    rank = U.shape[1]
    
    # Test with multiple random matrices using vectorized approach
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        # Vectorized: Cf[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                # For each rank index c, sum over j
                for cc in range(rank):
                    u_vec = U[ii * m:(ii+1) * m, cc]  # shape (m,)
                    v_vec = V.reshape(m, p, rank)[:, :, cc]  # shape (m, p)
                    Cf[ii, kk] += np.dot(u_vec, v_vec[:, kk]) * W[kk * n + ii, cc]
        if not np.allclose(C, Cf, atol=1e-10):
            print(f"Test failed for trial {trial}")
            print(f"C:\n{C}")
            print(f"Cf:\n{Cf}")
            exit(1)
    print(f"OK rank {rank} in {time.time()-t0:.1f}s")
