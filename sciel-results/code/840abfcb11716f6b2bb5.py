import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The decomposition splits the 4x4 matrix B into two 2x4 blocks B_T and B_B,
    and the 2x4 matrix A into two 2x2 blocks A_L and A_R.
    C = A_L B_T + A_R B_B.
    Each term A_L B_T and A_R B_B is a 2x2 by 2x4 product, which has rank 14.
    Total rank = 14 + 14 = 28.
    
    We use a rank-14 decomposition for 2x2 by 2x4 matrix multiplication.
    This is derived from two independent 2x2 multiplications (Strassen rank 7 each).
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_dict, v_dict, w_dict, col_idx):
        for idx, coeff in u_dict.items():
            if coeff != 0:
                U[idx, col_idx] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                V[idx, col_idx] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                W[idx, col_idx] += coeff
            
    # Strassen terms for 2x2 * 2x2
    # P = X Y
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    strassen_terms = [
        # M1 = (x0+x3)(y0+y3)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (x2+x3)y0
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = x0(y1-y3)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = x3(y2-y0)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (x0+x1)y3
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (x2-x0)(y0+y1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (x1-x3)(y2+y3)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_2x2_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        Apply Strassen terms to compute one 2x2 block of the product.
        a_col_offset: column offset in A (0 or 1)
        b_row_offset: row offset in B (0 or 1)
        b_col_offset: column offset in B (0 or 1)
        c_col_offset: column offset in C (0 or 1)
        """
        for t_idx, (u_local, v_local, w_local) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            u_global = {}
            for loc_idx, coeff in u_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = a_col_offset * 2 + c
                g_idx = r * 4 + g_col
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            v_global = {}
            for loc_idx, coeff in v_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * 4 + g_col
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            w_global = {}
            for loc_idx, coeff in w_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = c_col_offset * 2 + c
                g_idx = g_col * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(u_global, v_global, w_global, col)
            
    # 4 blocks:
    # A_L (cols 0,1) * B_TL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    apply_2x2_block(0, 0, 0, 0, 0)
    # A_R (cols 2,3) * B_BL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    apply_2x2_block(1, 1, 0, 0, 7)
    # A_L (cols 0,1) * B_TR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    apply_2x2_block(0, 0, 1, 1, 14)
    # A_R (cols 2,3) * B_BR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    apply_2x2_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-10):
            print(f"Test failed for trial {trial}")
            print(f"C:\n{C}")
            print(f"Cf:\n{Cf}")
            exit(1)
    print(f"OK rank {U.shape[1]}")
