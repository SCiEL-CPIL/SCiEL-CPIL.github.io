import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    Strategy:
    1. Split A (2x4) into A_L (2x2) and A_R (2x2).
    2. Split B (4x4) into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR.
    3. The result C (2x4) is split into C_L (2x2) and C_R (2x2).
       C_L = A_L @ B_TL + A_R @ B_BL
       C_R = A_L @ B_TR + A_R @ B_BR
    4. Use the identity X@Y + Z@W = 0.5 * ((X+Z)@(Y+W) + (X-Z)@(Y-W))
       to compute each half (C_L and C_R) using two 2x2 matrix multiplications.
    5. Each 2x2 matrix multiplication is decomposed using Strassen's algorithm (rank 7).
    6. Total rank = 2 (for C_L) * 7 + 2 (for C_R) * 7 = 28.
    
    Interface:
    - U: (n*m, rank) where n=2, m=4
    - V: (m*p, rank) where m=4, p=4
    - W: (p*n, rank) where p=4, n=2 (W[k*n + i, r])
    """
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
    # Indexing: local 2x2 indices are 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1)
    #
    # Standard Strassen formulas:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    #
    # Result:
    # P00 = M1 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6

    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col, scale=1):
        """
        Apply Strassen terms to compute one 2x2 block of the product.
        
        Parameters:
        - a_col_offset: column offset in A (0 or 1, each offset is 2 columns)
        - b_row_offset: row offset in B (0 or 1, each offset is 2 rows)
        - b_col_offset: column offset in B (0 or 1, each offset is 2 columns)
        - c_col_offset: column offset in C (0 or 1, each offset is 2 columns)
        - start_col: starting column in U, V, W for this block
        - scale: multiplier for all coefficients (used for the 0.5 factor in sum-of-products)
        """
        for t_idx, (u_local, v_local, w_local) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            # Map local A indices to global U indices
            # A is 2x4: global index = row * 4 + col
            # Local (r,c) -> global (r, a_col_offset*2 + c)
            u_global = {}
            for loc_idx, coeff in u_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = a_col_offset * 2 + c
                g_idx = r * 4 + g_col
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff * scale
            
            # Map local B indices to global V indices
            # B is 4x4: global index = row * 4 + col
            # Local (r,c) -> global (b_row_offset*2 + r, b_col_offset*2 + c)
            v_global = {}
            for loc_idx, coeff in v_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * 4 + g_col
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff * scale
            
            # Map local C indices to global W indices
            # C is 2x4: W[k*n + i] where k is column, i is row
            # Local (r,c) -> global (c_col_offset*2 + c is col k, r is row i)
            # W index = k * n + i = (c_col_offset*2 + c) * 2 + r
            w_global = {}
            for loc_idx, coeff in w_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = c_col_offset * 2 + c
                g_idx = g_col * 2 + r  # W[k*n + i] = W[col * 2 + row]
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff * scale
            
            add_term(u_global, v_global, w_global, col)
            
    # We use the identity: X@Y + Z@W = 0.5 * ((X+Z)@(Y+W) + (X-Z)@(Y-W))
    # This requires 4 Strassen blocks in total for the whole 2x4x4 multiplication:
    # 1. (A_L + A_R) @ (B_TL + B_BL) -> contributes to C_L
    # 2. (A_L - A_R) @ (B_TL - B_BL) -> contributes to C_L
    # 3. (A_L + A_R) @ (B_TR + B_BR) -> contributes to C_R
    # 4. (A_L - A_R) @ (B_TR - B_BR) -> contributes to C_R
    
    # However, the standard Strassen block application above assumes the inputs are
    # the raw blocks A_L, B_TL, etc.
    # To implement the "sum of products" trick, we would need to modify the inputs
    # to the Strassen blocks to be the sums/differences.
    # The current `apply_strassen_block` maps local indices directly to global indices.
    # It does NOT support linear combinations of input blocks easily without
    # modifying the U and V matrices to reflect the sums.
    
    # For simplicity and robustness, we stick to the 4 independent blocks:
    # C_L = A_L @ B_TL + A_R @ B_BL
    # C_R = A_L @ B_TR + A_R @ B_BR
    # This is 4 independent 2x2 matmuls, each rank 7. Total 28.
    
    # Block 1: A_L @ B_TL -> C_L (cols 0,1)
    apply_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A_R @ B_BL -> C_L (cols 0,1)
    apply_strassen_block(1, 1, 0, 0, 7)
    
    # Block 3: A_L @ B_TR -> C_R (cols 2,3)
    apply_strassen_block(0, 0, 1, 1, 14)
    
    # Block 4: A_R @ B_BR -> C_R (cols 2,3)
    apply_strassen_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with multiple random matrices
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
