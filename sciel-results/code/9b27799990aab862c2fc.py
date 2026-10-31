"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-16 exact tensor decomposition for 4x4 matrix multiplication.
    
    Uses the commutative algebra identity: 2x2 matrix multiplication over a
    commutative ring R can be done with 4 R-multiplications:
      M1 = (a+d)(b+c)
      M2 = (a+b)(c+d)  
      M3 = a*b
      M4 = d*c
    with:
      p = M1 + M2 - M3 - M4
      q = M3 + M2
      r = M4 + M1
      s = M4 + M3
    
    Applied hierarchically: 4 outer (2x2 block) x 4 inner (2x2 within block) = 16.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 5.0
    
    rank = 16
    U = np.zeros((n * m, rank))  # (16, 16)
    V = np.zeros((m * p, rank))  # (16, 16)
    W = np.zeros((p * n, rank))  # (16, 16)
    
    def w_idx(i, k):
        """W index: k*n + i"""
        return k * n + i
    
    # Outer Strassen-like 4-term decomposition for 2x2 block structure
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    # 
    # Outer terms (each is a 2x2 "matrix" of linear combos):
    # M1_outer: X = A00 + A11, Y = B00 + B01  -> contributes to C
    # M2_outer: X = A00 + A01, Y = B10 + B11  -> contributes to C
    # M3_outer: X = A00,            Y = B00        -> contributes to C
    # M4_outer: X = A11,            Y = B11        -> contributes to C
    #
    # C reconstruction from outer terms:
    # C00 = M1 + M2 - M3 - M4
    # C01 = M3 + M2
    # C10 = M4 + M1
    # C11 = M4 + M3
    
    # Define outer X and Y factors as 2x2 matrices of dicts (global_row, global_col) -> coeff
    def make_block(br, bc, sign=1):
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                gr = br * 2 + i
                gc = bc * 2 + j
                row.append({(gr, gc): sign})
            mat.append(row)
        return mat
    
    def add_blocks(b1, b2):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d = {}
                for key, val in b1[i][j].items():
                    d[key] = d.get(key, 0) + val
                for key, val in b2[i][j].items():
                    d[key] = d.get(key, 0) + val
                row.append({k: v for k, v in d.items() if v != 0})
            result.append(row)
        return result
    
    A00 = make_block(0, 0)
    A01 = make_block(0, 1)
    A10 = make_block(1, 0)
    A11 = make_block(1, 1)
    
    B00 = make_block(0, 0)
    B01 = make_block(0, 1)
    B10 = make_block(1, 0)
    B11 = make_block(1, 1)
    
    # Outer factors: X_t and Y_t are 2x2 matrices of linear combos
    outer_X = [
        add_blocks(A00, A11),       # M1: A00 + A11
        add_blocks(A00, A01),       # M2: A00 + A01
        A00,                        # M3: A00
        A11,                        # M4: A11
    ]
    outer_Y = [
        add_blocks(B00, B01),       # M1: B00 + B01
        add_blocks(B10, B11),       # M2: B10 + B11
        B00,                        # M3: B00
        B11,                        # M4: B11
    ]
    
    # How each outer term contributes to C blocks (with sign):
    # C00 = M1 + M2 - M3 - M4
    # C01 = M3 + M2
    # C10 = M4 + M1
    # C11 = M4 + M3
    c_targets = [
        [(0, 0, 1), (1, 0, 1)],     # M1 -> C00(+1), C10(+1)
        [(0, 0, 1), (0, 1, 1)],     # M2 -> C00(+1), C01(+1)
        [(0, 0, -1), (0, 1, 1)],    # M3 -> C00(-1), C01(+1)
        [(0, 0, -1), (1, 0, 1), (1, 1, 1)],  # M4 -> C00(-1), C10(+1), C11(+1)
    ]
    
    # Inner 4-term decomposition for 2x2 matrices:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X * Y where:
    #   N1 = (x00+x11)(y00+y10)
    #   N2 = (x00+x01)(y10+y11)
    #   N3 = x00 * y00
    #   N4 = x11 * y11
    #   Z00 = N1 + N2 - N3 - N4
    #   Z01 = N3 + N2
    #   Z10 = N4 + N1
    #   Z11 = N4 + N3
    
    def add_entries(mat, i1, j1, i2, j2):
        result = [[{} for _ in range(2)] for _ in range(2)]
        d = {}
        for key, val in mat[i1][j1].items():
            d[key] = d.get(key, 0) + val
        for key, val in mat[i2][j2].items():
            d[key] = d.get(key, 0) + val
        result[i1][j1] = {k: v for k, v in d.items() if v != 0}
        return result
    
    def single_entry(mat, i, j):
        result = [[{} for _ in range(2)] for _ in range(2)]
        result[i][j] = dict(mat[i][j])
        return result
    
    # Inner factors: 4 terms, each with Xf and Yf as 2x2 matrices of dicts
    # N1: (x00+x11)(y00+y10)
    # N2: (x00+x01)(y10+y11)
    # N3: x00 * y00
    # N4: x11 * y11
    inner_def = [
        # (x_indices, y_indices) where each is a list of (i,j) pairs to sum
        ([(0,0), (1,1)], [(0,0), (1,0)]),   # N1
        ([(0,0), (0,1)], [(1,0), (1,1)]),   # N2
        ([(0,0)], [(0,0)]),                  # N3
        ([(1,1)], [(1,1)]),                  # N4
    ]
    
    # Inner targets: how each inner term contributes to Z entries
    # Z00 = N1 + N2 - N3 - N4
    # Z01 = N3 + N2
    # Z10 = N4 + N1
    # Z11 = N4 + N3
    inner_targets = [
        [(0, 0, 1), (1, 0, 1)],     # N1 -> Z00(+1), Z10(+1)
        [(0, 0, 1), (0, 1, 1)],     # N2 -> Z00(+1), Z01(+1)
        [(0, 0, -1), (0, 1, 1), (1, 1, 1)],  # N3 -> Z00(-1), Z01(+1), Z11(+1)
        [(0, 0, -1), (1, 0, 1), (1, 1, 1)],  # N4 -> Z00(-1), Z10(+1), Z11(+1)
    ]
    
    col = 0
    for t in range(4):
        X = outer_X[t]
        Y = outer_Y[t]
        
        for k in range(4):
            # Compute inner factor Xf, Yf
            x_indices, y_indices = inner_def[k]
            
            Xf = [[{} for _ in range(2)] for _ in range(2)]
            Yf = [[{} for _ in range(2)] for _ in range(2)]
            
            for (xi, xj) in x_indices:
                for key, val in X[xi][xj].items():
                    # Accumulate into Xf[xi][xj]
                    if Xf[xi][xj] is None:
                        Xf[xi][xj] = {}
                    Xf[xi][xj][key] = Xf[xi][xj].get(key, 0) + val
            
            for (yi, yj) in y_indices:
                for key, val in Y[yi][yj].items():
                    if Yf[yi][yj] is None:
                        Yf[yi][yj] = {}
                    Yf[yi][yj][key] = Yf[yi][yj].get(key, 0) + val
            
            # Fill U from Xf
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c), a_v in Xf[i][j].items():
                        U[a_r * m + a_c, col] += a_v
            
            # Fill V from Yf
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c), b_v in Yf[i][j].items():
                        V[b_r * p + b_c, col] += b_v
            
            # Fill W: for each outer target and inner target
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k), col] += c_coeff * z_coeff
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify with numpy if time permits
    if time.time() < _DEADLINE - _MARGIN:
        np.random.seed(42)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_expected = A_test @ B_test
        
        U_r = U.reshape(4, 4, 16)  # [i, j, r]
        V_r = V.reshape(4, 4, 16)  # [j, k, r]
        W_r = W.reshape(4, 4, 16)  # [k, i, r]
        
        u_vals = np.einsum('ijr,ij->ir', U_r, A_test)  # (4, 16)
        v_vals = np.einsum('jkr,jk->kr', V_r, B_test)  # (4, 16)
        
        C_computed = np.einsum('ir,kr,kir->ik', u_vals, v_vals, W_r)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
