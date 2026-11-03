"""Matrix multiplication 2x4x4 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2_factors():
    """
    Return the 7 factor definitions for a 2x2 matrix product C = A*B.
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    C = [[c11, c12], [c21, c22]]
    
    Local indices:
    A: 0=a11, 1=a12, 2=a21, 3=a22  (row, col) -> (0,0),(0,1),(1,0),(1,1)
    B: 0=b11, 1=b12, 2=b21, 3=b22  (row, col) -> (0,0),(0,1),(1,0),(1,1)
    C: 0=c11, 1=c12, 2=c21, 3=c22  (row, col) -> (0,0),(0,1),(1,0),(1,1)
    
    Strassen:
    M1 = (a11 + a22) * (b11 + b22)
    M2 = (a21 + a22) * b11
    M3 = a11 * (b12 - b22)
    M4 = a22 * (b21 - b11)
    M5 = (a11 + a12) * b22
    M6 = (a21 - a11) * (b11 + b12)
    M7 = (a12 - a22) * (b21 + b22)
    
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    """
    # Each entry: (a_local_indices, a_signs, b_local_indices, b_signs, c_local_indices, c_signs)
    strassen_defs = [
        # M1 = (a11 + a22) * (b11 + b22)
        # A: a11(+1) at idx 0, a22(+1) at idx 3
        # B: b11(+1) at idx 0, b22(+1) at idx 3
        # C: c11(+1) at idx 0, c22(+1) at idx 3
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        
        # M2 = (a21 + a22) * b11
        # A: a21(+1) at idx 2, a22(+1) at idx 3
        # B: b11(+1) at idx 0
        # C: c21(+1) at idx 2, c22(-1) at idx 3
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        
        # M3 = a11 * (b12 - b22)
        # A: a11(+1) at idx 0
        # B: b12(+1) at idx 1, b22(-1) at idx 3
        # C: c12(+1) at idx 1, c22(+1) at idx 3
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        
        # M4 = a22 * (b21 - b11)
        # A: a22(+1) at idx 3
        # B: b21(+1) at idx 2, b11(-1) at idx 0
        # C: c11(+1) at idx 0, c21(+1) at idx 2
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        
        # M5 = (a11 + a12) * b22
        # A: a11(+1) at idx 0, a12(+1) at idx 1
        # B: b22(+1) at idx 3
        # C: c11(-1) at idx 0, c12(+1) at idx 1
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        
        # M6 = (a21 - a11) * (b11 + b12)
        # A: a21(+1) at idx 2, a11(-1) at idx 0
        # B: b11(+1) at idx 0, b12(+1) at idx 1
        # C: c22(+1) at idx 3
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        
        # M7 = (a12 - a22) * (b21 + b22)
        # A: a12(+1) at idx 1, a22(-1) at idx 3
        # B: b21(+1) at idx 2, b22(+1) at idx 3
        # C: c11(+1) at idx 0
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    return strassen_defs


def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    strassen_defs = _strassen_2x2_factors()
    rank = 28  # 4 products * 7 Strassen multiplications each
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Local index to (row, col) mapping for 2x2 matrices
    local_to_rc = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    
    # Define the 4 sub-products:
    # Each: (a_col_offset, b_row_offset, b_col_offset, c_k_offset, c_i_offset)
    # A is 2x4: A[i, j], i in [0,1], j in [0,3]
    # B is 4x4: B[j, k], j in [0,3], k in [0,3]
    # C is 4x2: C[k, i], k in [0,3], i in [0,1] (output W indexed by k*n+i)
    
    # Product 0: A[:,0:2] * B[0:2, 0:2] -> C[0:2, :]
    #   A cols 0-1, B rows 0-1, B cols 0-1, C k=0-1, i=0-1
    # Product 1: A[:,0:2] * B[0:2, 2:4] -> C[2:4, :]
    #   A cols 0-1, B rows 0-1, B cols 2-3, C k=2-3, i=0-1
    # Product 2: A[:,2:4] * B[2:4, 0:2] -> C[0:2, :]
    #   A cols 2-3, B rows 2-3, B cols 0-1, C k=0-1, i=0-1
    # Product 3: A[:,2:4] * B[2:4, 2:4] -> C[2:4, :]
    #   A cols 2-3, B rows 2-3, B cols 2-3, C k=2-3, i=0-1
    
    products = [
        (0, 0, 0, 0, 0),  # a_col_off, b_row_off, b_col_off, c_k_off, c_i_off
        (0, 0, 2, 2, 0),
        (2, 2, 0, 0, 0),
        (2, 2, 2, 2, 0),
    ]
    
    for prod_idx, (a_col_off, b_row_off, b_col_off, c_k_off, c_i_off) in enumerate(products):
        for s_idx, (a_idx, a_sig, b_idx, b_sig, c_idx, c_sig) in enumerate(strassen_defs):
            r = prod_idx * 7 + s_idx
            
            # Fill U: U[i*m + j, r]
            # A local (i_local, j_local) -> global (i_local, a_col_off + j_local)
            for a_local_idx, sign in zip(a_idx, a_sig):
                i_local, j_local = local_to_rc[a_local_idx]
                i_global = i_local
                j_global = a_col_off + j_local
                U[i_global * m + j_global, r] = sign
            
            # Fill V: V[j*p + k, r]
            # B local (j_local, k_local) -> global (b_row_off + j_local, b_col_off + k_local)
            for b_local_idx, sign in zip(b_idx, b_sig):
                j_local, k_local = local_to_rc[b_local_idx]
                j_global = b_row_off + j_local
                k_global = b_col_off + k_local
                V[j_global * p + k_global, r] = sign
            
            # Fill W: W[k*n + i, r]
            # C local (i_local, k_local) -> global (c_k_off + k_local, c_i_off + i_local)
            for c_local_idx, sign in zip(c_idx, c_sig):
                i_local, k_local = local_to_rc[c_local_idx]
                k_global = c_k_off + k_local
                i_global = c_i_off + i_local
                W[k_global * n + i_global, r] = sign
    
    # Verify tensor decomposition: for all (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let's verify with a random test
    np.random.seed(12345)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((p, n))
    for k in range(p):
        for i in range(n):
            total = 0.0
            for r in range(rank):
                ua = 0.0
                for j in range(m):
                    ua += U[i * m + j, r] * A_test[i, j]
                vb = 0.0
                for j in range(m):
                    vb += V[j * p + k, r] * B_test[j, k]
                total += W[k * n + i, r] * ua * vb
            C_computed[k, i] = total
    
    assert np.allclose(C_computed, C_expected, atol=1e-10), f"Verification failed! Max error: {np.max(np.abs(C_computed - C_expected))}"
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
