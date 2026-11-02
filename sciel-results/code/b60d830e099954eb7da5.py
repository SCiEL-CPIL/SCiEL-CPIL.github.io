import numpy as np
import time

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # U: (i, k) -> i*2+k
    # V: (k, j) -> k*2+j
    # W: (j, i) -> j*2+i
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Blocks:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = (none, A is 2x4, so no A10 in the sense of rows, A has 2 rows)
    # Wait, A is 2x4.
    # A = [[A00, A01]] where A00, A01 are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bi are 2x2.
    # C = [[C00, C01]] where C00, C01 are 2x2.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    
    # So we have 4 multiplications:
    # 1. A00 * B00 -> contributes to C00
    # 2. A01 * B10 -> contributes to C00
    # 3. A00 * B01 -> contributes to C01
    # 4. A01 * B11 -> contributes to C01
    
    # Mapping:
    # A00 is A[:, 0:2]. Indices in U: i*4 + k.
    #   i=0,1. k=0,1.
    #   U indices: 0,1, 4,5.
    # A01 is A[:, 2:4]. Indices in U: i*4 + k.
    #   i=0,1. k=2,3.
    #   U indices: 2,3, 6,7.
    
    # B00 is B[0:2, 0:2]. Indices in V: k*4 + j.
    #   k=0,1. j=0,1.
    #   V indices: 0,1, 4,5.
    # B10 is B[2:4, 0:2]. Indices in V: k*4 + j.
    #   k=2,3. j=0,1.
    #   V indices: 8,9, 12,13.
    # B01 is B[0:2, 2:4]. Indices in V: k*4 + j.
    #   k=0,1. j=2,3.
    #   V indices: 2,3, 6,7.
    # B11 is B[2:4, 2:4]. Indices in V: k*4 + j.
    #   k=2,3. j=2,3.
    #   V indices: 10,11, 14,15.
    
    # C00 is C[:, 0:2]. Indices in W: j*2 + i.
    #   j=0,1. i=0,1.
    #   W indices: 0,1, 2,3.
    # C01 is C[:, 2:4]. Indices in W: j*2 + i.
    #   j=2,3. i=0,1.
    #   W indices: 4,5, 6,7.
    
    blocks = [
        {
            'A_col_offset': 0, # A00
            'B_row_offset': 0, 'B_col_offset': 0, # B00
            'C_col_offset': 0, # C00
        },
        {
            'A_col_offset': 2, # A01
            'B_row_offset': 2, 'B_col_offset': 0, # B10
            'C_col_offset': 0, # C00
        },
        {
            'A_col_offset': 0, # A00
            'B_row_offset': 0, 'B_col_offset': 2, # B01
            'C_col_offset': 2, # C01
        },
        {
            'A_col_offset': 2, # A01
            'B_row_offset': 2, 'B_col_offset': 2, # B11
            'C_col_offset': 2, # C01
        }
    ]
    
    current_rank = 0
    for block in blocks:
        A_off = block['A_col_offset']
        B_row_off = block['B_row_offset']
        B_col_off = block['B_col_offset']
        C_col_off = block['C_col_offset']
        
        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term
            
            for loc_idx, val in coeff_A.items():
                li, lj = divmod(loc_idx, 2)
                u_idx = li * 4 + (A_off + lj)
                U[u_idx, current_rank] += val
            
            for loc_idx, val in coeff_B.items():
                li, lj = divmod(loc_idx, 2)
                v_idx = (B_row_off + li) * 4 + (B_col_off + lj)
                V[v_idx, current_rank] += val
            
            for loc_idx, val in coeff_C.items():
                li, lj = divmod(loc_idx, 2)
                w_idx = (C_col_off + lj) * 2 + li
                W[w_idx, current_rank] += val
            
            current_rank += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 2, 4, 4
    # Check tensor identity
    # T_ijk = sum_r U_{i*m+k, r} V_{k*p+j, r} W_{j*n+i, r}
    # Should be 1 for all i, j, k
    
    # Construct random A, B
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_direct = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            s = 0
            for k in range(m):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                s += np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            C_decomp[i, j] = s
            
    if np.allclose(C_direct, C_decomp):
        print("Verification passed")
    else:
        print("Verification failed")
        print(C_direct)
        print(C_decomp)
