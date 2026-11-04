"""Matrix multiplication 5x5x5 - Block decomposition with Strassen for 2x2x2 blocks."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = A * B
    
    Strassen products:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4,       M1-M2+M3+M6]]
    """
    strassen_terms = []
    
    # M1 = (a00+a11)(b00+b11)
    strassen_terms.append((
        [0, 3], [1, 1],
        [0, 3], [1, 1],
        [0, 2, 3], [1, 0, 1],
    ))
    
    # M2 = (a10+a11)b00
    strassen_terms.append((
        [2, 3], [1, 1],
        [0], [1],
        [2, 3], [1, -1],
    ))
    
    # M3 = a00(b01-b11)
    strassen_terms.append((
        [0], [1],
        [1, 3], [1, -1],
        [1, 3], [1, 1],
    ))
    
    # M4 = a11(b10-b00)
    strassen_terms.append((
        [3], [1],
        [2, 0], [1, -1],
        [0, 2], [1, 1],
    ))
    
    # M5 = (a00+a01)b11
    strassen_terms.append((
        [0, 1], [1, 1],
        [3], [1],
        [0, 1], [-1, 1],
    ))
    
    # M6 = (a10-a00)(b00+b01)
    strassen_terms.append((
        [2, 0], [1, -1],
        [0, 1], [1, 1],
        [3], [1],
    ))
    
    # M7 = (a01-a11)(b10+b11)
    strassen_terms.append((
        [1, 3], [1, -1],
        [2, 3], [1, 1],
        [0], [1],
    ))
    
    return strassen_terms

def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U[i*m+j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for j in range(m):
                v_vals += V[j*p+k, :] * B[j, k]
            C_decomp[i, k] = np.dot(W[k*n+i, :], u_vals * v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Partition: 2 + 2 + 1
    row_splits_A = [0, 2, 4, 5]
    col_splits_A = [0, 2, 4, 5]
    row_splits_B = [0, 2, 4, 5]
    col_splits_B = [0, 2, 4, 5]
    row_splits_C = [0, 2, 4, 5]
    col_splits_C = [0, 2, 4, 5]
    
    num_blocks = 3
    
    A_block_dims = {}
    for i in range(num_blocks):
        for j in range(num_blocks):
            hi = row_splits_A[i+1] - row_splits_A[i]
            wj = col_splits_A[j+1] - col_splits_A[j]
            A_block_dims[(i,j)] = (hi, wj)
            
    B_block_dims = {}
    for j in range(num_blocks):
        for k in range(num_blocks):
            hj = row_splits_B[j+1] - row_splits_B[j]
            wk = col_splits_B[k+1] - col_splits_B[k]
            B_block_dims[(j,k)] = (hj, wk)
            
    C_block_dims = {}
    for i in range(num_blocks):
        for k in range(num_blocks):
            hi = row_splits_C[i+1] - row_splits_C[i]
            wk = col_splits_C[k+1] - col_splits_C[k]
            C_block_dims[(i,k)] = (hi, wk)
            
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    terms = []
    
    strassen = _strassen_2x2()
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ha, wa = A_block_dims[(i,j)]
                hb, wb = B_block_dims[(j,k)]
                hc, wc = C_block_dims[(i,k)]
                
                assert ha == hc
                assert wa == hb
                
                a_row_start = row_splits_A[i]
                a_col_start = col_splits_A[j]
                b_row_start = row_splits_B[j]
                b_col_start = col_splits_B[k]
                c_row_start = row_splits_C[i]
                c_col_start = col_splits_C[k]
                
                if ha == 2 and wa == 2 and hb == 2 and wb == 2:
                    # Strassen 2x2x2
                    for term in strassen:
                        u_local, u_coeffs, v_local, v_coeffs, w_local, w_coeffs = term
                        u_dict = {}
                        for li, coeff in zip(u_local, u_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_u = (a_row_start + li_row) * m + (a_col_start + li_col)
                                u_dict[global_u] = u_dict.get(global_u, 0) + coeff
                                
                        v_dict = {}
                        for li, coeff in zip(v_local, v_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_v = (b_row_start + li_row) * p + (b_col_start + li_col)
                                v_dict[global_v] = v_dict.get(global_v, 0) + coeff
                                
                        w_dict = {}
                        for li, coeff in zip(w_local, w_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_w = (c_col_start + li_col) * n + (c_row_start + li_row)
                                w_dict[global_w] = w_dict.get(global_w, 0) + coeff
                                
                        terms.append((u_dict, v_dict, w_dict))
                else:
                    # Schoolbook for non-2x2x2 blocks
                    for li in range(ha):
                        for lj in range(wa):
                            for lk in range(wb):
                                global_u = (a_row_start + li) * m + (a_col_start + lj)
                                global_v = (b_row_start + lj) * p + (b_col_start + lk)
                                global_w = (c_col_start + lk) * n + (c_row_start + li)
                                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < U_rows:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < V_rows:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < W_rows:
                W[idx][c] = coeff
                
    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
