"""Matrix multiplication 5x5x5 - Optimized block decomposition with minimal-rank sub-blocks."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = time.time() + 140

def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    Returns: list of (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    Local indices: A: 0=a00,1=a01,2=a10,3=a11; B: 0=b00,1=b01,2=b10,3=b11; C: 0=c00,1=c01,2=c10,3=c11
    """
    return [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 2, 3], [1, 0, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]


def _block_decomp(ha, wa, hb, wb):
    """
    Return minimal-rank decomposition for ha x wa matrix times hb x wb matrix.
    Assumes ha == wc and wa == hb (consistent dimensions).
    Returns list of (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    where indices are local (row-major within each block).
    """
    if ha == 2 and wa == 2 and hb == 2 and wb == 2:
        return _strassen_2x2()
    
    # For other sizes, use optimized decompositions
    terms = []
    
    if wa == 1 and hb == 1:
        # Outer product: (ha x 1) * (1 x wb) -> ha x wb
        # Rank 1: single term
        u_idx = [0]  # The single column of A (only one element per row)
        # Actually A is ha x 1, so local indices are 0..ha-1
        u_idx = list(range(ha))
        u_coeffs = [1] * ha
        v_idx = [0]
        v_coeffs = [1]
        # W: all ha*wb entries get coefficient 1
        w_idx = list(range(ha * wb))
        w_coeffs = [1] * (ha * wb)
        terms.append((u_idx, u_coeffs, v_idx, v_coeffs, w_idx, w_coeffs))
    elif ha == 1 and wb == 1:
        # (1 x wa) * (wa x 1) -> 1 x 1, scalar product
        # Rank 1
        u_idx = list(range(wa))
        u_coeffs = [1] * wa
        v_idx = list(range(wa))
        v_coeffs = [1] * wa
        w_idx = [0]
        w_coeffs = [1]
        terms.append((u_idx, u_coeffs, v_idx, v_coeffs, w_idx, w_coeffs))
    elif wa == 1:
        # (ha x 1) * (1 x wb) already handled above
        pass
    elif hb == 1:
        # (ha x 1) * (1 x wb) - same as wa==1 case since hb==wa
        pass
    elif ha == 1:
        # (1 x wa) * (wa x wb) -> 1 x wb
        # Each output C[0,lk] = sum_lj A[0,lj]*B[lj,lk]
        # Rank wb (one term per output column)
        for lk in range(wb):
            u_idx = list(range(wa))
            u_coeffs = [1] * wa
            v_idx = [lk * 1 + lj for lj in range(wa)]  # B[lj, lk], local B is wa x wb, row-major: index = lj*wb + lk
            v_idx = [lj * wb + lk for lj in range(wa)]
            v_coeffs = [1] * wa
            w_idx = [lk]
            w_coeffs = [1]
            terms.append((u_idx, u_coeffs, v_idx, v_coeffs, w_idx, w_coeffs))
    elif wb == 1:
        # (ha x wa) * (wa x 1) -> ha x 1
        # Each output C[li,0] = sum_lj A[li,lj]*B[lj,0]
        # Rank ha (one term per output row)
        for li in range(ha):
            u_idx = [li * wa + lj for lj in range(wa)]
            u_coeffs = [1] * wa
            v_idx = list(range(wa))
            v_coeffs = [1] * wa
            w_idx = [li]
            w_coeffs = [1]
            terms.append((u_idx, u_coeffs, v_idx, v_coeffs, w_idx, w_coeffs))
    else:
        # General case: schoolbook
        for li in range(ha):
            for lj in range(wa):
                for lk in range(wb):
                    u_idx = [li * wa + lj]
                    u_coeffs = [1]
                    v_idx = [lj * wb + lk]
                    v_coeffs = [1]
                    w_idx = [li * wb + lk]
                    w_coeffs = [1]
                    terms.append((u_idx, u_coeffs, v_idx, v_coeffs, w_idx, w_coeffs))
    
    return terms


def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-6):
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
    
    if time.time() > DEADLINE:
        # Fallback to schoolbook immediately if deadline passed
        n, m, p = 5, 5, 5
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

    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Split 5 into 2+2+1
    splits = [0, 2, 4, 5]
    sizes = [splits[i+1] - splits[i] for i in range(3)]  # [2, 2, 1]
    
    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ha = sizes[i]  # A row block height
                wa = sizes[j]  # A col block width = B row block height
                hb = sizes[j]  # B row block height
                wb = sizes[k]  # B col block width
                
                a_row_start = splits[i]
                a_col_start = splits[j]
                b_row_start = splits[j]
                b_col_start = splits[k]
                c_row_start = splits[i]
                c_col_start = splits[k]
                
                local_terms = _block_decomp(ha, wa, hb, wb)
                
                for u_local, u_coeffs, v_local, v_coeffs, w_local, w_coeffs in local_terms:
                    u_dict = {}
                    for li, coeff in zip(u_local, u_coeffs):
                        if coeff != 0:
                            li_row = li // wa
                            li_col = li % wa
                            global_u = (a_row_start + li_row) * m + (a_col_start + li_col)
                            u_dict[global_u] = u_dict.get(global_u, 0) + coeff
                    
                    v_dict = {}
                    for li, coeff in zip(v_local, v_coeffs):
                        if coeff != 0:
                            li_row = li // wb
                            li_col = li % wb
                            global_v = (b_row_start + li_row) * p + (b_col_start + li_col)
                            v_dict[global_v] = v_dict.get(global_v, 0) + coeff
                    
                    w_dict = {}
                    for li, coeff in zip(w_local, w_coeffs):
                        if coeff != 0:
                            li_row = li // wb
                            li_col = li % wb
                            global_w = (c_col_start + li_col) * n + (c_row_start + li_row)
                            w_dict[global_w] = w_dict.get(global_w, 0) + coeff
                    
                    terms.append((u_dict, v_dict, w_dict))
    
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
    
    # Verify
    if time.time() > DEADLINE:
         # Fallback to schoolbook
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

    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
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
