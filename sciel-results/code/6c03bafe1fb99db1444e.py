"""Matrix multiplication 3x4x8 with rank-88 decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # U: 4x7, V: 4x7, W: 4x7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]   # d
    ], dtype=np.int64)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1]   # h
    ], dtype=np.int64)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=np.int64)
    
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    # A = [a, b], B = [[e, f], [g, h]], C = [ae+bg, af+bh]
    # P1 = ae, P2 = bg, P3 = af, P4 = bh
    U = np.array([
        [1, 0, 1, 0],  # a
        [0, 1, 0, 1]   # b
    ], dtype=np.int64)
    
    V = np.array([
        [1, 0, 0, 0],  # e
        [0, 1, 0, 0],  # g
        [0, 0, 1, 0],  # f
        [0, 0, 0, 1]   # h
    ], dtype=np.int64)
    
    W = np.array([
        [1, 1, 0, 0],  # C0
        [0, 0, 1, 1]   # C1
    ], dtype=np.int64)
    
    return U, V, W


def _build_decomposition():
    """Build the rank-88 decomposition."""
    n, m, p = 3, 4, 8
    total_rank = 8 * (7 + 4) # 8 j/k pairs, each with Strassen (7) + School (4)
    
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    # j-blocks: {0,1}, {2,3}
    j_blocks = [(0, 1), (2, 3)]
    # k-blocks: {0,1}, {2,3}, {4,5}, {6,7}
    k_blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    col_offset = 0
    for j0, j1 in j_blocks:
        for k0, k1 in k_blocks:
            # 1. Strassen block for i in {0, 1}
            # U rows: 0*m+j0, 0*m+j1, 1*m+j0, 1*m+j1
            u_rows_str = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            # V rows: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
            v_rows_str = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            # W rows: k0*n+0, k0*n+1, k1*n+0, k1*n+1
            w_rows_str = [k0 * n + 0, k0 * n + 1, k1 * n + 0, k1 * n + 1]
            
            U[u_rows_str, col_offset:col_offset+7] += U2
            V[v_rows_str, col_offset:col_offset+7] += V2
            W[w_rows_str, col_offset:col_offset+7] += W2
            col_offset += 7
            
            # 2. Schoolbook block for i=2
            # U rows: 2*m+j0, 2*m+j1
            u_rows_sch = [2 * m + j0, 2 * m + j1]
            # V rows: same as strassen
            v_rows_sch = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            # W rows: k0*n+2, k1*n+2
            w_rows_sch = [k0 * n + 2, k1 * n + 2]
            
            U[u_rows_sch, col_offset:col_offset+4] += U1
            V[v_rows_sch, col_offset:col_offset+4] += V1
            W[w_rows_sch, col_offset:col_offset+4] += W1
            col_offset += 4
            
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.sum(u_row * v_row * w_row)
                if val != 1:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 140  # safety margin
    
    result = _build_decomposition()
    n, m, p = DIMENSIONS
    
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook rank 96
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    # Convert to lists of lists
    result["U"] = result["U"].tolist()
    result["V"] = result["V"].tolist()
    result["W"] = result["W"].tolist()
    
    return result
