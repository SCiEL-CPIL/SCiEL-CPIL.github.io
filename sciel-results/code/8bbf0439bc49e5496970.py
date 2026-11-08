import time

_DIMENSIONS = (3, 4, 7)
_DEADLINE = 135.0

def _build_factors():
    n, m, p = _DIMENSIONS
    # n=3, m=4, p=7
    # Rank estimate: 74
    rank = 74
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
        # M1
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1
        # M2
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1
        # M3
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1
        # M4
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1
        # M5
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1
        # M6
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1
        # M7
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1
        return col_start + 7

    def add_schoolbook_2x2x1(col_start, i0, i1, j0, j1, k0):
        """Add 2 schoolbook terms for a 2x2x1 block."""
        U[i0 * m + j0][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j1 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        return col_start + 2

    def add_schoolbook_1x2x2(col_start, i0, j0, j1, k0, k1):
        """Add 4 schoolbook terms for a 1x2x2 block."""
        # y_k0 = a_j0 b_j0k0 + a_j1 b_j1k0
        # y_k1 = a_j0 b_j0k1 + a_j1 b_j1k1
        U[i0 * m + j0][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        
        U[i0 * m + j1][col_start + 1] = 1
        V[j1 * p + k0][col_start + 1] = 1
        W[k0 * n + i0][col_start + 1] = 1
        
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        W[k1 * n + i0][col_start + 2] = 1
        
        U[i0 * m + j1][col_start + 3] = 1
        V[j1 * p + k1][col_start + 3] = 1
        W[k1 * n + i0][col_start + 3] = 1
        return col_start + 4

    def add_schoolbook_1x2x1(col_start, i0, j0, j1, k0):
        """Add 2 schoolbook terms for a 1x2x1 block."""
        U[i0 * m + j0][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        
        U[i0 * m + j1][col_start + 1] = 1
        V[j1 * p + k0][col_start + 1] = 1
        W[k0 * n + i0][col_start + 1] = 1
        return col_start + 2

    # Process A1 (rows 0,1)
    # B1 (cols 0,1)
    # A11 (0,0; 1,1) x B1a (0,0; 1,1)
    col = add_strassen_block(col, 0, 1, 0, 1, 0, 1)
    # A12 (0,2; 1,3) x B1b (2,0; 3,1)
    col = add_strassen_block(col, 0, 1, 2, 3, 0, 1)
    
    # B2 (cols 2,3)
    col = add_strassen_block(col, 0, 1, 0, 1, 2, 3)
    col = add_strassen_block(col, 0, 1, 2, 3, 2, 3)
    
    # B3 (cols 4,5)
    col = add_strassen_block(col, 0, 1, 0, 1, 4, 5)
    col = add_strassen_block(col, 0, 1, 2, 3, 4, 5)
    
    # B4 (col 6)
    # A11 x B4a (rows 0,1)
    col = add_schoolbook_2x2x1(col, 0, 1, 0, 1, 6)
    # A12 x B4b (rows 2,3)
    col = add_schoolbook_2x2x1(col, 0, 1, 2, 3, 6)
    
    # Process A2 (row 2)
    # B1 (cols 0,1)
    # A21 (cols 0,1) x B1a (rows 0,1)
    col = add_schoolbook_1x2x2(col, 2, 0, 1, 0, 1)
    # A22 (cols 2,3) x B1b (rows 2,3)
    col = add_schoolbook_1x2x2(col, 2, 2, 3, 0, 1)
    
    # B2 (cols 2,3)
    col = add_schoolbook_1x2x2(col, 2, 0, 1, 2, 3)
    col = add_schoolbook_1x2x2(col, 2, 2, 3, 2, 3)
    
    # B3 (cols 4,5)
    col = add_schoolbook_1x2x2(col, 2, 0, 1, 4, 5)
    col = add_schoolbook_1x2x2(col, 2, 2, 3, 4, 5)
    
    # B4 (col 6)
    col = add_schoolbook_1x2x1(col, 2, 0, 1, 6)
    col = add_schoolbook_1x2x1(col, 2, 2, 3, 6)
    
    if col != rank:
        raise AssertionError(f"expected rank {rank}, got {col}")
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                total = 0
                for c in range(rank):
                    total += u_row[c] * v_row[c] * w_row[c]
                if total != 1:
                    raise ValueError(
                        f"decomposition failed at ({i},{j},{k}): got {total}, expected 1"
                    )

def construct():
    t0 = time.time()
    n, m, p = _DIMENSIONS
    U, V, W = _build_factors()
    rank = len(U[0])
    _verify(U, V, W, n, m, p, rank)
    if time.time() - t0 > _DEADLINE:
        pass
    return {"U": U, "V": V, "W": W}
