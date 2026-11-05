import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # 4 Strassen blocks for i in {0,1}
    # 1 Rank-1 term for i=2
    # Schoolbook for the rest
    
    # Strassen blocks:
    # 1: i{0,1}, j{0,1}, k{0,1}
    # 2: i{0,1}, j{2,3}, k{2,3}
    # 3: i{0,1}, j{0,1}, k{4,5}
    # 4: i{0,1}, j{2,3}, k{4,5}
    
    # Rank calculation:
    # 4 * 7 = 28
    # 1 * 1 = 1
    # Schoolbook: 84 - 4*8 - 1*8 = 84 - 32 - 8 = 44?
    # Wait, the Rank-1 term for i=2 covers 8 terms?
    # No, the Rank-1 term for i=2 covers ALL 28 terms for i=2.
    # So we don't need schoolbook for i=2.
    # Schoolbook is only for i in {0,1} not covered by Strassen.
    # Total terms for i in {0,1}: 56.
    # Covered by Strassen: 4 * 8 = 32.
    # Remaining: 56 - 32 = 24.
    # Total Rank = 28 (Strassen) + 1 (Rank-1) + 24 (Schoolbook) = 53.
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(col_start, i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
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

    # Add 4 Strassen blocks
    col = add_strassen_block(col, 0, 0, 0)
    col = add_strassen_block(col, 0, 2, 2)
    col = add_strassen_block(col, 0, 0, 4)
    col = add_strassen_block(col, 0, 2, 4)
    
    # Add Rank-1 term for i=2
    # U[2*m + j][col] = 1 for all j?
    # No, U index is i*m + j.
    # We need U[i=2, j][col] = 1 for all j.
    for j in range(m):
        U[2 * m + j][col] = 1
    for k in range(p):
        V[j * p + k][col] = 1 # Wait, V index is j*p + k.
        # We need V[j, k][col] = 1 for all j, k.
    for j in range(m):
        for k in range(p):
            V[j * p + k][col] = 1
    for i in range(n):
        W[k * n + i][col] = 1 # Wait, W index is k*n + i.
        # We need W[k, i][col] = 1 for all k, i=2.
    for k in range(p):
        W[k * n + 2][col] = 1
    col += 1
    
    # Schoolbook for remaining i in {0,1}
    covered = set()
    for i0 in range(2):
        for j0 in range(m):
            for k0 in range(p):
                if (i0 in (0,1) and j0 in (0,1) and k0 in (0,1)) or \
                   (i0 in (0,1) and j0 in (2,3) and k0 in (2,3)) or \
                   (i0 in (0,1) and j0 in (0,1) and k0 in (4,5)) or \
                   (i0 in (0,1) and j0 in (2,3) and k0 in (4,5)):
                    covered.add((i0, j0, k0))
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1
                
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
