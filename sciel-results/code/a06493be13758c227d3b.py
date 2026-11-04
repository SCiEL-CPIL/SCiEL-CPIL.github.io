"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j, c]*V[j*p+k, c] == W[k*n+i, c] delta structure holds."""
    rank = len(U[0])
    for c in range(rank):
        uc = [row[c] for row in U]
        vc = [row[c] for row in V]
        wc = [row[c] for row in W]
        # The product U[:,c] outer V[:,c] should equal sum over (i,j,k) wc[k*n+i] * e_{i*m+j, j*p+k}
        # i.e., (U[:,c] ⊗ V[:,c])[i*m+j, j*p+k] should equal W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = wc[k * n + i]
                    actual = uc[i * m + j] * vc[j * p + k]
                    if abs(expected - actual) > 1e-9:
                        return False
    return True

def _schoolbook_2x3x3():
    """Return (U, V, W) for 2x3x3 schoolbook, rank 9."""
    n, m, p = 2, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(U1, V1, W1, U2, V2, W2):
    """Concatenate two 2x3x3 decompositions into a 2x3x6 decomposition."""
    # U has shape (n*m, rank_block1) and (n*m, rank_block2) -> (n*m, rank1+rank2)
    # V has shape (m*p1, r1) and (m*p2, r2) -> (m*(p1+p2), r1+r2)
    # W has shape (p1*n, r1) and (p2*n, r2) -> ((p1+p2)*n, r1+r2)
    n, m, p1 = 2, 3, 3
    p2 = 3
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 + r2
    
    # U: just concatenate columns
    U = [row1 + row2 for row1, row2 in zip(U1, U2)]
    
    # V: first block has m*p1 rows, second has m*p2 rows
    # Pad to same number of columns (already same rank), then stack rows
    V = []
    for row in V1:
        V.append(row + [0] * r2)
    for row in V2:
        V.append([0] * r1 + row)
    
    # W: first block has p1*n rows, second has p2*n rows
    W = []
    for row in W1:
        W.append(row + [0] * r2)
    for row in W2:
        W.append([0] * r1 + row)
    
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try rank 18: two 2x3x3 schoolbook blocks
    U1, V1, W1 = _schoolbook_2x3x3()
    U2, V2, W2 = _schoolbook_2x3x3()
    U, V, W = _combine_blocks(U1, V1, W1, U2, V2, W2)
    
    # Verify
    if _verify(U, V, W, 2, 3, 6):
        rank = len(U[0])
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
