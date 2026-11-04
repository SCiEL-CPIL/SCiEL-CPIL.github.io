"""
Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
Uses a greedy tiling of 2x2x2 Strassen blocks to reduce rank from 75 to 71.
"""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_elements = n * m * p
    
    # We will build the matrices dynamically.
    # Max rank is total_elements (schoolbook).
    # We'll track the current column index.
    U = [[0] * total_elements for _ in range(n * m)]
    V = [[0] * total_elements for _ in range(m * p)]
    W = [[0] * total_elements for _ in range(p * n)]
    
    col = 0
    covered = set()
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Local indices
        i1, i2 = i0, i0 + 1
        j1, j2 = j0, j0 + 1
        k1, k2 = k0, k0 + 1
        
        # P1 = (A11+A22)(B11+B22) -> C11(+), C22(+)
        U[i1*m+j1][col] = 1; U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1; V[j2*p+k2][col] = 1
        W[k1*n+i1][col] = 1; W[k2*n+i2][col] = 1
        covered.update([(i1, j1, k1), (i1, j1, k2), (i1, j2, k1), (i1, j2, k2),
                        (i2, j1, k1), (i2, j1, k2), (i2, j2, k1), (i2, j2, k2)])
        col += 1
        
        # P2 = (A21+A22)B11 -> C21(+), C22(+)
        U[i2*m+j1][col] = 1; U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1
        W[k1*n+i2][col] = 1; W[k2*n+i2][col] = 1
        col += 1
        
        # P3 = A11(B12-B22) -> C12(+), C22(+)
        U[i1*m+j1][col] = 1
        V[j1*p+k2][col] = 1; V[j2*p+k2][col] = -1
        W[k2*n+i1][col] = 1; W[k2*n+i2][col] = 1
        col += 1
        
        # P4 = A22(B21-B11) -> C11(+), C21(+)
        U[i2*m+j2][col] = 1
        V[j2*p+k1][col] = 1; V[j1*p+k1][col] = -1
        W[k1*n+i1][col] = 1; W[k1*n+i2][col] = 1
        col += 1
        
        # P5 = (A11+A12)B22 -> C11(-), C12(+), C22(-)
        U[i1*m+j1][col] = 1; U[i1*m+j2][col] = 1
        V[j2*p+k2][col] = 1
        W[k1*n+i1][col] = -1; W[k2*n+i1][col] = 1; W[k2*n+i2][col] = -1
        col += 1
        
        # P6 = (A21-A11)(B11+B12) -> C22(+)
        U[i2*m+j1][col] = 1; U[i1*m+j1][col] = -1
        V[j1*p+k1][col] = 1; V[j1*p+k2][col] = 1
        W[k2*n+i2][col] = 1
        col += 1
        
        # P7 = (A12-A22)(B21+B22) -> C11(+)
        U[i1*m+j2][col] = 1; U[i2*m+j2][col] = -1
        V[j2*p+k1][col] = 1; V[j2*p+k2][col] = 1
        W[k1*n+i1][col] = 1
        col += 1

    # 1. Place Strassen blocks
    # We fit 2x2 blocks in the 3x5x5 grid.
    # i: 0-1 (one block), 2 (leftover)
    # j: 0-1, 2-3 (two blocks), 4 (leftover)
    # k: 0-1, 2-3 (two blocks), 4 (leftover)
    
    block_offsets = []
    for i0 in range(0, n, 2):
        if i0 + 1 < n:
            for j0 in range(0, m, 2):
                if j0 + 1 < m:
                    for k0 in range(0, p, 2):
                        if k0 + 1 < p:
                            block_offsets.append((i0, j0, k0))
    
    for offset in block_offsets:
        add_strassen(*offset)
        
    # 2. Fill remaining with Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)

    # Trim matrices to actual rank
    U_final = [[val for val in row] for row in U][:n*m][:, :col] # Python slicing is tricky with lists of lists directly, 
    # Let's just slice properly
    U_res = [row[:col] for row in U]
    V_res = [row[:col] for row in V]
    W_res = [row[:col] for row in W]
    
    return {"U": U_res, "V": V_res, "W": W_res}
# EVOLVE-BLOCK-END
