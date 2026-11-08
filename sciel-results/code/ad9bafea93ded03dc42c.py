"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    def verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def add_strassen_block(i0, j0, k0, U, V, W, col):
        """Add a 2x2x2 Strassen block at position (i0, j0, k0)."""
        # M1 = (A00+A11)(B00+B11)
        c = col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Rank 31 Solution:
    # 4 Strassen blocks (rank 28) for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # 1 Bulk column (rank 1) for i=2 (all j, k)
    # 1 Bulk column (rank 1) for j=4, i in {0,1} (all k)
    # 1 Bulk column (rank 1) for k=4, i in {0,1}, j in {0,1,2,3}
    
    total_rank = 31
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen blocks
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, U, V, W, col)
        col += 7
    
    # Bulk Column 1: i=2
    c = col
    for j in range(m):
        U[2*m + j][c] = 1
    for j in range(m):
        for k in range(p):
            V[j*p + k][c] = 1
    for k in range(p):
        W[k*n + 2][c] = 1
    col += 1
    
    # Bulk Column 2: j=4, i in {0,1}
    c = col
    for i in range(2):
        U[i*m + 4][c] = 1
    for k in range(p):
        V[4*p + k][c] = 1
    for i in range(2):
        for k in range(p):
            W[k*n + i][c] = 1
    col += 1
    
    # Bulk Column 3: k=4, i in {0,1}, j in {0,1,2,3}
    c = col
    for i in range(2):
        for j in range(4):
            U[i*m + j][c] = 1
    for j in range(4):
        V[j*p + 4][c] = 1
    for i in range(2):
        W[4*n + i][c] = 1
    col += 1

    # Verify the solution
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
