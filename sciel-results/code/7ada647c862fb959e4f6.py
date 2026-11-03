import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110
    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def build_strassen_44():
        """
        Rank-44 decomposition.
        Split m=3 into 1+2.
        Part 1: 2x1 * 1x8 -> 16 schoolbook mults.
        Part 2: 2x2 * 2x8 -> 4 blocks of 2x2 * 2x2 (Strassen, rank 7 each) -> 28 mults.
        Total: 16 + 28 = 44.
        """
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8)
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8)
        for block in range(4):
            b00 = 1 * p + block * 2
            b01 = 1 * p + block * 2 + 1
            b10 = 2 * p + block * 2
            b11 = 2 * p + block * 2 + 1
            
            c00 = (block * 2) * n + 0
            c01 = (block * 2 + 1) * n + 0
            c10 = (block * 2) * n + 1
            c11 = (block * 2 + 1) * n + 1
            
            # Strassen step 1: (a00+a11)(b00+b11)
            U[1][col] = 1
            U[5][col] = 1
            V[b00][col] = 1
            V[b11][col] = 1
            W[c00][col] = 1
            W[c11][col] = 1
            col += 1
            
            # Strassen step 2: (a10+a11)b00
            U[4][col] = 1
            U[5][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1
            W[c11][col] = -1
            col += 1
            
            # Strassen step 3: a00(b01-b11)
            U[1][col] = 1
            V[b01][col] = 1
            V[b11][col] = -1
            W[c01][col] = 1
            W[c11][col] = 1
            col += 1
            
            # Strassen step 4: a11(b10-b00)
            U[5][col] = 1
            V[b10][col] = 1
            V[b00][col] = -1
            W[c00][col] = 1
            W[c10][col] = 1
            col += 1
            
            # Strassen step 5: (a00+a01)b11
            U[1][col] = 1
            U[2][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1
            W[c01][col] = 1
            col += 1
            
            # Strassen step 6: (a10-a00)(b00+b01)
            U[4][col] = 1
            U[1][col] = -1
            V[b00][col] = 1
            V[b01][col] = 1
            W[c11][col] = 1
            col += 1
            
            # Strassen step 7: (a01-a11)(b10+b11)
            U[2][col] = 1
            U[5][col] = -1
            V[b10][col] = 1
            V[b11][col] = 1
            W[c00][col] = 1
            col += 1
        
        return U, V, W

    # Try rank-44 construction first (known to work)
    U, V, W = build_strassen_44()
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook (should not happen)
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
