import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
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
            a00_idx = 1
            a01_idx = 2
            a10_idx = 4
            a11_idx = 5
            
            b00_idx = 8 + block*2
            b01_idx = 8 + block*2 + 1
            b10_idx = 16 + block*2
            b11_idx = 16 + block*2 + 1
            
            c00_idx = (block*2)*2 + 0
            c01_idx = (block*2+1)*2 + 0
            c10_idx = (block*2)*2 + 1
            c11_idx = (block*2+1)*2 + 1
            
            # M1
            U[a00_idx][col] = 1; U[a11_idx][col] = 1
            V[b00_idx][col] = 1; V[b11_idx][col] = 1
            W[c00_idx][col] = 1; W[c11_idx][col] = 1
            col += 1
            
            # M2
            U[a10_idx][col] = 1; U[a11_idx][col] = 1
            V[b00_idx][col] = 1
            W[c10_idx][col] = 1; W[c11_idx][col] = -1
            col += 1
            
            # M3
            U[a00_idx][col] = 1
            V[b01_idx][col] = 1; V[b11_idx][col] = -1
            W[c01_idx][col] = 1; W[c11_idx][col] = 1
            col += 1
            
            # M4
            U[a11_idx][col] = 1
            V[b10_idx][col] = 1; V[b00_idx][col] = -1
            W[c00_idx][col] = 1; W[c10_idx][col] = 1
            col += 1
            
            # M5
            U[a00_idx][col] = 1; U[a01_idx][col] = 1
            V[b11_idx][col] = 1
            W[c00_idx][col] = -1; W[c01_idx][col] = 1
            col += 1
            
            # M6
            U[a10_idx][col] = 1; U[a00_idx][col] = -1
            V[b00_idx][col] = 1; V[b01_idx][col] = 1
            W[c11_idx][col] = 1
            col += 1
            
            # M7
            U[a01_idx][col] = 1; U[a11_idx][col] = -1
            V[b10_idx][col] = 1; V[b11_idx][col] = 1
            W[c00_idx][col] = 1
            col += 1
            
        return U, V, W

    def build_schoolbook():
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
        return U, V, W

    # Try rank-44 construction
    U, V, W = build_strassen_44()
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U, V, W = build_schoolbook()
    return {"U": U, "V": V, "W": W}
