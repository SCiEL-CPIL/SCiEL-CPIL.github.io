import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def verify(U, V, W, atol=1e-9):
        """Verify tensor decomposition exactly."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] += u[i,j] * v[j,k] * w[k,i]
        
        Texp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Texp[i*m+j, j*p+k, k*n+i] = 1.0
        
        return np.allclose(T, Texp, atol=atol)

    def build_rank_36():
        """
        Construct rank-36 decomposition:
        4 Strassen 2x2x2 blocks for B columns 0-3 (4*7 = 28 terms)
        8 schoolbook terms for B column 4 (2*4*1 = 8 terms)
        Total: 36 terms
        """
        # Strassen 2x2x2 factor matrices (rank 7)
        # Indices: U[i,j], V[j,k], W[k,i] for 2x2 matrices
        strassen_U = [
            [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
            [1,1,0,0],[-1,0,1,0],[0,1,0,-1]
        ]
        strassen_V = [
            [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
            [0,0,0,1],[1,1,0,0],[0,0,1,1]
        ]
        strassen_W = [
            [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
            [-1,0,1,0],[0,0,0,1],[1,0,0,0]
        ]
        
        rank = 4*7 + 8  # 36
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        # Four Strassen blocks: j-blocks (0, 2) x k-blocks (0, 2)
        for j0 in (0, 2):
            for k0 in (0, 2):
                for r in range(7):
                    # Fill U: 2x2 block at rows 0,1 and cols j0,j0+1
                    for li in range(2):
                        for lj in range(2):
                            U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                    # Fill V: 2x2 block at rows j0,j0+1 and cols k0,k0+1
                    for lj in range(2):
                        for lk in range(2):
                            V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                    # Fill W: 2x2 block at rows k0,k0+1 and cols 0,1
                    for lk in range(2):
                        for li in range(2):
                            W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                    col += 1
        
        # Schoolbook for B column 4: 2*4*1 = 8 terms
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j*p+4][col] = 1
                W[4*n+i][col] = 1
                col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W

    # Build the rank-36 decomposition
    U, V, W = build_rank_36()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook rank 40 (should not happen)
        rank_schoolbook = n * m * p
        U_school = [[0]*rank_schoolbook for _ in range(n*m)]
        V_school = [[0]*rank_schoolbook for _ in range(m*p)]
        W_school = [[0]*rank_schoolbook for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
