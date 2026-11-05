import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank_mp():
        """
        Decomposition with rank n*p.
        Exploits the structure where n is small.
        For each i in [0, n) and k in [0, p), we have a term.
        """
        rank = n * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for k in range(p):
                # U[i, j, col] = 1 for all j
                for j in range(m):
                    U[i * m + j, col] = 1.0
                # V[j, k, col] = 1 for all j
                for j in range(m):
                    V[j * p + k, col] = 1.0
                # W[k, i, col] = 1
                W[k * n + i, col] = 1.0
                col += 1
                
        assert col == rank
        return U, V, W

    # Get the rank-14 decomposition
    U14, V14, W14 = get_rank_mp()

    # Verify it's correct
    if not verify_decomp(U14, V14, W14):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
# EVOLVE-BLOCK-END
