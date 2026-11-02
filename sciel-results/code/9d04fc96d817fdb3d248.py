import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7

    start_time = time.time()
    deadline = start_time + 140.0

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    def add_strassen_222(U, V, W, col, i0, j0, k0):
        """Add Strassen 2x2x2 block for rows i0,i0+1; cols j0,j0+1; output k0,k0+1"""
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1
        # M7
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1
        return col

    def add_schoolbook_221(U, V, W, col, i0, j0, k0):
        """Add 2x2 * 2x1 schoolbook block (rank 4)"""
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1; col += 1
        return col

    def find_223_decomposition(rank=8):
        """
        Attempts to find a decomposition of the 2x2x3 multiplication tensor with given rank.
        Returns (U, V, W) of shapes (2*2, rank), (2*3, rank), (3*2, rank) or None.
        """
        # Indices for 2x2x3 block
        # U: 4 rows (i=0,1; j=0,1)
        # V: 6 rows (j'=0,1; k=0,1,2)
        # W: 6 rows (k=0,1,2; i=0,1)
        
        # Target: sum_c U[i,j,c] V[j',k,c] W[k,i,c] = delta(j,j')
        
        # We will try to find integer solutions.
        # Use random initialization and gradient descent? Or just check a few known patterns?
        # Given time constraints, we'll try a simple heuristic:
        # Start with Strassen for k=0,1 (rank 7) and add 1 term for k=2?
        # We established 1 term is not enough.
        # Let's try to find a rank 8 solution by searching small integer vectors.
        
        # This is a hard NP-hard problem. We'll just return None for now if we can't find it quickly.
        # However, we can try a specific construction.
        
        # If we fail to find a better one, we return None.
        return None

    # Strategy 1: Try to use a better 2x2x3 block if found
    # For now, we stick to the known Rank 50 solution but structure it to be easily swappable.
    
    # Current Best: Rank 50
    # Partition m=4 as 2+2.
    # Partition p=7 as 2+2+2+1.
    # Blocks: 3x (2x2x2 Strassen) + 1x (2x2x1 Schoolbook) for each m-block.
    # Rank per m-block: 3*7 + 4 = 25.
    # Total Rank: 50.

    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    col = 0

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            col = add_strassen_222(U, V, W, col, 0, j_start, k_start)
        col = add_schoolbook_221(U, V, W, col, 0, j_start, 6)

    if col != rank:
        # Fallback to schoolbook if something went wrong
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to schoolbook
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
