import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify(Uv, Vv, Wv, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uv[i*m+j], Vv[j*p+k] * Wv[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def add_strassen_2x2x2(U, V, W, i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        c = col_offset + 1
        U[(i0+1)*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += -1

        c = col_offset + 2
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        c = col_offset + 3
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + (i0+1), c] += 1

        c = col_offset + 4
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += -1
        W[(k0+1)*n + i0, c] += 1

        c = col_offset + 5
        U[(i0+1)*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        c = col_offset + 6
        U[i0*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+1), c] += -1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1

    # Strategy: Use 3x3x3 rank-23 subcube + 2 extra Strassen blocks
    # 3x3x3 subcube: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # Extra block 1: i in {0,1}, j in {3,4}, k in {0,1}
    # Extra block 2: i in {0,1}, j in {0,1}, k in {3,4}
    # Remaining: schoolbook
    
    # First, let's try the rank-71 baseline
    blocks_71 = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    def build_from_blocks(blocks):
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered[i0+di][j0+dj][k0+dk] = True
        
        remaining = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered[i][j][k])
        total_rank = 7 * len(blocks) + remaining
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        for i0, j0, k0 in blocks:
            add_strassen_2x2x2(U, V, W, i0, j0, k0, col)
            col += 7
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j, col] += 1
                        V[j*p+k, col] += 1
                        W[k*n+i, col] += 1
                        col += 1
        
        if verify(U, V, W):
            return U, V, W, total_rank
        return None
    
    # Try rank-71 baseline
    result_71 = build_from_blocks(blocks_71)
    if result_71 is not None:
        U, V, W, rank = result_71
        if time.time() > deadline - 10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-69: 3x3x3 rank-23 + 2 extra Strassen blocks
    # We need a rank-23 decomposition for the 3x3x3 subcube
    # For now, use 1 Strassen block (rank 7) + 20 schoolbook = rank 27 for the 3x3x3 subcube
    # Then + 2 extra Strassen blocks (rank 7 each) + remaining schoolbook
    # This gives: 27 + 7 + 7 + (75 - 27 - 8 - 8) = 27 + 14 + 32 = 73, which is worse
    
    # The key is the rank-23 decomposition. Let's try to find it via local search.
    # For now, return the rank-71 result.
    if result_71 is not None:
        return {"U": result_71[0].tolist(), "V": result_71[1].tolist(), "W": result_71[2].tolist()}
    
    # Fallback: schoolbook
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
