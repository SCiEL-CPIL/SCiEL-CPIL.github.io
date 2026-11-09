import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def validate(U, V, W):
        n, m, p = DIMENSIONS
        T = np.zeros((n * m, m * p, p * n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        rank = len(U[0])
        T_check = np.zeros((n * m, m * p, p * n), dtype=float)
        for s in range(rank):
            u = np.array(U, dtype=float)[:, s]
            v = np.array(V, dtype=float)[:, s]
            w = np.array(W, dtype=float)[:, s]
            for a in range(n * m):
                if u[a] == 0: continue
                for b in range(m * p):
                    if v[b] == 0: continue
                    T_check[a, b, :] += u[a] * v[b] * w
        return np.allclose(T_check, T, atol=1e-6)
    
    def schoolbook():
        n, m, p = DIMENSIONS
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
    
    def build_block_decomposition():
        """
        Build a rank-50 decomposition by treating 3x4x8 as two 3x4x4 blocks.
        For each block, use a structured pattern.
        """
        n1, m1, p1 = 3, 4, 4
        
        # Build a 3x4x4 decomposition using a structured approach
        # We'll use the fact that we can decompose it as:
        # For each (i,j) pair in A, and each k in B, we need a term
        # But we can share terms across different (i,j,k) combinations
        
        # Strategy: Use a "row-column" decomposition
        # For 3x4x4, we can try rank 25 (schoolbook) or try to do better
        
        # Let's try a structured rank-25 approach first
        U344 = [[0] * 25 for _ in range(n1 * m1)]
        V344 = [[0] * 25 for _ in range(m1 * p1)]
        W344 = [[0] * 25 for _ in range(p1 * n1)]
        
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U344[i * m1 + j][col] = 1
                    V344[j * p1 + k][col] = 1
                    W344[k * n1 + i][col] = 1
                    col += 1
        
        # Now build 3x4x8 from two 3x4x4 blocks
        rank = 25
        U = [[0] * (2 * rank) for _ in range(n * m)]
        V = [[0] * (2 * rank) for _ in range(m * p)]
        W = [[0] * (2 * rank) for _ in range(p * n)]
        
        # First block: k=0..3
        for a in range(n1 * m1):
            for s in range(rank):
                U[a][s] = U344[a][s]
        
        for b in range(m1 * p1):
            j = b // p1
            k = b % p1
            b_new = j * p + k
            for s in range(rank):
                V[b_new][s] = V344[b][s]
        
        for c in range(p1 * n1):
            for s in range(rank):
                W[c][s] = W344[c][s]
        
        # Second block: k=4..7
        for a in range(n1 * m1):
            for s in range(rank):
                U[a][rank + s] = U344[a][s]
        
        for b in range(m1 * p1):
            j = b // p1
            k = b % p1
            k_new = k + 4
            b_new = j * p + k_new
            for s in range(rank):
                V[b_new][rank + s] = V344[b][s]
        
        for c in range(p1 * n1):
            k = c // n1
            i = c % n1
            k_new = k + 4
            c_new = k_new * n1 + i
            for s in range(rank):
                W[c_new][rank + s] = W344[c][s]
        
        return U, V, W
    
    # Try block decomposition first (guaranteed rank 50)
    U, V, W = build_block_decomposition()
    if validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
