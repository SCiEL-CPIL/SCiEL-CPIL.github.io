import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def schoolbook(n, m, p):
        """Return U, V, W for schoolbook decomposition of n x m x p."""
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
        return U, V, W, rank
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r] 
                             for r in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
        """Combine two decompositions split on m dimension."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        new_U = [[0] * r for _ in range(n * (m1 + m2))]
        new_V = [[0] * r for _ in range((m1 + m2) * p)]
        new_W = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m1):
                new_U[i * (m1 + m2) + j] = U1[i * m1 + j]
            for j in range(m2):
                new_U[i * (m1 + m2) + m1 + j] = U2[i * m2 + j]
        
        for j in range(m1):
            for k in range(p):
                new_V[j * p + k] = V1[j * p + k]
        for j in range(m2):
            for k in range(p):
                new_V[(m1 + j) * p + k] = V2[j * p + k]
        
        for k in range(p):
            for i in range(n):
                new_W[k * n + i] = [w1 + w2 for w1, w2 in zip(W1[k * n + i], W2[k * n + i])]
        
        return new_U, new_V, new_W, r
    
    def combine_n_split(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
        """Combine two decompositions split on n dimension."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        new_U = [[0] * r for _ in range((n1 + n2) * m)]
        new_V = [[0] * r for _ in range(m * p)]
        new_W = [[0] * r for _ in range(p * (n1 + n2))]
        
        for i in range(n1):
            for j in range(m):
                new_U[i * m + j] = U1[i * m + j]
        for i in range(n2):
            for j in range(m):
                new_U[(n1 + i) * m + j] = U2[i * m + j]
        
        for j in range(m):
            for k in range(p):
                new_V[j * p + k] = [v1 + v2 for v1, v2 in zip(V1[j * p + k], V2[j * p + k])]
        
        for k in range(p):
            for i in range(n1):
                new_W[k * (n1 + n2) + i] = W1[k * n1 + i]
            for i in range(n2):
                new_W[k * (n1 + n2) + n1 + i] = W2[k * n2 + i]
        
        return new_U, new_V, new_W, r
    
    def combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """Combine two decompositions split on p dimension."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        new_U = [[0] * r for _ in range(n * m)]
        new_V = [[0] * r for _ in range(m * (p1 + p2))]
        new_W = [[0] * r for _ in range((p1 + p2) * n)]
        
        for i in range(n):
            for j in range(m):
                new_U[i * m + j] = [u1 + u2 for u1, u2 in zip(U1[i * m + j], U2[i * m + j])]
        
        for j in range(m):
            for k in range(p1):
                new_V[j * (p1 + p2) + k] = V1[j * p1 + k]
            for k in range(p2):
                new_V[j * (p1 + p2) + p1 + k] = V2[j * p2 + k]
        
        for k in range(p1):
            for i in range(n):
                new_W[k * n + i] = W1[k * n + i]
        for k in range(p2):
            for i in range(n):
                new_W[(p1 + k) * n + i] = W2[k * n + i]
        
        return new_U, new_V, new_W, r
    
    # Start with schoolbook
    best_U, best_V, best_W, best_rank = schoolbook(n, m, p)
    
    # Try all single splits on m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1, r1 = schoolbook(n, m1, p)
        U2, V2, W2, r2 = schoolbook(n, m2, p)
        r = r1 + r2
        if r < best_rank:
            new_U, new_V, new_W, r = combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
            if verify_decomp(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W, best_rank = new_U, new_V, new_W, r
    
    # Try all single splits on n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1, r1 = schoolbook(n1, m, p)
        U2, V2, W2, r2 = schoolbook(n2, m, p)
        r = r1 + r2
        if r < best_rank:
            new_U, new_V, new_W, r = combine_n_split(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
            if verify_decomp(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W, best_rank = new_U, new_V, new_W, r
    
    # Try all single splits on p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1, r1 = schoolbook(n, m, p1)
        U2, V2, W2, r2 = schoolbook(n, m, p2)
        r = r1 + r2
        if r < best_rank:
            new_U, new_V, new_W, r = combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
            if verify_decomp(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W, best_rank = new_U, new_V, new_W, r
    
    # Try nested splits: split m, then split one part further
    for m1 in range(1, m):
        m2 = m - m1
        for m1a in range(1, m1):
            m1b = m1 - m1a
            U1a, V1a, W1a, r1a = schoolbook(n, m1a, p)
            U1b, V1b, W1b, r1b = schoolbook(n, m1b, p)
            U1, V1, W1, r1 = combine_m_split(U1a, V1a, W1a, U1b, V1b, W1b, n, m1a, m1b, p)
            U2, V2, W2, r2 = schoolbook(n, m2, p)
            r = r1 + r2
            if r < best_rank:
                new_U, new_V, new_W, r = combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
                if verify_decomp(new_U, new_V, new_W, n, m, p):
                    best_U, best_V, best_W, best_rank = new_U, new_V, new_W, r
    
    # Try nested splits: split p, then split one part further
    for p1 in range(1, p):
        p2 = p - p1
        for p1a in range(1, p1):
            p1b = p1 - p1a
            U1a, V1a, W1a, r1a = schoolbook(n, m, p1a)
            U1b, V1b, W1b, r1b = schoolbook(n, m, p1b)
            U1, V1, W1, r1 = combine_p_split(U1a, V1a, W1a, U1b, V1b, W1b, n, m, p1a, p1b)
            U2, V2, W2, r2 = schoolbook(n, m, p2)
            r = r1 + r2
            if r < best_rank:
                new_U, new_V, new_W, r = combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
                if verify_decomp(new_U, new_V, new_W, n, m, p):
                    best_U, best_V, best_W, best_rank = new_U, new_V, new_W, r
    
    if time.time() > deadline:
        # Fallback to schoolbook
        best_U, best_V, best_W, best_rank = schoolbook(n, m, p)

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
