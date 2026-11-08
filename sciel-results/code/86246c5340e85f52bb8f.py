import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Verified Strassen decomposition
        # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (A21+A22)B11 -> C21 += P2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        
        # P3 = A11(B12-B22) -> C12 += P3, C22 -= P3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = -1
        
        # P4 = A22(B21-B11) -> C11 -= P4, C21 += P4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1
        
        # P5 = (A11+A12)B22 -> C11 += P5, C12 += P5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[2][4] = 1
        
        # P6 = (A21-A11)(B11+B12) -> C21 += P6, C22 -= P6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[1][5] = 1; W[3][5] = -1
        
        # P7 = (A12-A22)(B21+B22) -> C22 += P7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[3][6] = 1
        
        return U, V, W
    
    def verify_decomp(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if val != 1:
                        return False
        return True
    
    # Cache for get_decomp
    decomp_cache = {}
    
    def get_decomp(n, m, p):
        """Returns U, V, W as lists of lists for n x m x p decomposition."""
        key = (n, m, p)
        if key in decomp_cache:
            return decomp_cache[key]
        
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            decomp_cache[key] = (U, V, W)
            return U, V, W
        
        # Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            U, V, W = strassen_2x2x2()
            if verify_decomp(U, V, W, 2, 2, 2):
                decomp_cache[key] = (U, V, W)
                return U, V, W
        
        # Schoolbook fallback
        best_rank = n * m * p
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
        
        # Try splitting m
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m1):
                            new_U[i * m + j] = U1[i * m1 + j]
                        for j in range(m2):
                            new_U[i * m + m1 + j] = U2[i * m2 + j]
                            
                    for j in range(m1):
                        for k in range(p):
                            new_V[j * p + k] = V1[j * p + k]
                    for j in range(m2):
                        for k in range(p):
                            new_V[(m1 + j) * p + k] = V2[j * p + k]
                            
                    for k in range(p):
                        for i in range(n):
                            new_W[k * n + i] = [w1 + w2 for w1, w2 in zip(W1[k * n + i], W2[k * n + i])]
                    
                    if verify_decomp(new_U, new_V, new_W, n, m, p):
                        best_rank = r
                        best_U, best_V, best_W = new_U, new_V, new_W

        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
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
                            new_W[k * n + i] = W1[k * n1 + i]
                        for i in range(n2):
                            new_W[k * n + n1 + i] = W2[k * n2 + i]
                            
                    if verify_decomp(new_U, new_V, new_W, n, m, p):
                        best_rank = r
                        best_U, best_V, best_W = new_U, new_V, new_W

        # Try splitting p
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m):
                            new_U[i * m + j] = [u1 + u2 for u1, u2 in zip(U1[i * m + j], U2[i * m + j])]
                            
                    for j in range(m):
                        for k in range(p1):
                            new_V[j * p + k] = V1[j * p1 + k]
                        for k in range(p2):
                            new_V[j * p + p1 + k] = V2[j * p2 + k]
                            
                    for k in range(p1):
                        for i in range(n):
                            new_W[k * n + i] = W1[k * n + i]
                    for k in range(p2):
                        for i in range(n):
                            new_W[(p1 + k) * n + i] = W2[k * n + i]
                            
                    if verify_decomp(new_U, new_V, new_W, n, m, p):
                        best_rank = r
                        best_U, best_V, best_W = new_U, new_V, new_W

        decomp_cache[key] = (best_U, best_V, best_W)
        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)
    
    # Final verification
    if not verify_decomp(U, V, W, n, m, p):
        # Fallback to schoolbook
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
