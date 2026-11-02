import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2x2 base case.
    """
    n, m, p = DIMENSIONS
    
    def get_decomp(n, m, p):
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        # Base case: 2x2x2 (Strassen)
        if n == 2 and m == 2 and p == 2:
            # U shape (4, 7), V shape (4, 7), W shape (4, 7)
            # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
            U = [
                [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
                [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A01)
                [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A10)
                [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A11)
            ]
            V = [
                [1, 1, 0, 0, 0, 1, 0], # j=0, k=0 (B00)
                [0, 0, 1, 0, 0, 1, 0], # j=0, k=1 (B01)
                [0, 0, 0, 1, 0, 0, 1], # j=1, k=0 (B10)
                [1, 0, -1, -1, 1, 0, 1]# j=1, k=1 (B11)
            ]
            W = [
                [1, 0, 0, 1, -1, 0, 1], # k=0, i=0 (C00)
                [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1 (C10)
                [0, 1, 0, 1, 0, 0, 0],  # k=1, i=0 (C01)
                [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1 (C11)
            ]
            return U, V, W

        # Recursive case: Try splitting n, m, or p
        best_rank = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                r1 = len(U1[0])
                r2 = len(U2[0])
                total_r = r1 + r2
                if total_r < best_rank:
                    # Combine
                    U = [[0] * total_r for _ in range(n * m)]
                    V = [[0] * total_r for _ in range(m * p)]
                    W = [[0] * total_r for _ in range(p * n)]
                    
                    # Fill U
                    for i in range(n1):
                        for j in range(m):
                            for c in range(r1):
                                U[i * m + j][c] = U1[i * m + j][c]
                    for i in range(n2):
                        for j in range(m):
                            for c in range(r2):
                                U[(n1 + i) * m + j][r1 + c] = U2[i * m + j][c]
                    
                    # Fill V
                    for j in range(m):
                        for k in range(p):
                            for c in range(r1):
                                V[j * p + k][c] = V1[j * p + k][c]
                            for c in range(r2):
                                V[j * p + k][r1 + c] = V2[j * p + k][c]
                    
                    # Fill W
                    for k in range(p):
                        for i in range(n1):
                            for c in range(r1):
                                W[k * n + i][c] = W1[k * n1 + i][c]
                        for i in range(n2):
                            for c in range(r2):
                                W[k * n + (n1 + i)][r1 + c] = W2[k * n2 + i][c]
                    
                    best_rank = total_r
                    best_U, best_V, best_W = U, V, W

        # Try splitting m
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                r1 = len(U1[0])
                r2 = len(U2[0])
                total_r = r1 + r2
                if total_r < best_rank:
                    U = [[0] * total_r for _ in range(n * m)]
                    V = [[0] * total_r for _ in range(m * p)]
                    W = [[0] * total_r for _ in range(p * n)]
                    
                    # Fill U
                    for i in range(n):
                        for j in range(m1):
                            for c in range(r1):
                                U[i * m + j][c] = U1[i * m1 + j][c]
                        for j in range(m2):
                            for c in range(r2):
                                U[i * m + (m1 + j)][r1 + c] = U2[i * m2 + j][c]
                    
                    # Fill V
                    for j in range(m1):
                        for k in range(p):
                            for c in range(r1):
                                V[j * p + k][c] = V1[j * p + k][c]
                    for j in range(m2):
                        for k in range(p):
                            for c in range(r2):
                                V[(m1 + j) * p + k][r1 + c] = V2[j * p + k][c]
                    
                    # Fill W
                    for k in range(p):
                        for i in range(n):
                            for c in range(r1):
                                W[k * n + i][c] = W1[k * n + i][c]
                            for c in range(r2):
                                W[k * n + i][r1 + c] = W2[k * n + i][c]
                    
                    best_rank = total_r
                    best_U, best_V, best_W = U, V, W

        # Try splitting p
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                r1 = len(U1[0])
                r2 = len(U2[0])
                total_r = r1 + r2
                if total_r < best_rank:
                    U = [[0] * total_r for _ in range(n * m)]
                    V = [[0] * total_r for _ in range(m * p)]
                    W = [[0] * total_r for _ in range(p * n)]
                    
                    # Fill U
                    for i in range(n):
                        for j in range(m):
                            for c in range(r1):
                                U[i * m + j][c] = U1[i * m + j][c]
                            for c in range(r2):
                                U[i * m + j][r1 + c] = U2[i * m + j][c]
                    
                    # Fill V
                    for j in range(m):
                        for k in range(p1):
                            for c in range(r1):
                                V[j * p + k][c] = V1[j * p1 + k][c]
                        for k in range(p2):
                            for c in range(r2):
                                V[j * p + (p1 + k)][r1 + c] = V2[j * p2 + k][c]
                    
                    # Fill W
                    for k in range(p1):
                        for i in range(n):
                            for c in range(r1):
                                W[k * n + i][c] = W1[k * n + i][c]
                    for k in range(p2):
                        for i in range(n):
                            for c in range(r2):
                                W[(p1 + k) * n + i][r1 + c] = W2[k * n + i][c]
                    
                    best_rank = total_r
                    best_U, best_V, best_W = U, V, W

        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
