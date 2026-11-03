"""
Program for matrix multiplication tensor decomposition.
Task: Find a rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication.
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m x p using a recursive splitting strategy to minimize rank.
    """
    n, m, p = DIMENSIONS
    
    def get_decomp(n, m, p):
        # Base case: schoolbook
        if m == 1 or p == 1 or n == 1:
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
            
        # Try splitting m, p, n
        best_rank = n * m * p
        best_decomp = None
        
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            r = r1 + r2
            if r < best_rank:
                U = [row1 + row2 for row1, row2 in zip(U1, U2)]
                W = [row1 + row2 for row1, row2 in zip(W1, W2)]
                V = [[0] * r for _ in range(m * p)]
                for j in range(m1):
                    for k in range(p):
                        for c in range(r1):
                            V[j * p + k][c] = V1[j * p + k][c]
                for j in range(m2):
                    for k in range(p):
                        for c in range(r2):
                            V[(j + m1) * p + k][r1 + c] = V2[j * p + k][c]
                best_rank = r
                best_decomp = (U, V, W)
                
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            r1 = len(U1[0])
            r2 = len(U2[0])
            r = r1 + r2
            if r < best_rank:
                U = [row1 + row2 for row1, row2 in zip(U1, U2)]
                V = [row1 + row2 for row1, row2 in zip(V1, V2)]
                W = [[0] * r for _ in range(p * n)]
                for k in range(p1):
                    for i in range(n):
                        for c in range(r1):
                            W[k * n + i][c] = W1[k * n + i][c]
                for k in range(p2):
                    for i in range(n):
                        for c in range(r2):
                            W[(k + p1) * n + i][r1 + c] = W2[k * n + i][c]
                best_rank = r
                best_decomp = (U, V, W)
                
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            r = r1 + r2
            if r < best_rank:
                V = [row1 + row2 for row1, row2 in zip(V1, V2)]
                W = [row1 + row2 for row1, row2 in zip(W1, W2)]
                U = [[0] * r for _ in range(n * m)]
                for i in range(n1):
                    for j in range(m):
                        for c in range(r1):
                            U[i * m + j][c] = U1[i * m + j][c]
                for i in range(n2):
                    for j in range(m):
                        for c in range(r2):
                            U[(i + n1) * m + j][r1 + c] = U2[i * m + j][c]
                best_rank = r
                best_decomp = (U, V, W)

        return best_decomp

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
