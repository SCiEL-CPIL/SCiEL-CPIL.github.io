def construct():
    # Rank-49 decomposition via Kronecker product of Strassen's 2x2 rank-7 decomposition.
    # The 2x2 Strassen matrices (4x7)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ]
    
    # Kronecker product to get 4x4 (16x49)
    def kron(A, B):
        m, n = len(A), len(A[0])
        p, q = len(B), len(B[0])
        C = [[0] * (n * q) for _ in range(m * p)]
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    for l in range(q):
                        C[i * p + k][j * q + l] = A[i][j] * B[k][l]
        return C

    U = kron(U2, U2)
    V = kron(V2, V2)
    W = kron(W2, W2)
    
    return {"U": U, "V": V, "W": W}
