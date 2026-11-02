    def verify(U, V, W):
        # Check shapes
        if len(U) != n*m or len(V) != m*p or len(W) != p*n:
            return False
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        # Check entries are half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # val should be k/2
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        # Check decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
