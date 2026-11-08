import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    def get_decomp(n, m, p):
        # Returns U, V, W for M_{n,m,p}
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Base cases
        if m == 1:
            rank = n * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i * m][col] = 1
                    V[k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
            return U, V, W, rank

        if p == 1:
            rank = n * m
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = 1
                    V[j * p][col] = 1
                    W[i][col] = 1
                    col += 1
            return U, V, W, rank

        if n == 1:
            rank = m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for j in range(m):
                for k in range(p):
                    U[j][col] = 1
                    V[j * p + k][col] = 1
                    W[k][col] = 1
                    col += 1
            return U, V, W, rank

        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            
            # M1
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # M2
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1
            
            # M3
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[1][2] = 1
            
            # M4
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1
            
            # M5
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[1][4] = 1
            
            # M6
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # M7
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[3][6] = 1
            
            return U, V, W, rank

        best_rank = float('inf')
        best_decomp = None
        best_split_type = None
        best_split_val = None

        # Try splitting m
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1, r1 = get_decomp(n, m1, p)
                U2, V2, W2, r2 = get_decomp(n, m2, p)
                r = r1 + r2
                if r < best_rank:
                    best_rank = r
                    best_decomp = (U1, V1, W1, U2, V2, W2)
                    best_split_type = 'm'
                    best_split_val = m1

        # Try splitting p
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1, r1 = get_decomp(n, m, p1)
                U2, V2, W2, r2 = get_decomp(n, m, p2)
                r = r1 + r2
                if r < best_rank:
                    best_rank = r
                    best_decomp = (U1, V1, W1, U2, V2, W2)
                    best_split_type = 'p'
                    best_split_val = p1

        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1, r1 = get_decomp(n1, m, p)
                U2, V2, W2, r2 = get_decomp(n2, m, p)
                r = r1 + r2
                if r < best_rank:
                    best_rank = r
                    best_decomp = (U1, V1, W1, U2, V2, W2)
                    best_split_type = 'n'
                    best_split_val = n1

        if best_split_type is None:
            # Fallback to schoolbook if no split found (should not happen for n,m,p > 1)
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

        U1, V1, W1, U2, V2, W2 = best_decomp
        R1 = len(U1[0])
        R2 = len(U2[0])
        R = R1 + R2

        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]

        if best_split_type == 'm':
            m1 = best_split_val
            m2 = m - m1
            # Embed U1
            for i in range(n):
                for j in range(m1):
                    for c in range(R1):
                        U[i * m + j][c] = U1[i * m1 + j][c]
            # Embed U2
            for i in range(n):
                for j in range(m2):
                    for c in range(R2):
                        U[i * m + (m1 + j)][R1 + c] = U2[i * m2 + j][c]
            # Embed V1
            for j in range(m1):
                for k in range(p):
                    for c in range(R1):
                        V[j * p + k][c] = V1[j * p + k][c]
            # Embed V2
            for j in range(m2):
                for k in range(p):
                    for c in range(R2):
                        V[(m1 + j) * p + k][R1 + c] = V2[j * p + k][c]
            # Embed W
            for k in range(p):
                for i in range(n):
                    for c in range(R1):
                        W[k * n + i][c] = W1[k * n + i][c]
                    for c in range(R2):
                        W[k * n + i][R1 + c] = W2[k * n + i][c]

        elif best_split_type == 'p':
            p1 = best_split_val
            p2 = p - p1
            # Embed U
            # U is shared, so U1 and U2 should be identical in structure?
            # No, U1 is for M_{n,m,p1}, U2 for M_{n,m,p2}.
            # U1 has shape (n*m, R1), U2 has shape (n*m, R2).
            # The indices for i,j are the same.
            for i in range(n):
                for j in range(m):
                    for c in range(R1):
                        U[i * m + j][c] = U1[i * m + j][c]
                    for c in range(R2):
                        U[i * m + j][R1 + c] = U2[i * m + j][c]
            
            # Embed V1
            for j in range(m):
                for k in range(p1):
                    for c in range(R1):
                        V[j * p + k][c] = V1[j * p1 + k][c]
            # Embed V2
            for j in range(m):
                for k in range(p2):
                    for c in range(R2):
                        V[j * p + (p1 + k)][R1 + c] = V2[j * p2 + k][c]
            
            # Embed W1
            for k in range(p1):
                for i in range(n):
                    for c in range(R1):
                        W[k * n + i][c] = W1[k * n + i][c]
            # Embed W2
            for k in range(p2):
                for i in range(n):
                    for c in range(R2):
                        W[(p1 + k) * n + i][R1 + c] = W2[k * n + i][c]

        elif best_split_type == 'n':
            n1 = best_split_val
            n2 = n - n1
            # Embed U1
            for i in range(n1):
                for j in range(m):
                    for c in range(R1):
                        U[i * m + j][c] = U1[i * m + j][c]
            # Embed U2
            for i in range(n2):
                for j in range(m):
                    for c in range(R2):
                        U[(n1 + i) * m + j][R1 + c] = U2[i * m + j][c]
            
            # Embed V
            # V is shared? No, V is for M_{n1,m,p} and M_{n2,m,p}.
            # V1 shape (m*p, R1), V2 shape (m*p, R2).
            # Indices j,k are same.
            for j in range(m):
                for k in range(p):
                    for c in range(R1):
                        V[j * p + k][c] = V1[j * p + k][c]
                    for c in range(R2):
                        V[j * p + k][R1 + c] = V2[j * p + k][c]
            
            # Embed W1
            for k in range(p):
                for i in range(n1):
                    for c in range(R1):
                        W[k * n + i][c] = W1[k * n1 + i][c]
            # Embed W2
            for k in range(p):
                for i in range(n2):
                    for c in range(R2):
                        W[k * n + (n1 + i)][R1 + c] = W2[k * n2 + i][c]

        return U, V, W, R

    U, V, W, rank = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
