# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    
    def get_decomposition(n, m, p):
        """
        Returns U, V, W for matrix multiplication of sizes n x m, m x p -> n x p.
        U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        """
        # Base case: Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            # Strassen decomposition
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # Indices:
            # A: 00, 01, 10, 11 -> 0, 1, 2, 3
            # B: 00, 01, 10, 11 -> 0, 1, 2, 3
            # C: 00, 01, 10, 11 -> 0, 1, 2, 3
            
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
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
            W[0][6] = 1
            
            return U, V, W

        # Base case: If any dimension is 1, use schoolbook
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return U, V, W

        # Recursive case: Split a dimension into 2 and rest
        # Priority: Split n, then m, then p?
        # To match the optimal rank analysis, we should split any dimension >= 2 into 2 and rest.
        
        if n >= 2:
            n1, n2 = 2, n - 2
            # A is n x m, B is m x p
            # Split A into A1 (n1 x m) and A2 (n2 x m)
            # B is same
            # C splits into C1 (n1 x p) and C2 (n2 x p)
            
            U1, V1, W1 = get_decomposition(n1, m, p)
            U2, V2, W2 = get_decomposition(n2, m, p)
            
            rank = len(V1[0]) + len(V2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Place U1, U2
            for i in range(n1):
                for j in range(m):
                    for r in range(len(V1[0])):
                        U[i*m + j][r] = U1[i*m + j][r]
            for i in range(n2):
                for j in range(m):
                    for r in range(len(V2[0])):
                        U[(n1+i)*m + j][len(V1[0]) + r] = U2[i*m + j][r]
                        
            # Place V1, V2 (V is same for both? No, V depends on B, which is same size)
            # Wait, V is m x p. It is the same for both sub-problems?
            # Yes, B is the same.
            for j in range(m):
                for k in range(p):
                    for r in range(len(V1[0])):
                        V[j*p + k][r] = V1[j*p + k][r]
                    for r in range(len(V2[0])):
                        V[j*p + k][len(V1[0]) + r] = V2[j*p + k][r]
                        
            # Place W1, W2
            for k in range(p):
                for i in range(n1):
                    for r in range(len(W1[0])):
                        W[k*n + i][r] = W1[k*n1 + i][r]
            for k in range(p):
                for i in range(n2):
                    for r in range(len(W2[0])):
                        W[k*n + (n1+i)][len(W1[0]) + r] = W2[k*n2 + i][r]
                        
            return U, V, W
            
        elif m >= 2:
            m1, m2 = 2, m - 2
            # A splits into A1 (n x m1) and A2 (n x m2)
            # B splits into B1 (m1 x p) and B2 (m2 x p)
            # C is sum of C1 and C2
            
            U1, V1, W1 = get_decomposition(n, m1, p)
            U2, V2, W2 = get_decomposition(n, m2, p)
            
            rank = len(V1[0]) + len(V2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Place U
            for i in range(n):
                for j in range(m1):
                    for r in range(len(V1[0])):
                        U[i*m + j][r] = U1[i*m1 + j][r]
                for j in range(m2):
                    for r in range(len(V2[0])):
                        U[i*m + (m1+j)][len(V1[0]) + r] = U2[i*m2 + j][r]
                        
            # Place V
            for j in range(m1):
                for k in range(p):
                    for r in range(len(V1[0])):
                        V[j*p + k][r] = V1[j*p + k][r]
            for j in range(m2):
                for k in range(p):
                    for r in range(len(V2[0])):
                        V[(m1+j)*p + k][len(V1[0]) + r] = V2[j*p + k][r]
                        
            # Place W
            for k in range(p):
                for i in range(n):
                    for r in range(len(W1[0])):
                        W[k*n + i][r] = W1[k*n + i][r]
                    for r in range(len(W2[0])):
                        W[k*n + i][len(W1[0]) + r] = W2[k*n + i][r]
                        
            return U, V, W
            
        elif p >= 2:
            p1, p2 = 2, p - 2
            # A is same
            # B splits into B1 (m x p1) and B2 (m x p2)
            # C splits into C1 (n x p1) and C2 (n x p2)
            
            U1, V1, W1 = get_decomposition(n, m, p1)
            U2, V2, W2 = get_decomposition(n, m, p2)
            
            rank = len(V1[0]) + len(V2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Place U
            for i in range(n):
                for j in range(m):
                    for r in range(len(V1[0])):
                        U[i*m + j][r] = U1[i*m + j][r]
                    for r in range(len(V2[0])):
                        U[i*m + j][len(V1[0]) + r] = U2[i*m + j][r]
                        
            # Place V
            for j in range(m):
                for k in range(p1):
                    for r in range(len(V1[0])):
                        V[j*p + k][r] = V1[j*p1 + k][r]
                for k in range(p2):
                    for r in range(len(V2[0])):
                        V[j*p + (p1+k)][len(V1[0]) + r] = V2[j*p2 + k][r]
                        
            # Place W
            for k in range(p1):
                for i in range(n):
                    for r in range(len(W1[0])):
                        W[k*n + i][r] = W1[k*n + i][r]
            for k in range(p2):
                for i in range(n):
                    for r in range(len(W2[0])):
                        W[(p1+k)*n + i][len(W1[0]) + r] = W2[k*n + i][r]
                        
            return U, V, W
            
        else:
            # Should not happen if base cases handled
            return get_decomposition(n, m, p) # Fallback

    U, V, W = get_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
