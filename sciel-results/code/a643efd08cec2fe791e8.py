import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit
    
    # We will build the decomposition tree
    # Node: (n, m, p, U, V, W, rank)
    # But U, V, W are relative to the sub-problem.
    # We need to map them to the global U, V, W.
    
    # Let's use a list of operations.
    # Each operation is a rank-1 term or a sub-decomposition.
    
    # Recursive function to get the decomposition
    # Returns: rank, list of (i_indices, j_indices, k_indices, u_mat, v_mat, w_mat)
    # Where u_mat is (len(i)*len(j), rank), etc.
    
    def get_decomposition(n, m, p):
        # Check deadline
        if time.time() > deadline:
            # Fallback to schoolbook
            rank = n * m * p
            # This is complex to fallback mid-recursion.
            # Better to check at the start of the function.
            pass

        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return 1, U, V, W
        
        if p == 1:
            rank = n * m
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m)]
            W = [[0]*rank for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    c = i*m + j
                    U[i*m+j][c] = 1
                    V[j][c] = 1
                    W[i][c] = 1
            return rank, U, V, W
        
        if m == 1:
            rank = n * p
            U = [[0]*rank for _ in range(n)]
            V = [[0]*rank for _ in range(p)]
            W = [[0]*rank for _ in range(n*p)]
            for i in range(n):
                for k in range(p):
                    c = i*p + k
                    U[i][c] = 1
                    V[k][c] = 1
                    W[k*n+i][c] = 1
            return rank, U, V, W
            
        if n == 1:
            rank = m * p
            U = [[0]*rank for _ in range(m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p)]
            for j in range(m):
                for k in range(p):
                    c = j*p + k
                    U[j][c] = 1
                    V[j*p+k][c] = 1
                    W[k][c] = 1
            return rank, U, V, W
            
        if n == 2 and m == 2 and p == 2:
            # Winograd 2x2x2
            # Rank 7
            # Hardcode
            U = [
                [1, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, -1, 0],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1]
            ]
            V = [
                [1, 0, 1, 1, 1, 0, 1],
                [0, 1, 1, 0, -1, 1, 1],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1]
            ]
            W = [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, -1, 0],
                [0, 0, 0, 1, 1, 1, 0]
            ]
            # Verify: C11 = U11 V11 W11 + ...
            # This hardcoded matrix might be wrong.
            # Let's use a known correct Winograd 2x2.
            # Standard Winograd F2:
            # d1 = a11 b11
            # d2 = a12 b22
            # d3 = (a11+a12) b12
            # d4 = (a11+a12) b21
            # d5 = (a11-a12) b11
            # d6 = (a11-a12) b22
            # d7 = (a21+a22) (b11+b12) -- NO, this is not symmetric.
            
            # Correct Winograd 2x2 Matrix Mult:
            # m1 = a11 b11
            # m2 = a12 b22
            # m3 = (a11+a12) b12
            # m4 = (a11+a12) b21
            # m5 = (a11-a12) b11
            # m6 = (a11-a12) b22
            # m7 = (a21+a22) (b11+b12+b21+b22)? No.
            
            # Let's just use schoolbook for 2x2x2 if we can't verify Winograd quickly.
            # Rank 4.
            rank = 4
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(4)]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        c = i*4 + j*2 + k
                        U[i*2+j][c] = 1
                        V[j*2+k][c] = 1
                        W[k*2+i][c] = 1
            return rank, U, V, W

        # Recursive split
        best_rank = float('inf')
        best_decomp = None
        
        # Try splitting n
        for i in range(1, n):
            if time.time() > deadline: break
            n1, n2 = i, n-i
            r1, U1, V1, W1 = get_decomposition(n1, m, p)
            r2, U2, V2, W2 = get_decomposition(n2, m, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                # Combine
                # U: (n*m, r)
                # U1 is (n1*m, r1), U2 is (n2*m, r2)
                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                # Copy U1
                for row in range(n1*m):
                    for col in range(r1):
                        U[row][col] = U1[row][col]
                # Copy U2
                for row in range(n2*m):
                    for col in range(r2):
                        U[n1*m + row][r1 + col] = U2[row][col]
                        
                # Copy V
                for row in range(m*p):
                    for col in range(r1):
                        V[row][col] = V1[row][col]
                    for col in range(r2):
                        V[row][r1 + col] = V2[row][col]
                        
                # Copy W
                # W1 is (p*n1, r1), W2 is (p*n2, r2)
                # Global W index: k*n + i
                # For W1, i is in 0..n1-1
                # For W2, i is in n1..n-1
                for k in range(p):
                    for i1 in range(n1):
                        for col in range(r1):
                            W[k*n + i1][col] = W1[k*n1 + i1][col]
                    for i2 in range(n2):
                        for col in range(r2):
                            W[k*n + (n1 + i2)][r1 + col] = W2[k*n2 + i2][col]
                
                best_decomp = (r, U, V, W)
        
        # Try splitting m
        for j in range(1, m):
            if time.time() > deadline: break
            m1, m2 = j, m-j
            r1, U1, V1, W1 = get_decomposition(n, m1, p)
            r2, U2, V2, W2 = get_decomposition(n, m2, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                # U: (n*m, r)
                # U1: (n*m1, r1), U2: (n*m2, r2)
                for i in range(n):
                    for j1 in range(m1):
                        for col in range(r1):
                            U[i*m + j1][col] = U1[i*m1 + j1][col]
                    for j2 in range(m2):
                        for col in range(r2):
                            U[i*m + (m1 + j2)][r1 + col] = U2[i*m2 + j2][col]
                            
                # V: (m*p, r)
                # V1: (m1*p, r1), V2: (m2*p, r2)
                for j1 in range(m1):
                    for k in range(p):
                        for col in range(r1):
                            V[j1*p + k][col] = V1[j1*p + k][col]
                for j2 in range(m2):
                    for k in range(p):
                        for col in range(r2):
                            V[(m1 + j2)*p + k][r1 + col] = V2[j2*p + k][col]
                            
                # W: (p*n, r)
                for k in range(p):
                    for i in range(n):
                        for col in range(r1):
                            W[k*n + i][col] = W1[k*n + i][col]
                        for col in range(r2):
                            W[k*n + i][r1 + col] = W2[k*n + i][col]
                
                best_decomp = (r, U, V, W)
                
        # Try splitting p
        for k in range(1, p):
            if time.time() > deadline: break
            p1, p2 = k, p-k
            r1, U1, V1, W1 = get_decomposition(n, m, p1)
            r2, U2, V2, W2 = get_decomposition(n, m, p2)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                # U
                for i in range(n):
                    for j in range(m):
                        for col in range(r1):
                            U[i*m + j][col] = U1[i*m + j][col]
                        for col in range(r2):
                            U[i*m + j][r1 + col] = U2[i*m + j][col]
                            
                # V: (m*p, r)
                # V1: (m*p1, r1), V2: (m*p2, r2)
                for j in range(m):
                    for k1 in range(p1):
                        for col in range(r1):
                            V[j*p + k1][col] = V1[j*p1 + k1][col]
                    for k2 in range(p2):
                        for col in range(r2):
                            V[j*p + (p1 + k2)][r1 + col] = V2[j*p2 + k2][col]
                            
                # W: (p*n, r)
                # W1: (p1*n, r1), W2: (p2*n, r2)
                for k1 in range(p1):
                    for i in range(n):
                        for col in range(r1):
                            W[k1*n + i][col] = W1[k1*n + i][col]
                for k2 in range(p2):
                    for i in range(n):
                        for col in range(r2):
                            W[(p1 + k2)*n + i][r1 + col] = W2[k2*n + i][col]
                
                best_decomp = (r, U, V, W)
                
        if best_decomp is None:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][c] = 1
                        V[j*p+k][c] = 1
                        W[k*n+i][c] = 1
                        c += 1
            return rank, U, V, W
            
        return best_decomp

    rank, U, V, W = get_decomposition(3, 5, 7)
    return {"U": U, "V": V, "W": W}
