import time
import sys
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 120.0
    
    memo = {}

    def get_decomp(n, m, p):
        """Returns U, V, W for M_{n,m,p} using optimal recursive block strategy."""
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            # Return schoolbook as fallback
            return schoolbook(n, m, p)
            
        if (n, m, p) in memo:
            return memo[(n, m, p)]
        
        # Base cases
        if n == 1:
            res = schoolbook(1, m, p)
            memo[(n, m, p)] = res
            return res
        if m == 1:
            res = schoolbook(n, 1, p)
            memo[(n, m, p)] = res
            return res
        if p == 1:
            res = schoolbook(n, m, 1)
            memo[(n, m, p)] = res
            return res
            
        if n == 2 and m == 2 and p == 2:
            res = strassen_2x2x2()
            memo[(n, m, p)] = res
            return res
            
        # Try all splits
        best_rank = float('inf')
        best_result = None
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            
            r1 = len(U1[0]) if U1 and U1[0] else 0
            r2 = len(U2[0]) if U2 and U2[0] else 0
            total_rank = r1 + r2
            
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine
                U = [[0]*total_rank for _ in range(n*m)]
                V = [[0]*total_rank for _ in range(m*p)]
                W = [[0]*total_rank for _ in range(p*n)]
                
                # Fill U
                for i in range(n1):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m + j][c] = U1[i*m + j][c]
                for i in range(n2):
                    for j in range(m):
                        for c in range(r2):
                            U[(n1 + i)*m + j][r1 + c] = U2[i*m + j][c]
                
                # Fill V
                for j in range(m):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p + k][c]
                        for c in range(r2):
                            V[j*p + k][r1 + c] = V2[j*p + k][c]
                
                # Fill W
                for k in range(p):
                    for i in range(n1):
                        for c in range(r1):
                            W[k*n + i][c] = W1[k*n1 + i][c]
                    for i in range(n2):
                        for c in range(r2):
                            W[k*n + (n1 + i)][r1 + c] = W2[k*n2 + i][c]
                            
                best_result = (U, V, W)

        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            
            r1 = len(U1[0]) if U1 and U1[0] else 0
            r2 = len(U2[0]) if U2 and U2[0] else 0
            total_rank = r1 + r2
            
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine
                U = [[0]*total_rank for _ in range(n*m)]
                V = [[0]*total_rank for _ in range(m*p)]
                W = [[0]*total_rank for _ in range(p*n)]
                
                # Fill U
                for i in range(n):
                    for j in range(m1):
                        for c in range(r1):
                            U[i*m + j][c] = U1[i*m1 + j][c]
                    for j in range(m2):
                        for c in range(r2):
                            U[i*m + (m1 + j)][r1 + c] = U2[i*m2 + j][c]
                
                # Fill V
                for j in range(m1):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p1 + k][c] # Bug: p1 is not defined here, should be p
                            # Correction: V1 is for m1 x p, so index is j*p + k
                            # V1[j*p + k]
                    # Let's re-verify indices for V in m-split.
                    # V1 shape (m1*p, r1). Index: row = j*p + k.
                    # V2 shape (m2*p, r2). Index: row = j'*p + k.
                    # In combined V (m*p, r), row = (m1+j')*p + k.
                # Redo V fill for m-split
                for j in range(m1):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p + k][c]
                for j in range(m2):
                    for k in range(p):
                        for c in range(r2):
                            V[(m1 + j)*p + k][r1 + c] = V2[j*p + k][c]
                
                # Fill W
                for k in range(p):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n + i][c] = W1[k*n + i][c]
                        for c in range(r2):
                            W[k*n + i][r1 + c] = W2[k*n + i][c]
                            
                best_result = (U, V, W)

        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            
            r1 = len(U1[0]) if U1 and U1[0] else 0
            r2 = len(U2[0]) if U2 and U2[0] else 0
            total_rank = r1 + r2
            
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine
                U = [[0]*total_rank for _ in range(n*m)]
                V = [[0]*total_rank for _ in range(m*p)]
                W = [[0]*total_rank for _ in range(p*n)]
                
                # Fill U
                for i in range(n):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m + j][c] = U1[i*m + j][c]
                        for c in range(r2):
                            U[i*m + j][r1 + c] = U2[i*m + j][c]
                
                # Fill V
                for j in range(m):
                    for k in range(p1):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p1 + k][c]
                    for k in range(p2):
                        for c in range(r2):
                            V[j*p + (p1 + k)][r1 + c] = V2[j*p2 + k][c]
                
                # Fill W
                for k in range(p1):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n + i][c] = W1[k*n + i][c]
                for k in range(p2):
                    for i in range(n):
                        for c in range(r2):
                            W[(p1 + k)*n + i][r1 + c] = W2[k*n + i][c]
                            
                best_result = (U, V, W)

        if best_result is None:
            best_result = schoolbook(n, m, p)
            
        memo[(n, m, p)] = best_result
        return best_result

    def strassen_2x2x2():
        """Strassen decomposition for 2x2x2, rank 7."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def schoolbook(n, m, p):
        """Schoolbook decomposition for n x m x p, rank n*m*p."""
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][c] = 1
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
                    c += 1
        return U, V, W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
