import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 100  # 100s limit, 20s buffer
    
    def combine_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
        n = n1 + n2
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1+i)*m + j][r1+c] = U2[i*m + j][c]
                    
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
                for c in range(r2):
                    V[j*p + k][r1+c] = V2[j*p + k][c]
                    
        for k in range(p):
            for i in range(n1):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n1 + i][c]
            for i in range(n2):
                for c in range(r2):
                    W[k*n + (n1+i)][r1+c] = W2[k*n2 + i][c]
                    
        return U, V, W

    def combine_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
        m = m1 + m2
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m1 + j][c]
            for j in range(m2):
                for c in range(r2):
                    U[i*m + (m1+j)][r1+c] = U2[i*m2 + j][c]
                    
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1+j)*p + k][r1+c] = V2[j*p + k][c]
                    
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
                for c in range(r2):
                    W[k*n + i][r1+c] = W2[k*n + i][c]
                    
        return U, V, W

    def combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        p = p1 + p2
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
                for c in range(r2):
                    U[i*m + j][r1+c] = U2[i*m + j][c]
                    
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p1 + k][c]
            for k in range(p2):
                for c in range(r2):
                    V[j*p + (p1+k)][r1+c] = V2[j*p2 + k][c]
                    
        for k in range(p1):
            for i in range(n):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
        for k in range(p2):
            for i in range(n):
                for c in range(r2):
                    W[(p1+k)*n + i][r1+c] = W2[k*n + i][c]
                    
        return U, V, W

    def strassen_2x2x2():
        U = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, -1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1]
        ]
        W = [
            [1, 0, 0, -1, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, 0, -1, 0, 0, -1, 0]
        ]
        return U, V, W

    def schoolbook(n, m, p):
        rank = n*m*p
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

    def get_decomp(n, m, p):
        if time.time() > deadline:
            return schoolbook(n, m, p)
            
        if n == 1 and m == 1 and p == 1:
            return schoolbook(1,1,1)
        if n == 1:
            return schoolbook(1, m, p)
        if m == 1:
            return schoolbook(n, 1, p)
        if p == 1:
            return schoolbook(n, m, 1)
        
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
            
        best_rank = n*m*p
        best_U, best_V, best_W = None, None, None
        
        for n1 in range(1, n):
            if time.time() > deadline: break
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = combine_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
                
        for m1 in range(1, m):
            if time.time() > deadline: break
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = combine_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
                
        for p1 in range(1, p):
            if time.time() > deadline: break
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            r1 = len(U1[0])
            r2 = len(U2[0])
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
                
        if best_U is None:
            return schoolbook(n, m, p)
            
        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
