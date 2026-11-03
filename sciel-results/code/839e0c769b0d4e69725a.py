import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # Memoization cache
    cache = {}
    
    def get_base(n, m, p):
        """Returns (U, V, W) for base cases."""
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        if n == 2 and m == 2 and p == 2:
            # Strassen's algorithm
            # U: (n*m, rank) = (4, 7)
            # V: (m*p, rank) = (4, 7)
            # W: (p*n, rank) = (4, 7)
            # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
            # 0-indexed: i,j,k in {0,1}
            
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # M1 = (A11+A22)(B11+B22)
            # U: A11(0,0)->0, A22(1,1)->3
            U[0][0] = 1; U[3][0] = 1
            # V: B11(0,0)->0, B22(1,1)->3
            V[0][0] = 1; V[3][0] = 1
            # W: C11(0,0)->0, C22(1,1)->3
            W[0][0] = 1; W[3][0] = 1
            
            # M2 = (A21+A22)B11
            # U: A21(1,0)->2, A22(1,1)->3
            U[2][1] = 1; U[3][1] = 1
            # V: B11(0,0)->0
            V[0][1] = 1
            # W: C21(0,1)->2
            W[2][1] = 1
            
            # M3 = A11(B12-B22)
            # U: A11(0,0)->0
            U[0][2] = 1
            # V: B12(0,1)->1, B22(1,1)->3
            V[1][2] = 1; V[3][2] = -1
            # W: C12(1,0)->1
            W[1][2] = 1
            
            # M4 = A22(B21-B11)
            # U: A22(1,1)->3
            U[3][3] = 1
            # V: B21(1,0)->2, B11(0,0)->0
            V[2][3] = 1; V[0][3] = -1
            # W: C11(0,0)->0, C21(0,1)->2
            W[0][3] = 1; W[2][3] = 1
            
            # M5 = (A11+A12)B22
            # U: A11(0,0)->0, A12(0,1)->1
            U[0][4] = 1; U[1][4] = 1
            # V: B22(1,1)->3
            V[3][4] = 1
            # W: C11(0,0)->0, C12(1,0)->1
            W[0][4] = -1; W[1][4] = 1
            
            # M6 = (A21-A11)(B11+B12)
            # U: A21(1,0)->2, A11(0,0)->0
            U[2][5] = 1; U[0][5] = -1
            # V: B11(0,0)->0, B12(0,1)->1
            V[0][5] = 1; V[1][5] = 1
            # W: C22(1,1)->3
            W[3][5] = 1
            
            # M7 = (A12-A22)(B21+B22)
            # U: A12(0,1)->1, A22(1,1)->3
            U[1][6] = 1; U[3][6] = -1
            # V: B21(1,0)->2, B22(1,1)->3
            V[2][6] = 1; V[3][6] = 1
            # W: C11(0,0)->0, C22(1,1)->3
            W[0][6] = 1; W[3][6] = 1
            
            return U, V, W
        
        # Fallback: Schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def add_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
        """Combine two decompositions split along n."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        U = [[0]*r for _ in range((n1+n2)*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*(n1+n2))]
        
        # Copy U1, V1, W1
        for i in range(n1*m):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(m*p):
            for c in range(r1):
                V[i][c] = V1[i][c]
        for i in range(p*n1):
            for c in range(r1):
                W[i][c] = W1[i][c]
        
        # Copy U2, V2, W2 with offset
        for i in range(n2*m):
            for c in range(r2):
                U[n1*m + i][r1 + c] = U2[i][c]
        for i in range(m*p):
            for c in range(r2):
                V[i][r1 + c] = V2[i][c]
        for i in range(p*n2):
            for c in range(r2):
                W[p*n1 + i][r1 + c] = W2[i][c]
                
        return U, V, W

    def add_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
        """Combine two decompositions split along m."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        U = [[0]*r for _ in range(n*(m1+m2))]
        V = [[0]*r for _ in range((m1+m2)*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n*m1):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(m1*p):
            for c in range(r1):
                V[i][c] = V1[i][c]
        for i in range(p*n):
            for c in range(r1):
                W[i][c] = W1[i][c]
        
        for i in range(n*m2):
            for c in range(r2):
                U[n*m1 + i][r1 + c] = U2[i][c]
        for i in range(m2*p):
            for c in range(r2):
                V[m1*p + i][r1 + c] = V2[i][c]
        for i in range(p*n):
            for c in range(r2):
                W[i][r1 + c] = W2[i][c]
                
        return U, V, W

    def add_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """Combine two decompositions split along p."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*(p1+p2))]
        W = [[0]*r for _ in range((p1+p2)*n)]
        
        for i in range(n*m):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(m*p1):
            for c in range(r1):
                V[i][c] = V1[i][c]
        for i in range(p1*n):
            for c in range(r1):
                W[i][c] = W1[i][c]
        
        for i in range(n*m):
            for c in range(r2):
                U[i][r1 + c] = U2[i][c]
        for i in range(m*p2):
            for c in range(r2):
                V[m*p1 + i][r1 + c] = V2[i][c]
        for i in range(p2*n):
            for c in range(r2):
                W[p1*n + i][r1 + c] = W2[i][c]
                
        return U, V, W

    def build(n, m, p):
        key = (n, m, p)
        if key in cache:
            return cache[key]
        
        if time.time() > deadline - 10:
            # Fallback to schoolbook if running out of time
            U, V, W = get_base(n, m, p)
            cache[key] = (U, V, W)
            return U, V, W

        # Base case
        if n == 1 and m == 1 and p == 1:
            U, V, W = get_base(n, m, p)
            cache[key] = (U, V, W)
            return U, V, W
            
        if n == 2 and m == 2 and p == 2:
            U, V, W = get_base(n, m, p)
            cache[key] = (U, V, W)
            return U, V, W

        best_rank = float('inf')
        best_decomp = None
        
        # Try splitting n
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = build(n1, m, p)
            U2, V2, W2 = build(n2, m, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                U, V, W = add_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
                best_decomp = (U, V, W)
                
        # Try splitting m
        if m > 1:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = build(n, m1, p)
            U2, V2, W2 = build(n, m2, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                U, V, W = add_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
                best_decomp = (U, V, W)
                
        # Try splitting p
        if p > 1:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = build(n, m, p1)
            U2, V2, W2 = build(n, m, p2)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                U, V, W = add_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
                best_decomp = (U, V, W)
                
        if best_decomp is None:
            U, V, W = get_base(n, m, p)
            best_decomp = (U, V, W)
            
        cache[key] = best_decomp
        return best_decomp

    n, m, p = DIMENSIONS
    U, V, W = build(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
