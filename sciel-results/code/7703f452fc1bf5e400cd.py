import numpy as np

def construct():
    n, m, p = 3, 5, 5
    
    # We will build the decomposition recursively.
    # Returns U, V, W as lists of lists.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    def get_decomp(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
            
        if n == 1:
            # Rank m
            rank = m
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        if r == j:
                            U[j][r] = 1
                            V[j*p+k][r] = 1
                            W[k*n][r] = 1 # W index k*n + 0
            return U, V, W
            
        if m == 1:
            # Rank 1
            rank = 1
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n):
                for k in range(p):
                    U[i*m][0] = 1
                    V[0*p+k][0] = 1
                    W[k*n+i][0] = 1
            return U, V, W
            
        if p == 1:
            # Rank 1
            rank = 1
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][0] = 1
                    V[j*p][0] = 1
                    W[i][0] = 1 # W index 0*n + i
            return U, V, W

        if n == 2 and m == 2 and p == 2:
            # Strassen
            # Rank 7
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01+B11)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B11+B01)
            
            # U rows: 00, 01, 10, 11
            # V rows: 00, 01, 10, 11
            # W rows: 00, 01, 10, 11 (k*n+i -> 0*2+0=0, 1*2+0=2, 0*2+1=1, 1*2+1=3)
            # Wait, W index is k*n+i.
            # k=0, i=0 -> 0
            # k=1, i=0 -> 2
            # k=0, i=1 -> 1
            # k=1, i=1 -> 3
            
            U = [
                [1, 0, 1, 0, 1, -1, 0], # A00
                [0, 0, 0, 0, 1, 0, 1],  # A01
                [0, 1, 0, 0, 0, 1, 0],  # A10
                [1, 1, 0, 1, 0, 0, -1]  # A11
            ]
            V = [
                [1, 1, 0, 0, 0, 1, 0], # B00
                [0, 0, 0, 1, 0, 0, 1], # B01
                [0, 0, 1, 0, 0, 1, 0], # B10
                [1, 0, -1, 1, 1, 0, 1] # B11
            ]
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # W rows:
            # k=0, i=0 (C00): M1, M4, -M5, M7 -> [1, 0, 0, 1, -1, 0, 1]
            # k=1, i=0 (C10): M2, M4 -> [0, 1, 0, 1, 0, 0, 0]
            # k=0, i=1 (C01): M3, M5 -> [0, 0, 1, 0, 1, 0, 0]
            # k=1, i=1 (C11): M1, -M2, M3, M6 -> [1, -1, 1, 0, 0, 1, 0]
            
            W = [
                [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
                [0, 0, 1, 0, 1, 0, 0],  # C01 (k=0, i=1)
                [0, 1, 0, 1, 0, 0, 0],  # C10 (k=1, i=0)
                [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
            ]
            return U, V, W

        # Recursive case
        # Choose the dimension to split
        # Heuristic: split the largest dimension, or the one that allows a known low-rank split
        # Try splitting n
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            # Fill U1
            for i in range(n1):
                for j in range(m):
                    for r in range(rank1):
                        U[i*m+j][r] = U1[i*m+j][r]
            
            # Fill U2
            for i in range(n2):
                for j in range(m):
                    for r in range(rank2):
                        U[(n1+i)*m+j][rank1+r] = U2[i*m+j][r]
            
            # Fill V
            for j in range(m):
                for k in range(p):
                    for r in range(rank1):
                        V[j*p+k][r] = V1[j*p+k][r]
                    for r in range(rank2):
                        V[j*p+k][rank1+r] = V2[j*p+k][r]
            
            # Fill W
            # W1 rows correspond to i in [0, n1)
            # W2 rows correspond to i in [0, n2)
            # W index is k*n + i
            for k in range(p):
                for i in range(n1):
                    for r in range(rank1):
                        W[k*n+i][r] = W1[k*n1+i][r]
                for i in range(n2):
                    for r in range(rank2):
                        W[k*n + (n1+i)][rank1+r] = W2[k*n2+i][r]
                        
            return U, V, W

        # If n==1, handled above. So this shouldn't happen if n>1 logic is first.
        # But if n==1, we return.
        # What if n==1 but we didn't catch it?
        # The base case n==1 is caught.
        
        # What if n==1 is not the only base case?
        # If n==1, m>1, p>1, we return in base case.
        
        # If n>1, we split n.
        # Is it possible n==1, m==1, p>1? Handled.
        
        return None

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
