import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    # Cache for ranks to avoid recomputation
    rank_cache = {}
    
    def get_rank(n, m, p):
        if (n, m, p) in rank_cache:
            return rank_cache[(n, m, p)]
            
        # Base cases
        if n == 1 and m == 1 and p == 1:
            rank_cache[(n, m, p)] = 1
            return 1
        if n == 2 and m == 2 and p == 2:
            rank_cache[(n, m, p)] = 7
            return 7
            
        best = n * m * p
        
        # Try splitting n
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n - i, m, p)
            if r < best:
                best = r
                
        # Try splitting m
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m - j, p)
            if r < best:
                best = r
                
        # Try splitting p
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p - k)
            if r < best:
                best = r
                
        rank_cache[(n, m, p)] = best
        return best

    def build(n, m, p):
        """
        Returns U (n*m x r), V (m*p x r), W (p*n x r)
        for multiplying an n x m matrix by an m x p matrix.
        """
        r = get_rank(n, m, p)
        
        # Base case 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        # Base case 2x2x2 (Strassen)
        if n == 2 and m == 2 and p == 2:
            # U: 4x7, V: 4x7, W: 4x7
            # Indices for U: i*m+j -> i*2+j
            # Indices for V: j*p+k -> j*2+k
            # Indices for W: k*n+i -> k*2+i
            
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            # A00: U[0,0]=1, A11: U[3,0]=1
            # B00: V[0,0]=1, B11: V[3,0]=1
            # C00: W[0,0]=1 (k=0,i=0), C11: W[3,0]=1 (k=1,i=1)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # M2 = (A10+A11)B00 -> C10 += M2
            # A10: U[2,1]=1, A11: U[3,1]=1
            # B00: V[0,1]=1
            # C10: W[1,1]=1 (k=0,i=1)
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[1][1] = 1
            
            # M3 = A00(B01-B11) -> C01 += M3
            # A00: U[0,2]=1
            # B01: V[1,2]=1, B11: V[3,2]=-1
            # C01: W[2,2]=1 (k=1,i=0)
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[2][2] = 1
            
            # M4 = A11(B10-B00) -> C11 += M4
            # A11: U[3,3]=1
            # B10: V[2,3]=1, B00: V[0,3]=-1
            # C11: W[3,3]=1
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[3][3] = 1
            
            # M5 = (A00+A01)B11 -> C01 += M5
            # A00: U[0,4]=1, A01: U[1,4]=1
            # B11: V[3,4]=1
            # C01: W[2,4]=1
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[2][4] = 1
            
            # M6 = (A10-A00)(B00+B01) -> C10 += M6
            # A10: U[2,5]=1, A00: U[0,5]=-1
            # B00: V[0,5]=1, B01: V[1,5]=1
            # C10: W[1,5]=1
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[1][5] = 1
            
            # M7 = (A01-A11)(B10+B11) -> C11 += M7
            # A01: U[1,6]=1, A11: U[3,6]=-1
            # B10: V[2,6]=1, B11: V[3,6]=1
            # C11: W[3,6]=1
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[3][6] = 1
            
            return U, V, W

        # Find best split
        best_rank = n * m * p
        best_split = None
        best_type = None
        
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n - i, m, p)
            if r < best_rank:
                best_rank = r
                best_split = i
                best_type = 'n'
                
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m - j, p)
            if r < best_rank:
                best_rank = r
                best_split = j
                best_type = 'm'
                
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p - k)
            if r < best_rank:
                best_rank = r
                best_split = k
                best_type = 'p'

        # If no split found better than schoolbook, use schoolbook
        if best_type is None:
            U = [[0] * best_rank for _ in range(n * m)]
            V = [[0] * best_rank for _ in range(m * p)]
            W = [[0] * best_rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return U, V, W

        # Recursive construction
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        
        if best_type == 'n':
            i = best_split
            U1, V1, W1 = build(i, m, p)
            U2, V2, W2 = build(n - i, m, p)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            
            # Copy part 1
            for c in range(r1):
                for row in range(len(U1)):
                    U[row][c] = U1[row][c]
                for row in range(len(V1)):
                    V[row][c] = V1[row][c]
                for row in range(len(W1)):
                    W[row][c] = W1[row][c]
                    
            # Copy part 2
            offset = r1
            for c in range(r2):
                for row in range(len(U2)):
                    U[i * m + row][offset + c] = U2[row][c]
                for row in range(len(V2)):
                    V[row][offset + c] = V2[row][c]
                # W mapping for n-split:
                # W1 index: k*i + i'  (i' in 0..i-1)
                # W2 index: k*(n-i) + i'' (i'' in 0..n-i-1)
                # Parent W index: k*n + i_parent
                # For part 1: i_parent = i'
                # For part 2: i_parent = i + i''
                
                for row in range(len(W2)):
                    # row = k*(n-i) + i''
                    k = row // (n - i)
                    i_prime_prime = row % (n - i)
                    parent_i = i + i_prime_prime
                    parent_row = k * n + parent_i
                    W[parent_row][offset + c] = W2[row][c]
            
            return U, V, W
            
        elif best_type == 'm':
            j = best_split
            U1, V1, W1 = build(n, j, p)
            U2, V2, W2 = build(n, m - j, p)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            
            # Copy part 1
            for c in range(r1):
                for row in range(len(U1)):
                    U[row][c] = U1[row][c]
                for row in range(len(V1)):
                    V[row][c] = V1[row][c]
                for row in range(len(W1)):
                    W[row][c] = W1[row][c]
                    
            # Copy part 2
            offset = r1
            for c in range(r2):
                # U mapping for m-split:
                # U1 row: i*j + j' (j' in 0..j-1)
                # U2 row: i*(m-j) + j'' (j'' in 0..m-j-1)
                # Parent U row: i*m + j_parent
                # Part 1: j_parent = j'
                # Part 2: j_parent = j + j''
                
                for row in range(len(U2)):
                    i = row // (m - j)
                    j_prime_prime = row % (m - j)
                    parent_j = j + j_prime_prime
                    parent_row = i * m + parent_j
                    U[parent_row][offset + c] = U2[row][c]
                    
                # V mapping for m-split:
                # V1 row: j'*p + k
                # V2 row: j''*p + k
                # Parent V row: j_parent*p + k
                # Part 1: j_parent = j'
                # Part 2: j_parent = j + j''
                
                for row in range(len(V2)):
                    j_prime_prime = row // p
                    k = row % p
                    parent_j = j + j_prime_prime
                    parent_row = parent_j * p + k
                    V[parent_row][offset + c] = V2[row][c]
                    
                for row in range(len(W2)):
                    W[row][offset + c] = W2[row][c]
            
            return U, V, W
            
        elif best_type == 'p':
            k = best_split
            U1, V1, W1 = build(n, m, k)
            U2, V2, W2 = build(n, m, p - k)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            
            # Copy part 1
            for c in range(r1):
                for row in range(len(U1)):
                    U[row][c] = U1[row][c]
                for row in range(len(V1)):
                    V[row][c] = V1[row][c]
                for row in range(len(W1)):
                    W[row][c] = W1[row][c]
                    
            # Copy part 2
            offset = r1
            for c in range(r2):
                for row in range(len(U2)):
                    U[row][offset + c] = U2[row][c]
                    
                # V mapping for p-split:
                # V1 row: j*k + k' (k' in 0..k-1)
                # V2 row: j*(p-k) + k'' (k'' in 0..p-k-1)
                # Parent V row: j*p + k_parent
                # Part 1: k_parent = k'
                # Part 2: k_parent = k + k''
                
                for row in range(len(V2)):
                    j = row // (p - k)
                    k_prime_prime = row % (p - k)
                    parent_k = k + k_prime_prime
                    parent_row = j * p + parent_k
                    V[parent_row][offset + c] = V2[row][c]
                    
                # W mapping for p-split:
                # W1 row: k'*n + i
                # W2 row: k''*n + i
                # Parent W row: k_parent*n + i
                # Part 1: k_parent = k'
                # Part 2: k_parent = k + k''
                
                for row in range(len(W2)):
                    k_prime_prime = row // n
                    i = row % n
                    parent_k = k + k_prime_prime
                    parent_row = parent_k * n + i
                    W[parent_row][offset + c] = W2[row][c]
            
            return U, V, W

    U, V, W = build(n, m, p)
    return {"U": U, "V": V, "W": W}
