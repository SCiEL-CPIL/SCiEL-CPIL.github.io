import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # We will build the tensor decomposition recursively.
    # Base cases:
    # M_{1,1,1} rank 1
    # M_{2,2,2} rank 7 (Strassen)
    # M_{2,2,1} rank 4 (Schoolbook)
    # M_{1,k,1} rank k (Schoolbook)
    
    # We represent a decomposition as a list of tuples (u, v, w)
    # where u, v, w are sparse vectors (dicts or lists) representing the rank-1 terms.
    # Actually, it's easier to build U, V, W directly.
    
    # Let's use a helper that returns U, V, W for a given n, m, p using the best known strategy.
    
    def get_decomp(n, m, p):
        # Returns U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # Strategy: Split the largest dimension or use known blocks.
        
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        # Base case: Strassen 2x2x2
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = np.zeros((4, rank))
            V = np.zeros((4, rank))
            W = np.zeros((4, rank))
            
            # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
            # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
            # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
            
            # M1 = (A00+A11)(B00+B11)
            U[0,0] += 1; U[3,0] += 1
            V[0,0] += 1; V[3,0] += 1
            W[0,0] += 1; W[3,0] += 1
            
            # M2 = (A10+A11)B00
            U[2,1] += 1; U[3,1] += 1
            V[0,1] += 1
            W[2,1] += 1; W[3,1] -= 1
            
            # M3 = A00(B10-B11)
            U[0,2] += 1
            V[2,2] += 1; V[3,2] -= 1
            W[1,2] += 1; W[3,2] += 1
            
            # M4 = A11(B01-B00)
            U[3,3] += 1
            V[1,3] += 1; V[0,3] -= 1
            W[0,3] += 1; W[2,3] += 1
            
            # M5 = (A00+A01)B11
            U[0,4] += 1; U[1,4] += 1
            V[3,4] += 1
            W[0,4] -= 1; W[1,4] += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[2,5] += 1; U[0,5] -= 1
            V[0,5] += 1; V[1,5] += 1
            W[3,5] += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[1,6] += 1; U[3,6] -= 1
            V[2,6] += 1; V[3,6] += 1
            W[0,6] += 1
            
            return U.tolist(), V.tolist(), W.tolist()

        # Base case: 2x2x1 (Matrix Vector)
        if n == 2 and m == 2 and p == 1:
            rank = 4
            U = np.zeros((4, rank))
            V = np.zeros((2, rank)) # B is 2x1, so m*p = 2*1 = 2
            W = np.zeros((2, rank)) # C is 2x1, so p*n = 1*2 = 2
            
            # A indices: 0,1,2,3
            # B indices: 0,1
            # C indices: 0,1 (row 0, row 1)
            
            # C0 = A00*B0 + A01*B1
            # C1 = A10*B0 + A11*B1
            
            U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
            U[1,1] = 1; V[1,1] = 1; W[0,1] = 1
            U[2,2] = 1; V[0,2] = 1; W[1,2] = 1
            U[3,3] = 1; V[1,3] = 1; W[1,3] = 1
            
            return U.tolist(), V.tolist(), W.tolist()

        # Recursive Step: Split dimensions
        # We try to split the dimension that allows the best combination of sub-problems.
        # For 2x4x7, we split m=4 into 2+2.
        if n == 2 and m == 4 and p == 7:
            # Split m into m1=2, m2=2
            m1, m2 = 2, 2
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            
            # Combine
            rank = len(U1[0]) + len(U2[0])
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Map indices for part 1 (m1)
            # A part 1: rows 0..n-1, cols 0..m1-1
            # A part 2: rows 0..n-1, cols m1..m-1
            # B part 1: rows 0..m1-1, cols 0..p-1
            # B part 2: rows m1..m-1, cols 0..p-1
            # C is same for both
            
            r1 = len(U1[0])
            # Copy U1 to U
            for i in range(n):
                for j in range(m1):
                    U[i*m + j, 0:r1] = U1[i*m1 + j, 0:r1]
            # Copy U2 to U
            for i in range(n):
                for j in range(m2):
                    U[i*m + (m1+j), r1:rank] = U2[i*m2 + j, 0:r1] # Wait, U2 shape is n*m2 x r1? No, r2
            r2 = len(U2[0])
            for i in range(n):
                for j in range(m2):
                    U[i*m + (m1+j), r1:rank] = U2[i*m2 + j, 0:r2]
                    
            # Copy V1 to V
            for i in range(m1):
                for j in range(p):
                    V[i*p + j, 0:r1] = V1[i*p + j, 0:r1]
            # Copy V2 to V
            for i in range(m2):
                for j in range(p):
                    V[(m1+i)*p + j, r1:rank] = V2[i*p + j, 0:r2]
                    
            # Copy W (same for both, just sum? No, they are independent ranks)
            # W1 and W2 are in different rank columns, so just copy
            for i in range(p):
                for j in range(n):
                    W[i*n + j, 0:r1] = W1[i*n + j, 0:r1]
                    W[i*n + j, r1:rank] = W2[i*n + j, 0:r2]
                    
            return U.tolist(), V.tolist(), W.tolist()
            
        # Fallback: Schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        return U.tolist(), V.tolist(), W.tolist()

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
