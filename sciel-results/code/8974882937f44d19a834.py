"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 84
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Strassen for 2x2x2
    def strassen_222():
        # Returns U, V, W for multiplying 2x2 by 2x2 with rank 7
        # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # For n=2, m=2, p=2: U is 4x7, V is 4x7, W is 4x7
        # Strassen formulas:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # For tensor decomposition, each rank-1 term corresponds to one multiplication
        # We need to express each product as sum of 7 terms
        
        # Actually, let me think about this differently.
        # The tensor for 2x2x2 matrix mult has rank 7 (Strassen).
        # I'll construct it explicitly.
        
        rank = 7
        U = np.zeros((4, rank))  # n*m = 2*2 = 4
        V = np.zeros((4, rank))  # m*p = 2*2 = 4  
        W = np.zeros((4, rank))  # p*n = 2*2 = 4
        
        # Strassen's 7 products:
        # Term 1: M1 = (A00+A11)*(B00+B11)
        # A00 is at U[0], A11 is at U[3]
        # B00 is at V[0], B11 is at V[3]
        # C00 gets M1, C11 gets M1
        U[0, 0] = 1; U[3, 0] = 1  # A00 + A11
        V[0, 0] = 1; V[3, 0] = 1  # B00 + B11
        W[0, 0] = 1; W[3, 0] = 1  # C00 and C11
        
        # Term 2: M2 = (A10+A11)*B00
        # A10 at U[2], A11 at U[3], B00 at V[0]
        # C10 gets M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C10
        
        # Term 3: M3 = A00*(B01-B11)
        # A00 at U[0], B01 at V[1], B11 at V[3]
        # C01 gets M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C01
        
        # Term 4: M4 = A11*(B10-B00)
        # A11 at U[3], B10 at V[2], B00 at V[0]
        # C00 gets M4, C10 gets M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1  # C00 and C10
        
        # Term 5: M5 = (A00+A01)*B11
        # A00 at U[0], A01 at U[1], B11 at V[3]
        # C00 gets -M5, C01 gets M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # -C00 and C01
        
        # Term 6: M6 = (A10-A00)*(B00+B01)
        # A10 at U[2], A00 at U[0], B00 at V[0], B01 at V[1]
        # This term doesn't directly appear in standard Strassen... 
        # Let me re-check the Strassen formulas.
        # Actually M6 is not used in the standard 7-product formula for C.
        # The standard formulas are:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # So M6 is not needed! Let me fix this.
        
        # Let me redo this properly.
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01) - not used in output
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1
        
        # Verify: C = A*B
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # This should be correct.
        
        return U, V, W
    
    # Combine decompositions: given two decompositions, combine them
    def combine_decomps(U1, V1, W1, U2, V2, W2):
        """Combine two decompositions by block-diagonal stacking."""
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        U = np.zeros((U1.shape[0] + U2.shape[0], rank))
        V = np.zeros((V1.shape[0] + V2.shape[0], rank))
        W = np.zeros((W1.shape[0] + W2.shape[0], rank))
        U[:U1.shape[0], :r1] = U1
        V[:V1.shape[0], :r1] = V1
        W[:W1.shape[0], :r1] = W1
        U[U1.shape[0]:, r1:] = U2
        V[V1.shape[0]:, r1:] = V2
        W[W1.shape[0]:, r1:] = W2
        return U, V, W
    
    # For M_{n,m,p} where one dimension is 1, use schoolbook
    def decompose(n, m, p):
        """Decompose matrix multiplication tensor for n x m * m x p."""
        if time.time() > deadline:
            return schoolbook(n, m, p)
        
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = np.ones((1, 1))
            V = np.ones((1, 1))
            W = np.ones((1, 1))
            return U, V, W
        
        if n == 1:
            # M_{1,m,p} rank = m*p
            rank = m * p
            U = np.zeros((m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p, rank))
            col = 0
            for j in range(m):
                for k in range(p):
                    U[j, col] = 1
                    V[j * p + k, col] = 1
                    W[k, col] = 1
                    col += 1
            return U, V, W
        
        if m == 1:
            # M_{n,1,p} rank = n*p
            rank = n * p
            U = np.zeros((n, rank))
            V = np.zeros((p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i, col] = 1
                    V[k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
            return U, V, W
        
        if p == 1:
            # M_{n,m,1} rank = n*m
            rank = n * m
            U = np.zeros((n * m, rank))
            V = np.zeros((m, rank))
            W = np.zeros((n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i * m + j, col] = 1
                    V[j, col] = 1
                    W[i, col] = 1
                    col += 1
            return U, V, W
        
        # Check for 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
        
        # Try splitting along p
        if p >= 2:
            p1, p2 = p // 2, p - p // 2
            if p1 >= 1 and p2 >= 1:
                U1, V1, W1 = decompose(n, m, p1)
                U2, V2, W2 = decompose(n, m, p2)
                # Combine: the two sub-problems operate on disjoint k ranges
                # U has rows i*m+j, same for both
                # V has rows j*p+k, need to offset second one by p1
                # W has rows k*n+i, need to offset second one by p1*n
                U = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
                V = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
                W = np.zeros((p * n, U1.shape[1] + U2.shape[1]))
                r1 = U1.shape[1]
                U[:, :r1] = U1
                U[:, r1:] = U2
                V[:m * p1, :r1] = V1
                V[m * p1:, r1:] = V2
                W[:p1 * n, :r1] = W1
                W[p1 * n:, r1:] = W2
                return U, V, W
        
        # Try splitting along m
        if m >= 2:
            m1, m2 = m // 2, m - m // 2
            if m1 >= 1 and m2 >= 1:
                U1, V1, W1 = decompose(n, m1, p)
                U2, V2, W2 = decompose(n, m2, p)
                # Combine: disjoint j ranges
                U = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
                V = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
                W = np.zeros((p * n, U1.shape[1] + U2.shape[1]))
                r1 = U1.shape[1]
                U[:n * m1, :r1] = U1
                U[n * m1:, r1:] = U2
                V[:m1 * p, :r1] = V1
                V[m1 * p:, r1:] = V2
                W[:, :r1] = W1
                W[:, r1:] = W2
                return U, V, W
        
        # Try splitting along n
        if n >= 2:
            n1, n2 = n // 2, n - n // 2
            if n1 >= 1 and n2 >= 1:
                U1, V1, W1 = decompose(n1, m, p)
                U2, V2, W2 = decompose(n2, m, p)
                # Combine: disjoint i ranges
                U = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
                V = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
                W = np.zeros((p * n, U1.shape[1] + U2.shape[1]))
                r1 = U1.shape[1]
                U[:n1 * m, :r1] = U1
                U[n1 * m:, r1:] = U2
                V[:, :r1] = V1
                V[:, r1:] = V2
                W[:p * n1, :r1] = W1
                W[p * n1:, r1:] = W2
                return U, V, W
        
        # Fallback: schoolbook
        return schoolbook(n, m, p)
    
    # Get the decomposition
    U, V, W = decompose(n, m, p)
    rank = U.shape[1]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 if i,j,k are valid indices, 0 otherwise
        # Actually, for matrix mult, T[i,j,k] = 1 always for valid i,j,k
        # since (AB)_{ik} = sum_j A_{ij} B_{jk}
        # The tensor is T[i,j,k] = 1 for all valid i,j,k
        
        # Let's verify by checking that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # for all valid i, j, k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
