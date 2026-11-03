import numpy as np

def construct():
    n, m, p = 3, 5, 6
    
    # Cache for decompositions to avoid redundant work? 
    # The sizes are small, so direct recursion is fine.
    
    def get_decomp(n, m, p):
        """
        Returns U, V, W as numpy arrays of shapes (n*m, rank), (m*p, rank), (p*n, rank)
        for the multiplication of n x m by m x p.
        """
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1]])
            V = np.array([[1]])
            W = np.array([[1]])
            return U, V, W
        
        # Base case: 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = np.zeros((4, rank))
            V = np.zeros((4, rank))
            W = np.zeros((4, rank))
            
            # Strassen indices
            # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            # A: 2x2, B: 2x2, C: 2x2
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            # M2 = (A10+A11)B00 -> C10 += M2
            # M3 = A00(B01-B11) -> C01 += M3
            # M4 = A11(B10-B00) -> C10 += M4
            # M5 = (A00+A01)B11 -> C11 += M5
            # M6 = (A10-A00)(B00+B01) -> C11 += M6
            # M7 = (A01-A11)(B10+B11) -> C00 += M7
            
            # Col 0: M1
            U[0,0] = 1; U[3,0] = 1
            V[0,0] = 1; V[3,0] = 1
            W[0,0] = 1; W[3,0] = 1
            
            # Col 1: M2
            U[2,1] = 1; U[3,1] = 1
            V[0,1] = 1
            W[2,1] = 1
            
            # Col 2: M3
            U[0,2] = 1
            V[1,2] = 1; V[3,2] = -1
            W[1,2] = 1
            
            # Col 3: M4
            U[3,3] = 1
            V[2,3] = 1; V[0,3] = -1
            W[2,3] = 1
            
            # Col 4: M5
            U[0,4] = 1; U[1,4] = 1
            V[3,4] = 1
            W[3,4] = 1
            
            # Col 5: M6
            U[2,5] = 1; U[0,5] = -1
            V[0,5] = 1; V[1,5] = 1
            W[3,5] = 1
            
            # Col 6: M7
            U[1,6] = 1; U[3,6] = -1
            V[2,6] = 1; V[3,6] = 1
            W[0,6] = 1
            
            return U, V, W

        # Recursive Case
        # Determine split dimension
        # We want to split a dimension into 2 and rest if possible
        # Priority: Split n, then m, then p.
        
        if n > 2:
            n1, n2 = 2, n - 2
            # Decompose n1 x m x p
            U1, V1, W1 = get_decomp(n1, m, p)
            # Decompose n2 x m x p
            U2, V2, W2 = get_decomp(n2, m, p)
            
            # Combine
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # Block 1: rows 0..n1-1 of A
            U[:n1*m, :rank1] = U1
            V[:, :rank1] = V1
            # W for block 1: rows k*n + i where i in 0..n1-1
            # W1 has shape (p*n1, rank1). We need to place it in W (p*n, total_rank)
            # For each k, the rows i=0..n1-1 correspond to global rows k*n .. k*n+n1-1
            for k in range(p):
                W[k*n:(k+1)*n, :rank1][:n1] = W1[k*n1:(k+1)*n1]
                
            # Block 2: rows n1..n-1 of A
            U[n1*m:, rank1:] = U2
            V[:, rank1:] = V2
            # W for block 2: rows k*n + i where i in 0..n2-1 (local)
            # Global i is n1..n-1
            for k in range(p):
                # Local rows 0..n2-1 of W2 correspond to global rows k*n + n1 .. k*n + n - 1
                W[k*n + n1:(k+1)*n, rank1:] = W2[k*n2:(k+1)*n2]
                
            return U, V, W

        elif m > 2:
            m1, m2 = 2, m - 2
            # Decompose n x m1 x p
            U1, V1, W1 = get_decomp(n, m1, p)
            # Decompose n x m2 x p
            U2, V2, W2 = get_decomp(n, m2, p)
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # Block 1: cols 0..m1-1 of A, rows 0..m1-1 of B
            U[:, :rank1] = U1
            V[:m1*p, :rank1] = V1
            W[:, :rank1] = W1
            
            # Block 2: cols m1..m-1 of A, rows m1..m-1 of B
            U[:, rank1:] = U2
            V[m1*p:, rank1:] = V2
            W[:, rank1:] = W2
            
            return U, V, W

        elif p > 2:
            p1, p2 = 2, p - 2
            # Decompose n x m x p1
            U1, V1, W1 = get_decomp(n, m, p1)
            # Decompose n x m x p2
            U2, V2, W2 = get_decomp(n, m, p2)
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # Block 1: cols 0..p1-1 of B, cols 0..p1-1 of C
            U[:, :rank1] = U1
            V[:, :rank1] = V1
            # W1 has shape (p1*n, rank1). Global W cols 0..p1-1?
            # W index is k*n+i. k is 0..p1-1.
            W[:p1*n, :rank1] = W1
            
            # Block 2: cols p1..p-1 of B, cols p1..p-1 of C
            U[:, rank1:] = U2
            V[:, rank1:] = V2
            # W2 has shape (p2*n, rank2). Global W cols p1..p-1?
            # k is p1..p-1. Local k is 0..p2-1.
            # Global row index: k*n + i = (p1 + k_loc)*n + i = p1*n + k_loc*n + i.
            W[p1*n:, rank1:] = W2
            
            return U, V, W

        else:
            # All dims <= 2, and not 2,2,2.
            # Must be schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            return U, V, W

    U, V, W = get_decomp(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
