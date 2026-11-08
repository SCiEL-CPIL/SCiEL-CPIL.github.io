import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a recursive Strassen-like approach.
    
    The function `build(n, m, p)` returns (U, V, W) for n x m x p multiplication.
    Base cases use schoolbook or small Strassen.
    """
    
    def build(n, m, p):
        """
        Returns U, V, W for n x m x p.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1]])
            V = np.array([[1]])
            W = np.array([[1]])
            return U, V, W
            
        # Try splitting n
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            # Recursively solve for (n1, m, p) and (n2, m, p)
            U1, V1, W1 = build(n1, m, p)
            U2, V2, W2 = build(n2, m, p)
            
            # Combine
            # U indices: 0..n1*m-1 for U1, n1*m..n*m-1 for U2
            # V indices: same for both
            # W indices: 0..p*n1-1 for W1, p*n1..p*n-1 for W2
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # Fill U1
            for r in range(rank1):
                for i in range(n1):
                    for j in range(m):
                        U[i*m + j, r] = U1[i*m + j, r]
                        
            # Fill V1
            for r in range(rank1):
                for j in range(m):
                    for k in range(p):
                        V[j*p + k, r] = V1[j*p + k, r]
                        
            # Fill W1
            for r in range(rank1):
                for k in range(p):
                    for i in range(n1):
                        W[k*n + i, r] = W1[k*n1 + i, r] # Note: W1 is for n1, so index is k*n1+i? 
                        # Wait, W1 is shape (p*n1, rank1). Index in W1 is k*n1 + i.
                        # Global W index is k*n + i.
                        # This is correct.
            
            # Fill U2
            for r in range(rank2):
                for i in range(n2):
                    for j in range(m):
                        U[(n1 + i)*m + j, rank1 + r] = U2[i*m + j, r]
                        
            # Fill V2
            for r in range(rank2):
                for j in range(m):
                    for k in range(p):
                        V[j*p + k, rank1 + r] = V2[j*p + k, r]
                        
            # Fill W2
            for r in range(rank2):
                for k in range(p):
                    for i in range(n2):
                        W[k*n + (n1 + i), rank1 + r] = W2[k*n2 + i, r]
                        
            return U, V, W

        # Try splitting m
        if m > 1:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = build(n, m1, p)
            U2, V2, W2 = build(n, m2, p)
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # U1: A[:, :m1]
            for r in range(rank1):
                for i in range(n):
                    for j in range(m1):
                        U[i*m + j, r] = U1[i*m1 + j, r]
                        
            # V1: B[:m1, :]
            for r in range(rank1):
                for j in range(m1):
                    for k in range(p):
                        V[j*p + k, r] = V1[j*p + k, r]
                        
            # W1: C
            for r in range(rank1):
                for k in range(p):
                    for i in range(n):
                        W[k*n + i, r] = W1[k*n + i, r]
            
            # U2: A[:, m1:]
            for r in range(rank2):
                for i in range(n):
                    for j in range(m2):
                        U[i*m + (m1 + j), rank1 + r] = U2[i*m2 + j, r]
                        
            # V2: B[m1:, :]
            for r in range(rank2):
                for j in range(m2):
                    for k in range(p):
                        V[(m1 + j)*p + k, rank1 + r] = V2[j*p + k, r]
                        
            # W2: C
            for r in range(rank2):
                for k in range(p):
                    for i in range(n):
                        W[k*n + i, rank1 + r] = W2[k*n + i, r]
                        
            return U, V, W

        # Try splitting p
        if p > 1:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = build(n, m, p1)
            U2, V2, W2 = build(n, m, p2)
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # U1: A
            for r in range(rank1):
                for i in range(n):
                    for j in range(m):
                        U[i*m + j, r] = U1[i*m + j, r]
                        
            # V1: B[:, :p1]
            for r in range(rank1):
                for j in range(m):
                    for k in range(p1):
                        V[j*p + k, r] = V1[j*p1 + k, r]
                        
            # W1: C[:, :p1]
            for r in range(rank1):
                for k in range(p1):
                    for i in range(n):
                        W[k*n + i, r] = W1[k*n + i, r]
            
            # U2: A
            for r in range(rank2):
                for i in range(n):
                    for j in range(m):
                        U[i*m + j, rank1 + r] = U2[i*m + j, r]
                        
            # V2: B[:, p1:]
            for r in range(rank2):
                for j in range(m):
                    for k in range(p2):
                        V[j*p + (p1 + k), rank1 + r] = V2[j*p2 + k, r]
                        
            # W2: C[:, p1:]
            for r in range(rank2):
                for k in range(p2):
                    for i in range(n):
                        W[(p1 + k)*n + i, rank1 + r] = W2[k*n + i, r]
                        
            return U, V, W

        # Fallback: Schoolbook
        # This should not be reached for n,m,p > 1 if we split
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        return U, V, W

    # Use Strassen for 2x2x2 to improve base case
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1; V[2, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
        return U, V, W

    # Modified build to use Strassen for 2x2x2
    def build_opt(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        if n == 1 and m == 1 and p == 1:
            return np.array([[1]]), np.array([[1]]), np.array([[1]])
            
        # Simple recursive split
        # We can choose the split that minimizes rank, but for now just split n, then m, then p
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = build_opt(n1, m, p)
            U2, V2, W2 = build_opt(n2, m, p)
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for r in range(rank1):
                for i in range(n1):
                    for j in range(m):
                        U[i*m + j, r] = U1[i*m + j, r]
                for j in range(m):
                    for k in range(p):
                        V[j*p + k, r] = V1[j*p + k, r]
                for k in range(p):
                    for i in range(n1):
                        W[k*n + i, r] = W1[k*n1 + i, r]
            
            for r in range(rank2):
                for i in range(n2):
                    for j in range(m):
                        U[(n1 + i)*m + j, rank1 + r] = U2[i*m + j, r]
                for j in range(m):
                    for k in range(p):
                        V[j*p + k, rank1 + r] = V2[j*p + k, r]
                for k in range(p):
                    for i in range(n2):
                        W[k*n + (n1 + i), rank1 + r] = W2[k*n2 + i, r]
                        
            return U, V, W

        if m > 1:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = build_opt(n, m1, p)
            U2, V2, W2 = build_opt(n, m2, p)
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for r in range(rank1):
                for i in range(n):
                    for j in range(m1):
                        U[i*m + j, r] = U1[i*m1 + j, r]
                for j in range(m1):
                    for k in range(p):
                        V[j*p + k, r] = V1[j*p + k, r]
                for k in range(p):
                    for i in range(n):
                        W[k*n + i, r] = W1[k*n + i, r]
            
            for r in range(rank2):
                for i in range(n):
                    for j in range(m2):
                        U[i*m + (m1 + j), rank1 + r] = U2[i*m2 + j, r]
                for j in range(m2):
                    for k in range(p):
                        V[(m1 + j)*p + k, rank1 + r] = V2[j*p + k, r]
                for k in range(p):
                    for i in range(n):
                        W[k*n + i, rank1 + r] = W2[k*n + i, r]
                        
            return U, V, W

        if p > 1:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = build_opt(n, m, p1)
            U2, V2, W2 = build_opt(n, m, p2)
            
            rank1 = U1.shape[1]
            rank2 = U2.shape[1]
            total_rank = rank1 + rank2
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for r in range(rank1):
                for i in range(n):
                    for j in range(m):
                        U[i*m + j, r] = U1[i*m + j, r]
                for j in range(m):
                    for k in range(p1):
                        V[j*p + k, r] = V1[j*p1 + k, r]
                for k in range(p1):
                    for i in range(n):
                        W[k*n + i, r] = W1[k*n + i, r]
            
            for r in range(rank2):
                for i in range(n):
                    for j in range(m):
                        U[i*m + j, rank1 + r] = U2[i*m + j, r]
                for j in range(m):
                    for k in range(p2):
                        V[j*p + (p1 + k), rank1 + r] = V2[j*p2 + k, r]
                for k in range(p2):
                    for i in range(n):
                        W[(p1 + k)*n + i, rank1 + r] = W2[k*n + i, r]
                        
            return U, V, W

        # Schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        return U, V, W

    U, V, W = build_opt(3, 4, 5)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
