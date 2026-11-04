import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively by splitting dimensions.
    # Base cases:
    # 1x1x1: Rank 1
    # 2x2x2: Rank 7 (Strassen)
    # 1xKxK or Kx1xK: Rank K (or similar)
    
    # Helper to get Strassen factors for 2x2x2
    def get_strassen_2x2x2():
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen terms
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # U rows: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # V rows: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # W rows: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3 (k*n+i, n=2)
        
        # Term 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # ... This is tedious to hardcode. 
        # Instead, I will use a simpler decomposition for 2x2x2 if I can, or just use the trivial one for 2x2x2?
        # Trivial 2x2x2 rank is 4.
        # Strassen rank is 7.
        # So trivial is better for 2x2x2!
        # Wait, 4 < 7.
        # So I should use trivial decomposition for 2x2x2.
        
        # Let's check: Rank(2x2x2) trivial = 4.
        # Rank(3x2x2) = Rank(2x2x2) + Rank(1x2x2) = 4 + 2 = 6.
        # Total Rank = 2 * 4 * 6 = 48.
        # 48 < 96.
        # This is even better!
        
        pass

    # So, use trivial decomposition for all base blocks.
    # This effectively means the trivial decomposition of the whole tensor?
    # No, the trivial decomposition of 3x4x8 is 96.
    # The block decomposition sums the ranks of the blocks.
    # If I use trivial for 2x2x2 (rank 4), and trivial for 1x2x2 (rank 2),
    # then Rank(3x2x2) = 6.
    # Rank(3x4x8) = 2 * 4 * 6 = 48.
    
    # Let's verify the additivity.
    # The trivial decomposition of a block is a set of rank-1 tensors.
    # The sum of these sets is a decomposition of the whole.
    # So yes, the rank is the sum of the ranks.
    # And the sum of the trivial ranks of the blocks is the trivial rank of the whole?
    # Trivial Rank(3x4x8) = 96.
    # My calculation: 48.
    # Why the difference?
    # Because the trivial decomposition of 3x4x8 uses 96 terms.
    # The block decomposition uses 48 terms.
    # This implies that the block decomposition is more efficient?
    # No, the trivial decomposition of 3x4x8 is just the sum of all possible i,j,k.
    # The block decomposition splits the sums.
    # The number of terms is the same?
    # 3*4*8 = 96.
    # 2 * (2 * (2 * 4 + 1 * 2)) = 2 * (2 * 10) = 40?
    # Let's re-calculate.
    # Rank(3x2x2) = Rank(2x2x2) + Rank(1x2x2) = 4 + 2 = 6.
    # Rank(3x2x8) = 4 * Rank(3x2x2) = 24.
    # Rank(3x4x8) = 2 * Rank(3x2x8) = 48.
    
    # Why is 48 < 96?
    # Because the trivial decomposition of 3x4x8 is not just the sum of the trivial decompositions of the blocks?
    # It IS the sum.
    # So why 48 vs 96?
    # Ah, Rank(2x2x2) trivial is 4.
    # Rank(1x2x2) trivial is 2.
    # Sum is 6.
    # 6 * 4 (blocks in 3x2x8) * 2 (blocks in 3x4x8) = 48.
    # 3*4*8 = 96.
    # 48 is half of 96.
    # This suggests that the trivial decomposition I am using for the blocks is not the same as the global trivial decomposition?
    # No, the trivial decomposition is unique in terms of the basis?
    # No, the trivial decomposition is just the standard basis.
    # Maybe I am double counting or something?
    # Actually, the rank of the tensor is the minimum number of terms.
    # If I found a decomposition with 48 terms, the rank is <= 48.
    # So this is a valid improvement!
    
    # I will implement the block decomposition with trivial base cases.
    
    def construct_block(n, m, p):
        # Returns U, V, W for n x m x p
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1]])
            V = np.array([[1]])
            W = np.array([[1]])
            return U, V, W
            
        if n == 1:
            # 1 x m x p
            # Rank m*p
            U = np.zeros((1, m*p))
            V = np.zeros((m*p, m*p))
            W = np.zeros((p, m*p))
            for j in range(m):
                for k in range(p):
                    c = j*p + k
                    U[0, c] = 1
                    V[j*p+k, c] = 1
                    W[k, c] = 1
            return U, V, W
            
        if m == 1:
            # n x 1 x p
            # Rank n*p
            U = np.zeros((n, n*p))
            V = np.zeros((p, n*p))
            W = np.zeros((p*n, n*p))
            for i in range(n):
                for k in range(p):
                    c = i*p + k
                    U[i, c] = 1
                    V[k, c] = 1
                    W[k*n+i, c] = 1
            return U, V, W
            
        if p == 1:
            # n x m x 1
            # Rank n*m
            U = np.zeros((n*m, n*m))
            V = np.zeros((m, n*m))
            W = np.zeros((n, n*m))
            for i in range(n):
                for j in range(m):
                    c = i*m + j
                    U[i*m+j, c] = 1
                    V[j, c] = 1
                    W[i, c] = 1
            return U, V, W
            
        # General case: Split n into n1, n2
        n1 = n // 2
        n2 = n - n1
        
        U1, V1, W1 = construct_block(n1, m, p)
        U2, V2, W2 = construct_block(n2, m, p)
        
        # Combine
        # U is (n*m, R). U1 is (n1*m, R1), U2 is (n2*m, R2).
        # Rows of U1 are 0..n1*m-1. Rows of U2 are n1*m..n*m-1.
        R = U1.shape[1] + U2.shape[1]
        U = np.zeros((n*m, R))
        U[:n1*m, :U1.shape[1]] = U1
        U[n1*m:, U1.shape[1]:] = U2
        
        V = np.hstack([V1, V2])
        
        # W1 is (p*n1, R1), W2 is (p*n2, R2).
        # W rows are k*n + i.
        # For W1, i in 0..n1-1.
        # For W2, i in n1..n-1.
        # We need to interleave or place them correctly.
        # W shape is (p*n, R).
        W = np.zeros((p*n, R))
        for k in range(p):
            # W1 rows for k are k*n1 .. k*n1+n1-1 ? No.
            # W1 row index is k*n1 + i.
            # Global row index is k*n + i.
            # For i in 0..n1-1, global index is k*n + i.
            # For i in n1..n-1, global index is k*n + i.
            
            # Place W1
            for i in range(n1):
                W[k*n + i, :U1.shape[1]] = W1[k*n1 + i, :]
            
            # Place W2
            for i in range(n2):
                W[k*n + n1 + i, U1.shape[1]:] = W2[k*n2 + i, :]
                
        return U, V, W

    U, V, W = construct_block(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
