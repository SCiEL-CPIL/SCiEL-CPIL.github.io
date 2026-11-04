"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def schoolbook(n, m, p):
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

    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # The tensor T[i,j,k2,i2,j2,k] should be 1 if i==i2 and j==j2 and k==k2, else 0
        # Using the convention: U indexed by (i*m+j), V by (j2*p+k), W by (k2*n+i2)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        for k2 in range(p):
                            for i2 in range(n):
                                c = k2 * n + i2
                                val = 0.0
                                for cc in range(rank):
                                    val += U[a, cc] * V[b, cc] * W[c, cc]
                                if i == i2 and j == j2 and k == k2:
                                    expected = 1.0
                                else:
                                    expected = 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    def winograd_2x5x5():
        """
        Winograd algorithm for 2x5 * 5x5 multiplication.
        This uses the fact that we can group columns of A and rows of B.
        
        For 2x5 * 5x5, we can use Winograd's technique on the 5-column dimension.
        We'll split the 5 columns into pairs and use a 2x2 Winograd-like transform.
        
        Actually, let me use a simpler approach: use the known result that
        2x5x5 can be done in rank 48 (from Winograd's work on small matrix mult).
        
        For now, let's implement a basic Winograd-style reduction.
        """
        # For 2x5 * 5x5, schoolbook is 50.
        # We can potentially save 2 multiplications using Winograd's technique.
        # 
        # One approach: treat the 5 columns as [c0, c1, c2, c3, c4]
        # Use the identity that (a+b)(c+d) = ac + ad + bc + bd can be computed
        # with 3 multiplications instead of 4 if we have the right structure.
        #
        # For 2x5 * 5x5, we can pair up some of the 5 columns.
        # Let's use the following decomposition:
        # 
        # Actually, let me just use a known result: the rank of 2x5x5 is at most 48.
        # I'll construct it using a combination of schoolbook and Winograd steps.
        
        # For simplicity and correctness, let me implement a direct construction
        # that achieves rank 48 for 2x5x5.
        
        # The key insight: for multiplying a 2x5 matrix by a 5x5 matrix,
        # we can use the following trick on pairs of columns.
        
        # Let A = [a0 a1 a2 a3 a4] (2x5) and B = [b0 b1 b2 b3 b4] (5x5, each bi is 5x1)
        # C = A * B = sum_{j=0}^{4} a_j * b_j
        
        # We can group: (a0+b0) * (b0+b4) gives us terms involving a0b0, a0b4, a4b0, a4b4
        # But this doesn't directly help unless we have symmetry.
        
        # Let me use a different approach: use the fact that we can reduce the rank
        # by finding linear dependencies in the tensor.
        
        # For now, return schoolbook for 2x5x5 (rank 50) and combine two of them
        # to get rank 100 for 4x5x5. This is the baseline.
        
        rank = 50
        U = np.zeros((2*5, rank))
        V = np.zeros((5*5, rank))
        W = np.zeros((5*2, rank))
        col = 0
        for i in range(2):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W

    def combine_blocks(U1, V1, W1, U2, V2, W2, n, m, p):
        """
        Combine two block decompositions for 2x5x5 into one for 4x5x5.
        Block 1 handles rows 0,1 of the 4x5 matrix.
        Block 2 handles rows 2,3 of the 4x5 matrix.
        """
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Block 1: rows 0,1
        for i in range(2):
            for j in range(m):
                U[i*m+j, 0:rank1] = U1[i*m+j, :]
            for k in range(p):
                W[k*n+i, 0:rank1] = W1[k*2+i, :]
        for j in range(m):
            for k in range(p):
                V[j*p+k, 0:rank1] = V1[j*p+k, :]
        
        # Block 2: rows 2,3
        for i in range(2, 4):
            for j in range(m):
                U[i*m+j, rank1:total_rank] = U2[(i-2)*m+j, :]
            for k in range(p):
                W[k*n+i, rank1:total_rank] = W2[k*2+(i-2), :]
        for j in range(m):
            for k in range(p):
                V[j*p+k, rank1:total_rank] = V2[j*p+k, :]
        
        return U, V, W

    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    
    # Try Winograd-style reduction
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    if time.time() < deadline - 30:
        # Create two 2x5x5 decompositions
        U1, V1, W1 = winograd_2x5x5()
        U2, V2, W2 = winograd_2x5x5()
        
        # Combine them
        U_comb, V_comb, W_comb = combine_blocks(U1, V1, W1, U2, V2, W2, n, m, p)
        
        # Verify
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            best_U, best_V, best_W = U_comb, V_comb, W_comb
            best_rank = U_comb.shape[1]
    
    # Return the best result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
