import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T for (n x m) @ (m x p)
    # T[a, b, c] where:
    #   a = i*m + j (row of A, col of A) -> i in [0,n), j in [0,m)
    #   b = j*p + k (row of B, col of B) -> j in [0,m), k in [0,p)
    #   c = k*n + i (row of C, col of C) -> k in [0,p), i in [0,n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We seek a rank-r decomposition:
    # T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    
    # Strategy: Use the fact that for small dimensions, we can try
    # structured decompositions. 
    
    # For 2x4 * 4x8, let's try to find a rank decomposition.
    # The trivial rank is 64 (n*m*p = 2*4*8 = 64).
    # The lower bound is max(n*m, m*p, n*p) = max(8, 32, 16) = 32.
    
    # Let's try a structured approach based on Kronecker products or
    # block decompositions.
    
    # One approach: if we can write the multiplication as a sum of
    # rank-1 terms using the structure of the indices.
    
    # Let's try to find a rank-r decomposition by solving a system
    # of equations. For small r, we can use random initialization and
    # gradient descent, but for exact solutions, we need a different approach.
    
    # Let's try a specific structured decomposition.
    # For 2x4 * 4x8, we can try to use the fact that 4 = 2*2 and 8 = 2*4.
    
    # Let's attempt a rank-32 decomposition (the lower bound).
    # We'll use a structured approach:
    
    # The tensor can be viewed as a sum over j (the shared dimension):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = A[i,j] * B[j,k]
    
    # This suggests that we can factor through j. But that's the standard
    # decomposition with rank m = 4, which doesn't work directly because
    # the tensor entries are 1, not A[i,j]*B[j,k].
    
    # Let me reconsider. The tensor T is defined as:
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # This is the matrix multiplication tensor.
    
    # For the matrix multiplication tensor, the rank is known to be at most
    # n*m*p (trivial) and at least max(n*m, m*p, n*p).
    
    # Let's try to find a rank-r decomposition for r < 64.
    # We'll use a randomized approach with verification.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W give an exact decomposition of the matmul tensor."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        # Check that T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # for all a,b,c
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        #            0 otherwise
        
        # Build the tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Compute the decomposition
        # T_decomp[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # This is a tensor of shape (n*m, m*p, p*n)
        # We can compute it as: for each t, outer product of U[:,t], V[:,t], W[:,t]
        
        # To avoid building the full tensor, let's check entry by entry
        # But that's O(n*m*m*p*p*n*rank) which might be slow.
        
        # Alternative: check that for all i,j,k:
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        # And for all other (a,b,c) not of that form, the sum = 0.
        
        # Check the 1 entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check some of the 0 entries (not all, for speed)
        # The total number of entries is n*m * m*p * p*n = (n*m*p)^2
        # For 2,4,8: 64^2 = 4096 entries, of which 64 are 1 and 4032 are 0.
        # Let's check all 0 entries for correctness.
        one_entries = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    one_entries.add((a, b, c))
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in one_entries:
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > 1e-9:
                            return False
        
        return True
    
    # Try to find a rank-r decomposition for decreasing r
    # Start from a reasonable upper bound and go down
    
    # For now, let's use the trivial rank-64 decomposition as a fallback
    # and try to find something better.
    
    # Let's try a structured approach:
    # The matmul tensor for (n,m,p) can be decomposed using the
    # "Strassen-like" approach if we group indices.
    
    # For 2x4 * 4x8, let's try to find a rank-32 decomposition.
    # We'll set up the problem as a linear system.
    
    # For rank r, we have r unknowns per entry of U, V, W.
    # Total unknowns: r * (n*m + m*p + p*n) = r * (8 + 32 + 16) = 56*r
    # Constraints: 64 equations (one for each 1 entry) + 4032 equations (for 0 entries)
    # But the 0 entry constraints are homogeneous, so they're easier.
    
    # Let's try a different approach: use the fact that we can fix
    # some entries to 0 or 1 and solve for the rest.
    
    # For the trivial decomposition, we have 64 rank-1 terms.
    # Can we combine them?
    
    # Let's try to find a rank-r decomposition by solving a least-squares
    # problem and then rounding, but that's not exact.
    
    # For exact decomposition, let's try a specific structured form.
    
    # Idea: Use the fact that 2*4*8 = 64, and try to find a rank-32
    # decomposition by pairing up the 64 terms.
    
    # For each pair (i1,j1,k1) and (i2,j2,k2), can we combine them?
    # T1[a,b,c] = delta(a,i1*m+j1) * delta(b,j1*p+k1) * delta(c,k1*n+i1)
    # T2[a,b,c] = delta(a,i2*m+j2) * delta(b,j2*p+k2) * delta(c,k2*n+i2)
    # T1 + T2 = U[a]*V[b]*W[c] for some U,V,W?
    # This would require U,V,W to be non-zero on the union of supports.
    
    # This is getting complex. Let's just return the trivial decomposition
    # for now, but with a cleaner implementation.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
