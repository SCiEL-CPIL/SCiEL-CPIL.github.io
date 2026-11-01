import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a low-rank decomposition
    # The tensor is T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T is nonzero only when the j in a matches the j in b, and i,k are consistent with c
    
    # Let's build the tensor as a 3D array
    U_dim = n * m  # 4
    V_dim = m * p  # 28
    W_dim = p * n  # 28
    
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try different ranks using numerical SVD-like approach
    # For a 3-tensor, we can use the unfolding approach
    
    # Start with schoolbook rank = 56
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Build schoolbook decomposition as baseline
    rank = best_rank
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    # Verify schoolbook
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if U, V, W decompose T"""
        # T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We check against the original T
        for a in range(U_dim):
            for b in range(V_dim):
                for c in range(W_dim):
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    expected = T[a, b, c]
                    if abs(val - expected) > tol:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies
    # The tensor has structure: for each (i,k), we need T[i*m+j, j*p+k, k*n+i] = 1 for j=0,1
    # This means for fixed (i,k), the entries at positions (i*m+0, 0*p+k, k*n+i) and (i*m+1, 1*p+k, k*n+i) are 1
    
    # Let's try rank = 28 + 14 = 42 or similar
    # Actually, let me try a more systematic approach
    
    # For each (i,k) pair, we have two terms: A[i,0]*B[0,k] and A[i,1]*B[1,k]
    # Total: 2 * 2 * 14 = 56 terms
    
    # Can we merge any? If two terms have the same U, V, W pattern, we can combine.
    # But each term has unique (a,b,c) pattern, so no direct merging.
    
    # Let's try to use a bilinear trick:
    # For a given i, compute (A[i,0]+A[i,1]) * (B[0,k]+B[1,k]) for each k
    # This gives A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k]
    # We only want A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # So we'd need to subtract A[i,0]*B[1,k] + A[i,1]*B[0,k]
    # That's 2 extra multiplications, so total 3 per (i,k) instead of 2. Worse.
    
    # Another idea: use the fact that we have 2 rows in A and 2 columns in B's "inner" dimension
    # Maybe use a 2x2 Strassen-like decomposition for the "middle" part?
    
    # Actually, let me just try to numerically find a low-rank decomposition
    # using iterative methods with a time budget
    
    deadline = time.time() + 120  # 120 second deadline
    
    # Try ranks from low to high
    for test_rank in range(30, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Use a simple approach: try to find U, V, W of given rank
        # This is NP-hard in general, but for small tensors we can try
        
        # For rank r, we need U (4 x r), V (28 x r), W (28 x r)
        # such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # This is equivalent to: for each (a,b), the vector (T[a,b,:]) must be in
        # the span of {U[a,r]*V[b,r] for r=1..r} applied to W columns
        
        # Let's use a different approach: unfold the tensor
        # T_unfolded (W_dim x U_dim*V_dim) where T_unfolded[c, a*V_dim+b] = T[a,b,c]
        # Then T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Let M_r = U[:,r] * V[:,r]^T (outer product, U_dim x V_dim)
        # Then T_unfolded[c, :] = sum_r W[c,r] * vec(M_r)^T
        # So T_unfolded = W^T * (matrix with rows vec(M_r)^T)
        
        # This is getting complex. Let me try a simpler numerical approach.
        
        # Use the fact that for small tensors, we can set up a system of equations
        # For rank r, we have r * (4 + 28 + 28) = r * 60 unknowns
        # And 4 * 28 * 28 = 3136 equations (but only 56 are nonzero)
        
        # Actually, only 56 entries of T are nonzero (one for each (i,j,k) triple)
        # So we have 56 equations and 60r unknowns
        
        # For r = 56/2 = 28, we have 1680 unknowns and 56 equations - very underdetermined
        # But we also need to satisfy that the nonzero entries are exactly 1 and zeros are 0
        
        # Let me try a constructive approach for specific ranks
        
        pass
    
    # Given the complexity, let's stick with the schoolbook rank of 56
    # but try to see if we can do slightly better with a clever construction
    
    # Actually, let me try rank = 56 but verify it's correct
    result = verify(U, V, W)
    if result:
        # Convert to lists of integers/half-integers
        U_list = [[int(U[i,j]) if abs(U[i,j] - round(U[i,j])) < 1e-10 else U[i,j] for j in range(rank)] for i in range(U_dim)]
        V_list = [[int(V[i,j]) if abs(V[i,j] - round(V[i,j])) < 1e-10 else V[i,j] for j in range(rank)] for i in range(V_dim)]
        W_list = [[int(W[i,j]) if abs(W[i,j] - round(W[i,j])) < 1e-10 else W[i,j] for j in range(rank)] for i in range(W_dim)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook
    U_list = [[int(U[i,j]) if abs(U[i,j] - round(U[i,j])) < 1e-10 else U[i,j] for j in range(rank)] for i in range(U_dim)]
    V_list = [[int(V[i,j]) if abs(V[i,j] - round(V[i,j])) < 1e-10 else V[i,j] for j in range(rank)] for i in range(V_dim)]
    W_list = [[int(W[i,j]) if abs(W[i,j] - round(W[i,j])) < 1e-10 else W[i,j] for j in range(rank)] for i in range(W_dim)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
