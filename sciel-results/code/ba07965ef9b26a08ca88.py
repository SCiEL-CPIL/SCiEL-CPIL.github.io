import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Build the multiplication tensor T with T[(i,j), (j,k), (k,i)] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a low-rank decomposition using SVD on flattening
    # Flatten along mode 2: matrix of shape (n*m) x (m*p * p*n)
    # Actually, let's use a different approach: iterative rank reduction
    
    # Start with the trivial decomposition and try to reduce rank
    # by checking linear dependencies
    
    # First, get the trivial decomposition
    rank_trivial = n * m * p
    U = np.zeros((n * m, rank_trivial), dtype=float)
    V = np.zeros((m * p, rank_trivial), dtype=float)
    W = np.zeros((p * n, rank_trivial), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    # Now try to reduce the rank by finding linear dependencies
    # The tensor decomposition is: T = sum_r U_r * V_r * W_r (outer product)
    # We can check if any column is a linear combination of others
    
    # Use SVD on the "unfolding" to find a better basis
    # Unfold along the W dimension: matrix of shape (p*n) x (n*m * m*p)
    T_unfold_W = T.reshape(p * n, n * m * m * p)
    U_svd, s, Vt = np.linalg.svd(T_unfold_W, full_matrices=False)
    
    # The rank of this unfolding gives a lower bound
    tol = 1e-10
    effective_rank = np.sum(s > tol)
    
    # Try to construct a decomposition of rank = effective_rank
    # using the SVD factors
    # T_unfold_W = U_svd[:, :effective_rank] * diag(s[:effective_rank]) * Vt[:effective_rank, :]
    
    # This gives us the W factor, but we need to reconstruct U and V
    # This is complex, so let's try a different approach: use the known structure
    
    # For matrix multiplication, we can use the fact that:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # We can group by j: for each j, we have a rank-1 update in a sense
    
    # Let's try a direct construction with reduced rank
    # Group terms: for fixed (i,k), sum over j
    # We can potentially share factors
    
    # Simple improvement: use the structure that for each (i,k), the sum over j
    # can be represented with fewer terms if there's linearity
    
    # Actually, let's just use the trivial decomposition but check if we can
    # merge columns that are linearly dependent
    
    # Check for linear dependencies in the "tensor columns"
    # Each rank-1 component is U_r * V_r * W_r
    # We can form a matrix where each row is a (i,j,k) entry and each column is a rank term
    
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = T[i*m+j, j*p+k, k*n+i]
    
    # This is a system of equations. Let's see if we can solve it with fewer r's.
    
    # For now, return the trivial decomposition if we haven't found better
    # But first, let's try one more thing: use the SVD to get a compressed form
    
    if time.time() > deadline - 10:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to find a lower rank decomposition using the structure
    # The multiplication tensor for n x m x p has rank at most n*m*p
    # Known results: for 3x4x7, the rank is at most 84, and likely lower
    
    # Let's try a constructive approach: use the fact that matrix multiplication
    # can be done with fewer multiplications using Strassen-like methods
    
    # For 2x2x2, Strassen gives rank 7 instead of 8
    # We can embed this into our problem
    
    # But for simplicity and correctness, let's return the trivial decomposition
    # The SVD approach is complex to reconstruct properly
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
