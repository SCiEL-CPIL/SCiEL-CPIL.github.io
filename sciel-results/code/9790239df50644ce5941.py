"""Rewritten program for matrix multiplication tensor decomposition (2x4 * 4x4)."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x4 matrix.
    
    The multiplication tensor T is of size (n*m) x (m*p) x (p*n) = 8 x 16 x 8.
    We want to find matrices U (8 x r), V (16 x r), W (8 x r) such that:
    T[a, b, c] = sum_{t=1}^{r} U[a,t] * V[b,t] * W[c,t]
    
    where a indexes (i,j) with i in [0,n), j in [0,m), so a = i*m + j
          b indexes (j,k) with j in [0,m), k in [0,p), so b = j*p + k
          c indexes (k,i) with k in [0,p), i in [0,n), so c = k*n + i
    
    This is a 3-way tensor with dimensions 8 x 16 x 8.
    
    The schoolbook rank is 8*16*8... wait, no. The schoolbook for matrix mult A(n x m) * B(m x p):
    rank = n*m*p = 2*4*4 = 32.
    
    For this specific case (2x4 * 4x4), we can potentially do better than 32.
    The known best ranks for small matrix multiplications are documented in literature.
    For 2x4 * 4x4, let's think about what's achievable.
    
    Actually, let me reconsider. The tensor dimensions are:
    - First mode: n*m = 8 (input A entries)
    - Second mode: m*p = 16 (input B entries) 
    - Third mode: p*n = 8 (output C entries)
    
    The standard schoolbook gives rank 32.
    
    Let me try to construct a decomposition with rank less than 32 by exploiting structure.
    
    One approach: use the fact that the matrix multiplication tensor has special structure.
    We can use the "baby step giant step" or other algebraic methods.
    
    For now, let me implement the schoolbook solution as a baseline, and then try to find
    a lower rank decomposition using random initialization and optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 32
    
    # Build the schoolbook solution first
    best_U, best_V, best_W, best_rank = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition using optimization
    # The tensor T has entries T[a,b,c] where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if the indices are consistent, 0 otherwise
    
    # Actually, for matrix multiplication, the tensor is defined as:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # All other entries are 0.
    
    # Let's build the tensor explicitly
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try various ranks below schoolbook
    # For rank r, we need to solve: T = sum_t U_t (n*m) x V_t (m*p) x W_t (p*n)
    # This is a system of (n*m)*(m*p)*(p*n) = 8*16*8 = 1024 equations
    # with r*(n*m + m*p + p*n) = r*(8+16+8) = 32r unknowns
    
    # Let's try to find a decomposition with rank < 32
    # We'll use a simple approach: initialize with random tensors and use ALS or similar
    
    # Try ranks from 24 to 31
    for target_rank in range(24, 32):
        if time.time() > deadline - 5:
            break
        result = _try_rank_decomposition(T, target_rank, n, m, p, time_budget=10)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, T, n, m, p):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break  # Found the smallest rank in our search range
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
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
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, T, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reproduces the tensor T."""
    # T_reconstructed[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # Use einsum for efficiency
    T_rec = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(T_rec, T, atol=tol, rtol=0)

def _try_rank_decomposition(T, target_rank, n, m, p, time_budget=10):
    """
    Try to find a rank-target_rank decomposition of tensor T using ALS.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Reshape T for ALS: T is (n*m, m*p, p*n)
    # For ALS, fix two matrices and solve for the third
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    
    # Precompute flattened tensor
    T_flat = T.reshape(-1, p*n)  # (n*m*m*p, p*n)
    
    # Try multiple random initializations
    for trial in range(5):
        if time.time() - start_time > time_budget - 1:
            break
        
        np.random.seed(42 + trial)
        
        # Initialize
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # ALS iterations
        for it in range(200):
            if time.time() - start_time > time_budget - 1:
                break
            
            # Update U: for each a, solve for U[a,:]
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # For fixed a: sum_t U[a,t] * (V[:,t] * W[c,t]) = T[a,:,c]
            # This is: U[a,:] @ (V.T * W[c,:]) = T[a,:,c] for each c
            # Better: reshape and solve linear system
            
            # For U update: fix V and W
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # Let M[b,c] = sum_t V[b,t] * W[c,t] = V @ W.T
            # Then T[a,b,c] = sum_t U[a,t] * (V @ W.T)[b,c]... no that's not right
            # Actually: T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # For fixed a: T[a,:,:] = sum_t U[a,t] * outer(V[:,t], W[:,t])
            # So: T[a,:,:].reshape(-1) = U[a,:] @ (V[:,t] * W[:,t]) stacked
            # Let's build the matrix: for each t, column is V[:,t] * W[:,t] (flattened)
            
            # Build matrix A where A[b*c + c', t] = V[b,t] * W[c',t]
            # Then T[a, :, :].reshape(-1) = U[a,:] @ A
            # A shape: (m*p * p*n, target_rank)
            
            # More efficiently: 
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # For fixed a, we have a linear system in U[a,:]:
            # For each (b,c): T[a,b,c] = sum_t U[a,t] * (V[b,t] * W[c,t])
            # So: T[a,:,:] = U[a,:] @ B where B[b*c+c', t] = V[b,t] * W[c,t]
            
            # Construct B: (m*p * p*n, target_rank)
            # B[bp*pn + c, t] = V[bp, t] * W[c, t]
            B = np.einsum('bt,ct->bct', V, W).reshape(m*p * p*n, target_rank)
            # T_flat_a: for each a, T[a,:,:].reshape(-1)
            T_reshaped = T.reshape(n*m, m*p * p*n)
            # Solve: T_reshaped @ B.T = U @ B @ B.T
            # U = T_reshaped @ B.T @ (B @ B.T)^{-1}
            try:
                BtB = B.T @ B
                BtT = B.T @ T_reshaped.T  # (target_rank, n*m)
                U = (np.linalg.solve(BtB, BtT)).T
            except np.linalg.LinAlgError:
                break
            
            # Update V: similar logic
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # For fixed b: T[:,b,c] = V[b,:] @ C where C[a*pn+c, t] = U[a,t] * W[c,t]
            C = np.einsum('at,ct->act', U, W).reshape(n*m * p*n, target_rank)
            T_for_V = np.zeros((m*p, n*m * p*n))
            for b in range(m*p):
                # T[:, b, :] has shape (n*m, p*n)
                T_for_V[b, :] = T[:, b, :].reshape(-1)
            try:
                CtC = C.T @ C
                CtT = C.T @ T_for_V.T
                V = (np.linalg.solve(CtC, CtT)).T
            except np.linalg.LinAlgError:
                break
            
            # Update W: similar
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            # For fixed c: T[:,:,c] = W[c,:] @ D where D[ab*mn+b', t] = U[a,t] * V[b',t]
            D = np.einsum('at,bt->abt', U, V).reshape(n*m * m*p, target_rank)
            T_for_W = np.zeros((p*n, n*m * m*p))
            for c in range(p*n):
                T_for_W[c, :] = T[:, :, c].reshape(-1)
            try:
                DtD = D.T @ D
                DtT = D.T @ T_for_W.T
                W = (np.linalg.solve(DtD, DtT)).T
            except np.linalg.LinAlgError:
                break
        
        # Check convergence
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        if np.allclose(T_rec, T, atol=1e-8, rtol=0):
            return U, V, W
    
    return None

# EVOLVE-BLOCK-END
