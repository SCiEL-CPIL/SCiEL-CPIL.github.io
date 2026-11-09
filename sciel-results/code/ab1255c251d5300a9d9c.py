"""Tensor decomposition for matrix multiplication 3x3 * 3x8."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[:,c] V^T[:,c] W^T[:,c] = T where T[i*m+j, j*p+k, k*n+i] = 1."""
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_comp, T, atol=tol)

def _build_shared_form_decomposition(n, m, p, deadline):
    """
    Build a low-rank decomposition by sharing linear forms on the A-side
    across output columns.
    
    Strategy: For 3x3 * 3x8, we compute C = A*B where C is 3x8.
    For each output column k, C[:,k] = A * B[:,k].
    
    We can write: C[i,k] = sum_j A[i,j] * B[j,k]
    
    Key insight: We can define shared linear combinations of A entries
    that are reused across different k values.
    
    Construction: Use rank = 3*p + 3*m - 3 = 3*8 + 9 - 3 = 30
    This comes from the "rank of matrix multiplication" formula for
    rectangular cases where one dimension is 1 on the output side.
    
    Actually, let me use a direct optimization approach with smart initialization.
    """
    start_time = time.time()
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook
    
    # Try structured decomposition first
    # For 3x3*3x8: use the fact that we can share A-side forms
    # 
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Define for each i: a_i = (A[i,0], A[i,1], A[i,2]) as a row vector
    # Then C[i,k] = a_i dot B[:,k]
    #
    # We can write a_i as linear combinations of shared basis vectors.
    # 
    # Alternative: Use rank 3*p + 3*(m-1) = 24 + 6 = 30
    # 
    # Let's try: for each of the 3 rows of A, and for each of the 8 columns of B,
    # we need 24 products. But we can share.
    #
    # Actually, the simplest good construction:
    # For each k in 0..7: compute A * B[:,k] using 9 multiplications = 72 total
    # But we can do better.
    #
    # Let me try optimization with rank starting from 45 down.
    
    for rank in range(45, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            
            # Smart initialization: start from schoolbook and randomly perturb/merge
            np.random.seed(12345 + trial)
            
            # Initialize with random rank-r factors
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Alternating least squares optimization
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each a, U[a,:] = argmin ||T[a,:,:] - sum_r U[a,r] V[r,:] W[r,:]^T||
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V, W: T[a,:,:] = U[a,:] @ (V * W^T)^T ... 
                # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # So T[a,:,:] = U[a,:] @ (V ⊙ W) where ⊙ is element-wise in r dimension
                # Let M = V * W (broadcast), shape (m*p, p*n, rank) -> no
                # T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])^T reshaped
                # Define K[r] = outer(V[:,r], W[:,r]), shape (m*p, p*n)
                # T[a,:,:] = sum_r U[a,r] * K[r]
                # So: T[:,:,a]... no. T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # where ⊗ here means outer product
                # Let K be matrix of shape (rank, m*p*p*n) where K[r] = vec(outer(V[:,r], W[:,r]))
                # Then T_reshaped[a, :] = U[a, :] @ K
                # So U = T_reshaped @ pinv(K)
                
                # Build K: K[r, b*p*n + c] = V[b,r] * W[c,r]
                K = np.einsum('br,cr->bcr', V, W).reshape(rank, m * p * p * n)
                T_flat = T.reshape(n * m, m * p * p * n)
                
                try:
                    U_new = T_flat @ np.linalg.pinv(K)
                except:
                    break
                
                # Update V
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b: T[:,:,b]... T[a,b,c] = sum_r V[b,r] * (U[:,r] * W[:,r])[a] * W[c,r]... 
                # T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # For fixed b: T[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
                # Let K2[r] = vec(outer(U[:,r], W[:,r])), shape (n*m, p*n)
                # T[:,b,:] reshaped = V[b,:] @ K2
                K2 = np.einsum('ar,cr->acr', U, W).reshape(rank, n * m * p * n)
                T_flat2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                
                try:
                    V_new = T_flat2 @ np.linalg.pinv(K2)
                except:
                    break
                
                # Update W
                # T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
                # For fixed c: T[:,:,c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
                K3 = np.einsum('ar,br->abr', U, V).reshape(rank, n * m * m * p)
                T_flat3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                
                try:
                    W_new = T_flat3 @ np.linalg.pinv(K3)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                loss = np.sum((T - T_comp) ** 2)
                if loss < 1e-10:
                    found = True
                    break
            
            if found:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if found:
            break
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition
    better = _build_shared_form_decomposition(n, m, p, deadline)
    if better is not None:
        U, V, W = better
        if _verify(U, V, W, n, m, p):
            # Round to nearest half-integer
            U_list = [[int(round(x * 2)) / 2 for x in row] for row in U.tolist()]
            V_list = [[int(round(x * 2)) / 2 for x in row] for row in V.tolist()]
            W_list = [[int(round(x * 2)) / 2 for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    U_list = [[int(round(x * 2)) / 2 for x in row] for row in U.tolist()]
    V_list = [[int(round(x * 2)) / 2 for x in row] for row in V.tolist()]
    W_list = [[int(round(x * 2)) / 2 for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
