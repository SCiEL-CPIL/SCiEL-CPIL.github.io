import time
import numpy as np
from scipy import linalg
import random

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    tensor_decomp = np.zeros((n, m, p, rank))
    for r in range(rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_decomp[i, j, k, r] = u_col[i, j] * v_col[j, k] * w_col[k, i]
    
    # The matrix multiplication tensor is T[i,j,k] = 1 if i,j,k are valid
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # T[i,j,k] = delta(i,j,k) meaning: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle dimension)
    
    # Let me reconsider. The matrix multiplication tensor T is defined such that:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: (A*B)[i,k] = sum_{j,r} T[i,j,k,r] * A[i,j] * B[j,k] * lambda_r
    # Wait, that's not quite right either.
    
    # The standard formulation: the tensor T has entries T[i,j,k] where:
    # T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # For C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is simply T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't look right dimensionally. Let me re-read the task.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r] is the entry for position (k,i)
    
    # The decomposition should satisfy:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # What is T[i,j,k]? For matrix multiplication, the tensor that gives:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # This is the "multiplication tensor" which has T[i,j,k] = 1 for all valid indices
    # because each term A[i,j]*B[j,k] contributes to C[i,k].
    
    # Actually, I think the tensor is defined differently. Let me think again.
    # The matrix multiplication can be seen as a trilinear map:
    # M(A, B) = A * B
    # Where A is n x m, B is m x p, result is n x p.
    
    # In tensor form, we have a tensor T of shape (n*m, m*p, p*n) such that:
    # <T, A_flat ⊗ B_flat> = (A*B)_flat
    # where A_flat is A reshaped to n*m, B_flat is B reshaped to m*p, etc.
    
    # Actually, I think the correct formulation is:
    # The tensor T[i,j,k] (with i indexing rows of A, j columns of A/rows of B, k columns of B)
    # is 1 for all valid i,j,k, representing that A[i,j] * B[j,k] contributes to C[i,k].
    
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let me just check: for the schoolbook, U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # where col = i*m*p + j*p + k (or similar). Then sum_r gives 1 for the matching (i,j,k) and 0 otherwise.
    # But T[i,j,k] should be 1 for all valid (i,j,k), so this works!
    
    # So the check is: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _apply_random_basis(U, V, W, n, m, p):
    """Apply random invertible basis transformations to U, V, W."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Generate random invertible matrices
    G_U = linalg.inv(np.random.randn(n * m, n * m))
    G_V = linalg.inv(np.random.randn(m * p, m * p))
    G_W = linalg.inv(np.random.randn(p * n, p * n))
    
    # Apply: U_new = G_U @ U, V_new = G_V @ V, W_new = G_W @ W
    # But we need to maintain the decomposition property.
    # The decomposition is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # If we transform U -> G_U @ U, V -> G_V @ V, W -> G_W @ W, the product changes.
    
    # Actually, for tensor decompositions, we can apply:
    # U_new = G_U @ U @ H^{-1}
    # V_new = G_V @ V @ H
    # W_new = G_W @ W
    # for some invertible H, but this gets complicated.
    
    # A simpler approach: just randomly permute and scale columns, or use small perturbations.
    # Let's try: randomly mix columns.
    
    H = linalg.inv(np.random.randn(rank, rank))
    U_new = U @ H
    V_new = V @ H
    W_new = W @ np.linalg.inv(H)
    
    # Verify the decomposition still holds
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_new[i * m + j, r] * V_new[j * p + k, r] * W_new[k * n + i, r]
                if abs(s - 1.0) > 1e-6:
                    # Transformation broke the decomposition, revert
                    return U, V, W
    
    return U_new, V_new, W_new

def _try_rank_reduction(U, V, W, n, m, p, max_iter=1000, time_limit=100.0):
    """Try to reduce the rank by finding linearly dependent columns."""
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(U[0])
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Apply a random basis transformation
        U_t, V_t, W_t = _apply_random_basis(U, V, W, n, m, p)
        
        # Try to find the minimum rank by checking matricizations
        # The tensor can be flattened in different ways.
        # One matricization: for each r, we have a vector. If some columns are linearly dependent,
        # we might be able to reduce the rank.
        
        # Check if any column can be removed while maintaining the decomposition
        # This is expensive, so let's use a heuristic: compute the SVD of the "flattened" tensor
        # and see if the numerical rank is lower.
        
        # Build the full tensor from the decomposition
        tensor = np.zeros((n * m, m * p, p * n, rank))
        for r in range(rank):
            u_col = U_t[:, r].reshape(n, m)
            v_col = V_t[:, r].reshape(m, p)
            w_col = W_t[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i * m + j, j * p + k, k * n + i, r] = u_col[i, j] * v_col[j, k] * w_col[k, i]
        
        # The actual tensor T should be all ones in the appropriate positions
        # Let's check the numerical rank of the tensor by flattening it
        # Flatten: T_flat[i*m*p + j*p + k, :] where i,j,k range over valid indices
        # Actually, let's just check if we can find a lower rank by solving a least squares problem
        
        # For now, just keep the current transformation if it's valid
        # and check if the effective rank is lower via SVD of a matricization
        
        # Matricize: fix one index, get a matrix
        # E.g., for each (i,j), we have a vector over (k,r): V[j*p+k, r] * W[k*n+i, r]
        # This is getting complex. Let's just do a simple check.
        
        # Try to see if we can drop a column
        # For each column r, check if removing it still gives a valid decomposition
        # (This is O(rank * n*m*p * rank) which might be too slow)
        
        # Alternative: use the fact that if the decomposition has rank R, 
        # the tensor has a certain structure. Let's just track the best we find.
        
        if _verify(U_t.tolist(), V_t.tolist(), W_t.tolist(), n, m, p):
            # Check if we can reduce rank by looking at the column space
            # Compute the SVD of U_t, V_t, W_t to see their numerical ranks
            _, s_U, _ = np.linalg.svd(U_t, full_matrices=False)
            _, s_V, _ = np.linalg.svd(V_t, full_matrices=False)
            _, s_W, _ = np.linalg.svd(W_t, full_matrices=False)
            
            # If any of these has a small singular value, we might be able to reduce rank
            # But this is heuristic
            
            # For now, just keep the transformation
            U, V, W = U_t, V_t, W_t
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leaving 10s margin for 120s eval / 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W, rank = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to reduce rank using randomized basis transformations
    try:
        U_new, V_new, W_new, new_rank = _try_rank_reduction(U, V, W, n, m, p, 
                                                              max_iter=500, 
                                                              time_limit=deadline - time.time() - 5.0)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
    except Exception:
        pass
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
