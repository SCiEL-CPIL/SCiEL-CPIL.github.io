# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    # Convert to numpy arrays for speed
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] == 1 for all i,j,k
    
    # Reshape U to (n, m, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute the tensor product sum
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    
    # We can compute this using einsum or loops
    # Let's use einsum for speed
    # T[i,j,k] = U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # Note: W is indexed by k, i.
    
    # Create the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                T[i, j, k] = val
                
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-6)

def _als_decomposition(n, m, p, rank, iters=50, seed=0):
    """Attempt to find a decomposition of given rank using ALS."""
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Target tensor T[i,j,k] = 1
    # We want to minimize || T - sum_c U_c V_c W_c ||^2
    
    for it in range(iters):
        # Update U
        # For each i, j:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is linear in U[i,j,c].
        # Let A[i,j] be the matrix where A[i,j][k, c] = V[j,k,c] * W[k,i,c]
        # Then T[i,j,:] = U[i,j,:] @ A[i,j].T ? No.
        # T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
        # So T[i,j,:] = U[i,j,:] @ M[i,j] where M[i,j][c, k] = V[j,k,c] * W[k,i,c]
        
        for i in range(n):
            for j in range(m):
                # Construct M[i,j] of shape (rank, p)
                # M[c, k] = V[j, k, c] * W[k, i, c]
                # V is (m, p, rank) -> V[j, :, :] is (p, rank)
                # W is (p, n, rank) -> W[:, i, :] is (p, rank)
                V_j = V.reshape(m, p, rank)[j, :, :] # (p, rank)
                W_i = W.reshape(p, n, rank)[:, i, :] # (p, rank)
                M = (V_j * W_i).T # (rank, p)
                
                # Solve U[i, j, :] @ M = 1 (vector of ones of length p)
                # U[i, j, :] is (1, rank)
                # M is (rank, p)
                # We want U @ M = 1
                # This is underdetermined if rank > p.
                # Use least squares
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
                    
        # Update V
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For fixed j, k:
        # T[i,j,k] = V[j,k,c] * sum_c' U[i,j,c'] * W[k,i,c'] * delta_{c,c'}? No.
        # T[i,j,k] = sum_c (U[i,j,c] * W[k,i,c]) * V[j,k,c]
        # Let B[j,k][i, c] = U[i,j,c] * W[k,i,c]
        # Then T[:, j, k] = V[j, k, :] @ B[j, k].T ?
        # T[i, j, k] = sum_c V[j, k, c] * B[j, k][i, c]
        # So T[:, j, k] (vector of length n) = V[j, k, :] (1, rank) @ B[j, k] (rank, n)
        
        for j in range(m):
            for k in range(p):
                # Construct B[j,k] of shape (rank, n)
                # B[c, i] = U[i, j, c] * W[k, i, c]
                U_j = U.reshape(n, m, rank)[:, j, :] # (n, rank)
                W_k = W.reshape(p, n, rank)[k, :, :] # (n, rank)
                B = (U_j * W_k).T # (rank, n)
                
                # Solve V[j, k, :] @ B = 1 (vector of ones of length n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                except:
                    pass
                    
        # Update W
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For fixed k, i:
        # T[i,j,k] = W[k,i,c] * sum_c' U[i,j,c'] * V[j,k,c'] * delta_{c,c'}? No.
        # T[i,j,k] = sum_c W[k,i,c] * (U[i,j,c] * V[j,k,c])
        # Let C[k,i][j, c] = U[i,j,c] * V[j,k,c]
        # Then T[i,:,k] (vector of length m) = W[k,i,:] (1, rank) @ C[k,i] (rank, m)
        
        for k in range(p):
            for i in range(n):
                # Construct C[k,i] of shape (rank, m)
                # C[c, j] = U[i, j, c] * V[j, k, c]
                U_i = U.reshape(n, m, rank)[i, :, :] # (m, rank)
                V_k = V.reshape(m, p, rank)[:, k, :] # (m, rank)
                C = (U_i * V_k).T # (rank, m)
                
                # Solve W[k, i, :] @ C = 1 (vector of ones of length m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                except:
                    pass

    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 30 to best_rank - 1
    for rank in range(30, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random seeds
        for seed in range(10):
            if time.time() > deadline:
                break
                
            result = _als_decomposition(n, m, p, rank, iters=20, seed=seed)
            if result is not None:
                # Attempt to round to integer/half-integer
                # Task requires integer/half-integer entries
                try:
                    U_rounded = [[round(2*x)/2 for x in row] for row in result['U']]
                    V_rounded = [[round(2*x)/2 for x in row] for row in result['V']]
                    W_rounded = [[round(2*x)/2 for x in row] for row in result['W']]
                    
                    # Verify the rounded version
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                        if rank < best_rank:
                            best = {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
                            best_rank = rank
                            if best_rank <= 35:
                                return best
                except:
                    pass
                    
    return best
# EVOLVE-BLOCK-END
