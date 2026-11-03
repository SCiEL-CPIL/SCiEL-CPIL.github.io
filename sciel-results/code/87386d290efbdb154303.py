import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T
    # T has shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u = i*m + j, v = j*p + k, w = k*n + i
    # Otherwise 0
    
    # We will search for rank r from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Precompute the non-zero indices of T
    # T is a tensor of size (n*m, m*p, p*n)
    # We can store it as a list of (u, v, w) tuples where T[u,v,w] = 1
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zeros.append((u, v, w))
    
    # Function to compute the residual for given U, V, W
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    def compute_residual(U, V, W):
        # We want to minimize sum over non_zeros of (1 - sum_c U[u,c]V[v,c]W[w,c])^2
        # This is equivalent to minimizing the Frobenius norm of (T - T_approx)
        # T_approx[u, v, w] = sum_c U[u, c] * V[v, c] * W[w, c]
        
        # For efficiency, we can iterate over the non-zero entries
        # But vectorization is better.
        # Let's create a dense tensor? No, it's small enough.
        # Size: 6 * 18 * 12 = 1296 elements.
        # We can just create a dense array of zeros and fill it.
        
        T_approx = np.zeros((n*m, m*p, p*n))
        # Use einsum for speed?
        # T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        # This might be slow for large r, but r is small (<36).
        # Let's try it.
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        
        # Calculate residual on non-zero entries
        # T_target has 1s at non_zeros
        # Residual = sum (T_target - T_approx)^2
        # Since T_target is 0 elsewhere, and we only care about non-zeros for the fit?
        # No, the decomposition must be exact for the whole tensor.
        # But the tensor is 0 elsewhere.
        # So we need T_approx to be 0 at all zero entries of T.
        # And 1 at all non-zero entries.
        
        # So we need to check all entries?
        # The tensor size is 1296. This is small.
        # We can just compute the full residual.
        
        # Create target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for u, v, w in non_zeros:
            T_target[u, v, w] = 1.0
            
        diff = T_target - T_approx
        return np.sum(diff**2)

    # Search for rank
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    best_solution = None
    
    for r in range(lower_bound, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution of rank r
        found = False
        num_restarts = 10
        max_iters = 500
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Simple gradient descent or alternating least squares
            # ALS is faster and more stable for this small problem.
            # For each c, U_c, V_c, W_c are vectors.
            # We can optimize each vector in turn.
            
            # Actually, let's use a simple projection method or gradient descent with numpy.
            # Given the small size, we can use scipy.optimize.minimize?
            # No, let's stick to numpy ALS.
            
            # ALS:
            # Fix V, W, solve for U_c.
            # U_c is a vector of length n*m.
            # We want to minimize sum_{u,v,w} (T_target[u,v,w] - U_c[u] V_c[v] W_c[w])^2
            # This is a linear least squares problem for U_c.
            
            # Let's implement ALS.
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                for c in range(r):
                    # Residual R = T_target - sum_{c' != c} U_c' V_c' W_c'
                    # We want to find U_c such that <U_c, R_{:, :, :} contracted with V_c, W_c> is minimized?
                    # Actually, we minimize || R - U_c \otimes V_c \otimes W_c ||^2
                    # This is equivalent to finding U_c that best fits R with the fixed V_c, W_c.
                    
                    # Compute R
                    # R = T_target
                    # for c_prime in range(r):
                    #     if c_prime != c:
                    #         R -= np.einsum('i,j,k->ijk', U[:, c_prime], V[:, c_prime], W[:, c_prime])
                    
                    # This is O(r * size).
                    # Let's do it.
                    
                    R = T_target.copy()
                    for c_prime in range(r):
                        if c_prime != c:
                            R -= np.einsum('i,j,k->ijk', U[:, c_prime], V[:, c_prime], W[:, c_prime])
                    
                    # Now solve for U_c.
                    # We want to minimize || R - U_c \otimes V_c \otimes W_c ||^2
                    # Let M be a matrix where M[u, (v,w)] = V_c[v] * W_c[w]
                    # Then we want to solve M U_c = vec(R)
                    # This is a least squares problem.
                    
                    M = np.outer(V[:, c], W[:, c]).reshape(m*p, p*n)
                    # Wait, dimensions.
                    # U_c is (n*m,).
                    # V_c is (m*p,).
                    # W_c is (p*n,).
                    # The term is U_c[u] * V_c[v] * W_c[w].
                    # We can reshape R to (n*m, m*p*p*n).
                    # And M to (m*p*p*n, n*m)? No.
                    
                    # Let's reshape R to (n*m, -1).
                    R_reshaped = R.reshape(n*m, -1)
                    
                    # The model is R_reshaped[u, :] = U_c[u] * (V_c \otimes W_c)_flat
                    # Let z = V_c \otimes W_c. z has length m*p*p*n.
                    # We want to minimize || R_reshaped - U_c z^T ||_F^2
                    # This is a rank-1 approximation of R_reshaped.
                    # The best U_c is proportional to R_reshaped z.
                    # Specifically, U_c = (R_reshaped @ z) / (z @ z)
                    
                    z = np.outer(V[:, c], W[:, c]).flatten()
                    denom = np.dot(z, z)
                    if denom < 1e-10:
                        continue
                    U[:, c] = np.dot(R_reshaped, z) / denom
                    
                # Update V
                for c in range(r):
                    R = T_target.copy()
                    for c_prime in range(r):
                        if c_prime != c:
                            R -= np.einsum('i,j,k->ijk', U[:, c_prime], V[:, c_prime], W[:, c_prime])
                    
                    # Minimize || R - U_c \otimes V_c \otimes W_c ||^2
                    # Reshape R to (m*p, -1).
                    R_reshaped = R.transpose(1, 0, 2).reshape(m*p, -1) # (m*p, n*m*p*n)
                    
                    z = np.outer(U[:, c], W[:, c]).flatten()
                    # z length is n*m*p*n.
                    # R_reshaped shape is (m*p, n*m*p*n).
                    # Model: R_reshaped[v, :] = V_c[v] * (U_c \otimes W_c)_flat
                    # V_c = (R_reshaped @ z) / (z @ z)
                    
                    denom = np.dot(z, z)
                    if denom < 1e-10:
                        continue
                    V[:, c] = np.dot(R_reshaped, z) / denom
                    
                # Update W
                for c in range(r):
                    R = T_target.copy()
                    for c_prime in range(r):
                        if c_prime != c:
                            R -= np.einsum('i,j,k->ijk', U[:, c_prime], V[:, c_prime], W[:, c_prime])
                    
                    # Minimize || R - U_c \otimes V_c \otimes W_c ||^2
                    # Reshape R to (p*n, -1).
                    R_reshaped = R.transpose(2, 0, 1).reshape(p*n, -1) # (p*n, n*m*m*p)
                    
                    z = np.outer(U[:, c], V[:, c]).flatten()
                    # z length is n*m*m*p.
                    # R_reshaped shape is (p*n, n*m*m*p).
                    # Model: R_reshaped[w, :] = W_c[w] * (U_c \otimes V_c)_flat
                    # W_c = (R_reshaped @ z) / (z @ z)
                    
                    denom = np.dot(z, z)
                    if denom < 1e-10:
                        continue
                    W[:, c] = np.dot(R_reshaped, z) / denom
            
            # Check residual
            res = compute_residual(U, V, W)
            if res < 1e-6:
                # Try rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                res_rounded = compute_residual(U_rounded, V_rounded, W_rounded)
                if res_rounded < 1e-9:
                    # Found a valid solution
                    best_solution = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    if best_solution is not None:
        return best_solution
    else:
        # Fallback to schoolbook
        rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
