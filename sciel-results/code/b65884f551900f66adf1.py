import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # Note: The problem statement says W[k*n+i, r].
    # My indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Pre-flatten T for easier solving
    # T_flat_u[v, w] for fixed u?
    # We need to solve for U rows.
    # Equation: sum_r U[i, r] * (V[:, r] * W[:, r]) = T[i, :, :] flattened
    
    # We will try ranks from 20 to 26. If found, return.
    # Schoolbook is 27.
    
    def snap_to_half_integer(x):
        # Snap to nearest multiple of 0.5
        return np.round(x * 2) / 2.0

    def check_validity(U, V, W):
        # Reconstruct tensor
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is a tensor contraction.
        # We can check a few random entries or all 243 entries.
        # 243 entries * R multiplications. R=23. 243*23 is small.
        R = U.shape[1]
        T_recon = np.zeros_like(T)
        for r in range(R):
            T_recon += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        # Check if T_recon matches T
        return np.allclose(T_recon, T, atol=1e-9)

    def try_rank(R, time_limit=140):
        start_time = time.time()
        max_iterations = 2000 # Arbitrary high, controlled by time
        
        # Initialize V, W with random integers -1, 0, 1
        # Using random integers helps with exactness
        for _ in range(100): # 100 restarts
            if time.time() - start_time > time_limit:
                return None
            
            V = np.random.choice([-1, 0, 1], size=(m*p, R))
            W = np.random.choice([-1, 0, 1], size=(p*n, R))
            
            # Initialize U
            U = np.zeros((n*m, R))
            
            for it in range(max_iterations):
                if time.time() - start_time > time_limit:
                    return None
                
                # Update U
                for i in range(n):
                    # T[i, :, :] is a matrix of size (m*p, p*n)
                    # We want U[i, :] such that sum_r U[i, r] V[:, r] W[:, r] = T[i, :, :]
                    # Let A_r = V[:, r] * W[:, r] (element-wise product? No, outer product? No.)
                    # The term is V[v, r] * W[w, r].
                    # So for fixed r, the contribution to the matrix T[i, :, :] is V[:, r] * W[:, r] (outer product).
                    # Let M be a matrix of size (m*p*p*n, R) where each column r is vec(V[:, r] * W[:, r]).
                    # Then U[i, :] M = vec(T[i, :, :]).
                    
                    # Construct M
                    # V is (m*p, R), W is (p*n, R)
                    # V[:, r] is (m*p,), W[:, r] is (p*n,)
                    # Outer product is (m*p, p*n). Flatten to (m*p*p*n,)
                    
                    # We can do this efficiently.
                    # M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    
                    M = np.zeros((m*p * p*n, R))
                    for r in range(R):
                        M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    
                    b = T[i, :, :].flatten()
                    
                    # Solve U[i, :] M = b
                    # This is an underdetermined system (R > m*p*p*n usually? No, R=23, 9*9=81. So overdetermined.)
                    # Use least squares.
                    sol, residuals, rank, sv = np.linalg.lstsq(M.T, b, rcond=None)
                    # sol is U[i, :]
                    U[i, :] = snap_to_half_integer(sol)
                
                # Update V
                for j in range(m):
                    # T[:, j, :] is not directly a slice because indices are coupled.
                    # T[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i.
                    # For fixed j, we vary i and k.
                    # u = i*m + j
                    # v = j*p + k
                    # w = k*n + i
                    # We need T[i*m+j, j*p+k, k*n+i] for all i, k.
                    # This forms a matrix of size (n, p).
                    # Let's extract this sub-tensor.
                    
                    T_sub = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            T_sub[i, k] = T[u, v, w]
                    
                    # Equation: sum_r V[j, r] * U[i*m+j, r] * W[k*n+i, r] = T_sub[i, k]
                    # For fixed r, term is U[i*m+j, r] * W[k*n+i, r].
                    # This is not a simple outer product of two vectors because the indices i and k are coupled in U and W.
                    # U index depends on i. W index depends on k.
                    # Let A_r = vector of U[i*m+j, r] for i in 0..n-1. Size n.
                    # Let B_r = vector of W[k*n+i, r] for k in 0..p-1, i in 0..n-1? No.
                    # W index is k*n+i.
                    # For fixed r, we have a matrix M_r of size (n, p) where M_r[i, k] = U[i*m+j, r] * W[k*n+i, r].
                    # We want sum_r V[j, r] M_r = T_sub.
                    # This is a linear system for V[j, :].
                    
                    # Construct M_matrix of size (n*p, R)
                    M_mat = np.zeros((n*p, R))
                    for r in range(R):
                        for i in range(n):
                            for k in range(p):
                                u = i * m + j
                                w = k * n + i
                                M_mat[i*p+k, r] = U[u, r] * W[w, r]
                    
                    b = T_sub.flatten()
                    sol, residuals, rank, sv = np.linalg.lstsq(M_mat.T, b, rcond=None)
                    V[j, :] = snap_to_half_integer(sol)
                
                # Update W
                for k in range(p):
                    # Fixed k. Vary i, j.
                    # u = i*m+j, v=j*p+k, w=k*n+i.
                    # T_sub[i, j] = T[i*m+j, j*p+k, k*n+i]
                    
                    T_sub = np.zeros((n, m))
                    for i in range(n):
                        for j in range(m):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            T_sub[i, j] = T[u, v, w]
                    
                    # Equation: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = T_sub[i, j]
                    # Note: W index depends on i. U index depends on i, j. V index depends on j.
                    # This is complex.
                    # Let's just iterate and check.
                    # For each i, solve for W[k*n+i, :]?
                    # No, W[k*n+i, :] is a single row.
                    # For fixed k, we have n rows of W to update: W[k*n+0, :], ..., W[k*n+n-1, :].
                    # They are coupled in the equation?
                    # T_sub[i, j] = sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r])
                    # For fixed i, this is a linear equation for W[k*n+i, :].
                    # Let A_r = U[i*m+j, r] * V[j*p+k, r] for all j.
                    # So for fixed i, we have:
                    # sum_r W[k*n+i, r] * A_r(j) = T_sub[i, j] for all j.
                    # This is a linear system for W[k*n+i, :].
                    
                    for i in range(n):
                        M_vec = np.zeros((m, R))
                        for r in range(R):
                            for j in range(m):
                                u = i * m + j
                                v = j * p + k
                                M_vec[j, r] = U[u, r] * V[v, r]
                        
                        b = T_sub[i, :]
                        sol, residuals, rank, sv = np.linalg.lstsq(M_vec.T, b, rcond=None)
                        W[k*n+i, :] = snap_to_half_integer(sol)
                
                # Check validity
                if check_validity(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # End restarts

    # Try ranks 20 to 26
    for R in range(20, 27):
        res = try_rank(R)
        if res:
            return res
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
