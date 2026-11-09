"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s

    # Schoolbook decomposition (rank = n*m*p = 63)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for c in range(rank_sb):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        U_sb[i * m + j, c] = 1
        V_sb[j * p + k, c] = 1
        W_sb[k * n + i, c] = 1

    # Build the multiplication tensor T as a 3D array
    # T[i, j, k] = 1 (for all valid i, j, k)
    T = np.ones((n, m, p))

    # Function to check if a decomposition is exact
    def check_decomposition(U, V, W, tol=1e-10):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # Reshape U, V, W to 3D arrays
        # U[i*m+j, c] -> U3[i, j, c]
        U3 = U.reshape(n, m, r)
        # V[j*p+k, c] -> V3[j, k, c]
        V3 = V.reshape(m, p, r)
        # W[k*n+i, c] -> W3[k, i, c]
        W3 = W.reshape(p, n, r)
        
        # Compute the tensor from the decomposition
        # T_recon[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is a bit tricky to do efficiently without loops, but for small r it's fine
        T_recon = np.zeros((n, m, p))
        for c in range(r):
            uc = U3[:, :, c]  # (n, m)
            vc = V3[:, :, c]  # (m, p)
            wc = W3[:, :, c]  # (p, n)
            # outer product: uc[i,j] * vc[j,k] * wc[k,i]
            # We can do: for each i, j: uc[i,j] * (vc[j,:] * wc[:,i])
            # Or more efficiently:
            # term = np.einsum('ij,jk,ki->ijk', uc, vc, wc)
            term = np.einsum('ij,jk,ki->ijk', uc, vc, wc)
            T_recon += term
        
        return np.allclose(T_recon, T, atol=tol)

    # Function to try to round a decomposition and check
    def try_round(U, V, W):
        # Try rounding to nearest integer or half-integer
        # First try integers
        U_int = np.round(U).astype(float)
        V_int = np.round(V).astype(float)
        W_int = np.round(W).astype(float)
        if check_decomposition(U_int, V_int, W_int):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        # Try half-integers: round(2*x)/2
        U_half = np.round(2 * U) / 2.0
        V_half = np.round(2 * V) / 2.0
        W_half = np.round(2 * W) / 2.0
        if check_decomposition(U_half, V_half, W_half):
            return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
        return None

    # Best result found so far
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try to find a lower rank decomposition using gradient descent
    # We'll try ranks from rank_sb-1 down to some minimum
    # For each rank, initialize with random Gaussian and run a few iterations
    
    np.random.seed(42)
    
    for r in range(rank_sb - 1, 30, -1):  # Try ranks from 62 down to 31
        if time.time() > deadline - 10:
            break
            
        # Initialize with random matrices
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to approximate the tensor
        # The Frobenius norm of the tensor is sqrt(n*m*p)
        scale = np.linalg.norm(T) / (np.linalg.norm(U) * np.linalg.norm(V) * np.linalg.norm(W)) ** (1/3)
        U *= scale
        V *= scale
        W *= scale
        
        # Simple gradient descent for a few iterations
        # We'll use a simplified approach: alternating least squares
        # For each factor, fix the others and solve for it
        
        for iteration in range(20):
            if time.time() > deadline - 5:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    # We want to find u such that sum_k u[i,j,k] * v[j,k,:] * w[k,i,:] = T[i,j,:]
                    # This is a linear system: A @ u = b
                    # where A[c, c'] = sum_k v[j,k,c] * w[k,i,c']
                    # and b[c] = T[i,j,c]
                    A_mat = np.zeros((r, r))
                    for k in range(p):
                        A_mat += np.outer(V[j*p+k, :], W[k*n+i, :])
                    b = T[i, j, :]  # (p,) -- wait, T is (n,m,p), so T[i,j,:] is (p,)
                    # But we need b to be (r,) for the linear system
                    # Actually, the equation is: sum_{c'} u[i,j,c'] * (V * W)[i,j,c'] = T[i,j,k] for each k
                    # This is not a single linear system for u, because the left side depends on k
                    # Let me rethink.
                    # The residual for position (i,j,k) is:
                    # R[i,j,k] = T[i,j,k] - sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed i,j, we want to minimize sum_k (T[i,j,k] - sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])^2
                    # Let u = U[i*m+j,:], and for each k, let v_k = V[j*p+k,:], w_k = W[k*n+i,:]
                    # We want to minimize sum_k (T[i,j,k] - dot(u, v_k * w_k))^2
                    # This is a least squares problem: M @ u = t
                    # where M[k, c] = V[j*p+k,c] * W[k*n+i,c] and t[k] = T[i,j,k]
                    M_mat = np.zeros((p, r))
                    for k in range(p):
                        M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    t_vec = T[i, j, :]
                    # Solve M @ u = t in least squares sense
                    u_new, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                    U[i*m+j, :] = u_new
                    
            # Update V
            for j in range(m):
                for k in range(p):
                    M_mat = np.zeros((n, r))
                    for i in range(n):
                        M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    t_vec = T[:, j, k]
                    v_new, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                    V[j*p+k, :] = v_new
                    
            # Update W
            for k in range(p):
                for i in range(n):
                    M_mat = np.zeros((m, r))
                    for j in range(m):
                        M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    t_vec = T[i, :, k]
                    w_new, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                    W[k*n+i, :] = w_new
        
        # Check if the decomposition is exact (or close to exact)
        if check_decomposition(U, V, W):
            rounded = try_round(U, V, W)
            if rounded:
                best_result = {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
                best_rank = r
                break
        else:
            # Even if not exactly zero residual, try rounding
            rounded = try_round(U, V, W)
            if rounded:
                best_result = {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
                best_rank = r
                break

    return best_result
# EVOLVE-BLOCK-END
