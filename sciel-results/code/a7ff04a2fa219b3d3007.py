"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # 115s safety margin within 120s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    # Schoolbook rank decomposition (rank = n*m*p = 75)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k"""
        rank = U.shape[1]
        # Efficient reconstruction using einsum or loops
        # reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        reconstructed = np.zeros((n, m, p))
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # Outer product sum
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        target = np.ones((n, m, p))
        return np.max(np.abs(reconstructed - target)) < tol

    def als_decomposition(T, n, m, p, rank, max_iters=50, tol=1e-8, seed=0):
        """
        Alternating Least Squares for tensor decomposition.
        T is (n, m, p).
        We want U (n*m, rank), V (m*p, rank), W (p*n, rank)
        such that sum_c U_c V_c W_c^T approximates T.
        Note: The indices in the task are U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
        This corresponds to T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
        """
        np.random.seed(seed)
        
        # Initialize factors
        U = np.random.normal(0, 1, (n * m, rank))
        V = np.random.normal(0, 1, (m * p, rank))
        W = np.random.normal(0, 1, (p * n, rank))
        
        prev_error = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Solve for U
            # For each (i,j), we need U[i*m+j, :] to minimize || V[j*p+k, :] W[k*n+i, :]^T - T[i,j,:] ||
            # This is a linear least squares problem: A U_row = b
            # Where A is (rank x rank) formed by V and W, and b is T[i,j,:]
            
            # We can vectorize this by constructing the design matrix for all (i,j)
            # However, the design matrix depends on j and i.
            # Let's loop over i and j for simplicity, as n*m is small (15).
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Construct A: (rank x rank)
                    # A[a, b] = sum_k V[j*p+k, a] * W[k*n+i, b]
                    # V_slice = V[j*p : (j+1)*p, :] (p x rank)
                    # W_slice = W[:, i] -> reshape to (p, n) then take col i? No, W is (p*n, rank).
                    # W[k*n+i, b] is the element at row k*n+i, col b.
                    # Let's reshape W to (p, n, rank). Then W[k, i, b] = W_flat[k*n+i, b].
                    W_reshaped = W.reshape(p, n, rank)
                    V_slice = V[j*p : (j+1)*p, :] # (p, rank)
                    
                    # A = V_slice.T @ W_reshaped[:, i, :] ? No.
                    # A[a, b] = sum_k V_slice[k, a] * W_reshaped[k, i, b]
                    # This is (V_slice.T @ W_reshaped[:, i, :])?
                    # V_slice.T is (rank, p). W_reshaped[:, i, :] is (p, rank).
                    # Yes.
                    A = V_slice.T @ W_reshaped[:, i, :] # (rank, rank)
                    b = T[i, j, :] # (p,)
                    
                    # Wait, the equation is sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] W[k*n+i, c]) = T[i,j,k]?
                    # No, the target is T[i,j,k].
                    # The term is sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
                    # This is not a simple linear regression for U because V and W are coupled in the sum over c.
                    # Actually, for fixed i, j, and k, the equation is:
                    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                    # This is a system of linear equations for U[i*m+j, :]?
                    # Let u = U[i*m+j, :].
                    # T[i,j,:] = u @ (V_slice * W_reshaped[:, i, :]).T ?
                    # Let M = V_slice * W_reshaped[:, i, :] (element-wise product, p x rank)
                    # Then T[i,j,k] = sum_c u[c] * M[k,c]
                    # So T[i,j,:] = M.T @ u
                    # We solve M.T @ u = T[i,j,:]
                    # M is (p, rank). M.T is (rank, p).
                    # If rank > p, this is underdetermined. We use least squares.
                    M = V_slice * W_reshaped[:, i, :] # (p, rank)
                    # Solve M.T @ u = T[i,j,:]
                    # u = np.linalg.lstsq(M.T, T[i,j,:], rcond=None)[0]
                    # Using normal equations: (M @ M.T) u = M @ T[i,j,:]
                    # M @ M.T is (rank, rank).
                    A_normal = M @ M.T # (rank, rank)
                    b_normal = M @ T[i,j,:] # (rank,)
                    try:
                        u_sol = np.linalg.solve(A_normal, b_normal)
                    except np.linalg.LinAlgError:
                        u_sol = np.linalg.lstsq(A_normal, b_normal, rcond=None)[0]
                    U_new[i*m+j, :] = u_sol
            
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                    # For fixed j,k, let v = V[j*p+k, :].
                    # T[:,j,k] = (U_reshaped[:, j, :] * W_reshaped[k, :, :]).T @ v ?
                    # U is (n*m, rank) -> reshape to (n, m, rank). U[i,j,c].
                    # W is (p*n, rank) -> reshape to (p, n, rank). W[k,i,c].
                    U_reshaped = U.reshape(n, m, rank)
                    W_reshaped = W.reshape(p, n, rank)
                    
                    # N[i,c] = U[i,j,c] * W[k,i,c]
                    N = U_reshaped[:, j, :] * W_reshaped[k, :, :] # (n, rank)
                    # T[:,j,k] = N.T @ v
                    # N.T is (rank, n). T[:,j,k] is (n,).
                    A_normal = N @ N.T # (rank, rank)
                    b_normal = N @ T[:, j, k] # (rank,)
                    try:
                        v_sol = np.linalg.solve(A_normal, b_normal)
                    except np.linalg.LinAlgError:
                        v_sol = np.linalg.lstsq(A_normal, b_normal, rcond=None)[0]
                    V_new[j*p+k, :] = v_sol
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                    # For fixed k,i, let w = W[k*n+i, :].
                    # T[i,:,k] = (U_reshaped[i, :, :] * V_reshaped[:, k, :]).T @ w ?
                    # U is (n, m, rank). U[i,j,c].
                    # V is (m, p, rank). V[j,k,c].
                    U_reshaped = U.reshape(n, m, rank)
                    V_reshaped = V.reshape(m, p, rank)
                    
                    # P[j,c] = U[i,j,c] * V[j,k,c]
                    P = U_reshaped[i, :, :] * V_reshaped[:, k, :] # (m, rank)
                    # T[i,:,k] = P.T @ w
                    A_normal = P @ P.T # (rank, rank)
                    b_normal = P @ T[i, :, k] # (rank,)
                    try:
                        w_sol = np.linalg.solve(A_normal, b_normal)
                    except np.linalg.LinAlgError:
                        w_sol = np.linalg.lstsq(A_normal, b_normal, rcond=None)[0]
                    W_new[k*n+i, :] = w_sol
            W = W_new
            
            # Calculate error
            # reconstructed[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            # Use einsum for speed
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            recon = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            error = np.linalg.norm(recon - T)
            
            if np.abs(prev_error - error) < tol:
                break
            prev_error = error
            
        return U, V, W, error

    # Try ranks from 74 down to 69
    for target_rank in range(74, 68, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random restarts
        num_restarts = 5
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            U, V, W, error = als_decomposition(T, n, m, p, target_rank, max_iters=20, tol=1e-10, seed=seed)
            
            if error < 1e-6:
                # Verify with high precision
                if verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            # If we found a valid decomposition for this rank, we can stop
            # or continue to see if we can go lower?
            # The task asks for rank-minimal. So we should try to go lower.
            # But if we found one for 74, we should check 73, etc.
            # The loop continues to lower ranks.
            pass

    return best_result
# EVOLVE-BLOCK-END
