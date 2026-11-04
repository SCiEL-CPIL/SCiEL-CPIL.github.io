"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Verify a decomposition exactly (integer/half-integer entries)
    def verify_exact(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W give an exact decomposition of the matmul tensor."""
        rank = len(U[0]) if U else 0
        # Check non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for t in range(rank):
                        val += U[a][t] * V[b][t] * W[c][t]
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero entries: T[a,b,c] = 0 when indices are inconsistent
        # a = i*m+j, b = j2*p+k, c = k2*n+i2
        # Zero when not (i==i2 and j==j2 and k==k2)
        # We need to check all entries, but that's n*m * m*p * p*n = 10*25*10 = 2500
        # For each, compute sum_t U[a,t]*V[b,t]*W[c,t] and check it's ~0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue  # Already checked
                    val = 0.0
                    for t in range(rank):
                        val += U[a][t] * V[b][t] * W[c][t]
                    if abs(val) > tol:
                        return False
        return True
    
    # Try to find lower-rank decompositions
    # Strategy: Use iterative truncated SVD on unfolded tensor matrices
    # The tensor T has shape (n*m, m*p, p*n) = (10, 25, 10)
    
    # Build the full tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Unfold the tensor along different modes
    # Mode-1 unfolding: T_1 is (n*m) x (m*p * p*n)
    T_1 = T.reshape(n * m, -1)  # (10, 250)
    # Mode-2 unfolding: T_2 is (m*p) x (n*m * p*n)
    T_2 = np.transpose(T, (1, 0, 2)).reshape(m * p, -1)  # (25, 100)
    # Mode-3 unfolding: T_3 is (p*n) x (n*m * m*p)
    T_3 = np.transpose(T, (2, 0, 1)).reshape(p * n, -1)  # (10, 250)
    
    # Get singular values to estimate numerical rank
    s1 = np.linalg.svd(T_1, compute_uv=False)
    s2 = np.linalg.svd(T_2, compute_uv=False)
    s3 = np.linalg.svd(T_3, compute_uv=False)
    
    # The tensor rank is at least the max of the numerical ranks of all unfoldings
    # But we want to find an exact low-rank decomposition
    
    # Try ranks from a lower bound up to the schoolbook rank
    # Lower bound: max of the numerical ranks
    tol_sv = 1e-10
    r1 = np.sum(s1 > tol_sv)
    r2 = np.sum(s2 > tol_sv)
    r3 = np.sum(s3 > tol_sv)
    lower_bound = max(r1, r2, r3)
    
    # For the matrix multiplication tensor, the rank is known to be at most n*m*p
    # and at least max(n*m, m*p, p*n) / something... 
    # Actually, for matmul, the rank is at least max(n*m, m*p, p*n) in some cases
    # But more precisely, it's at least the maximum of the singular value ranks
    
    # Let's try to find a decomposition of rank r for r from lower_bound to schoolbook
    # We'll use a simple approach: try random rank-r decompositions and check
    
    # Actually, let's use a more structured approach.
    # The tensor T can be seen as a linear map from R^{m*p} to R^{n*m} ⊗ R^{p*n}
    # or we can use the fact that for matmul, there are known low-rank decompositions
    
    # For 2x5 * 5x5, let's try various ranks
    # We'll use a simple random search for small ranks
    
    def try_rank_decomposition(rank, n, m, p, T, max_iters=100):
        """Try to find a rank decomposition of the tensor."""
        # Use iterative SVD-based approach
        # Initialize U, V, W randomly
        np.random.seed(42)
        
        # Initialize with random matrices
        U_mat = np.random.randn(n * m, rank)
        V_mat = np.random.randn(m * p, rank)
        W_mat = np.random.randn(p * n, rank)
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, U[a,:] should satisfy sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
            # This is a linear system in U[a,:] for each a
            # Fix V and W, solve for U
            # T[a,:,:] = sum_t U[a,t] * outer(V[:,t], W[:,t])
            # This is: T[a,:,:].reshape(-1) = U[a,:] @ (V[:,t] ⊗ W[:,t]) for each t
            # Let M be (m*p*p*n) x rank matrix where M[bp*pn+c, t] = V[b,t]*W[c,t]
            # Then T[a,:].reshape(-1) = M @ U[a,:]
            
            # Build M
            # M has shape (m*p * p*n, rank)
            # M[b * p*n + c, t] = V_mat[b, t] * W_mat[c, t]
            M = np.zeros((m * p * p * n, rank))
            for t in range(rank):
                M[:, t] = np.outer(V_mat[:, t], W_mat[:, t]).reshape(-1)
            
            # For each a, solve M @ U[a,:] = T[a,:].reshape(-1)
            # Use least squares
            for a in range(n * m):
                rhs = T[a, :, :].reshape(-1)
                # Solve M @ x = rhs using least squares
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    U_mat[a, :] = sol
                except:
                    pass
            
            # Similarly update V
            # T[:,b,:].reshape(-1) = N @ V[b,:]
            # N[a * p*n + c, t] = U[a,t] * W[c,t]
            N = np.zeros((n * m * p * n, rank))
            for t in range(rank):
                N[:, t] = np.outer(U_mat[:, t], W_mat[:, t]).reshape(-1)
            
            for b in range(m * p):
                rhs = T[:, b, :].reshape(-1)
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(N, rhs, rcond=None)
                    V_mat[b, :] = sol
                except:
                    pass
            
            # Similarly update W
            # T[:,:,c].reshape(-1) = P @ W[c,:]
            # P[a * m*p + b, t] = U[a,t] * V[b,t]
            P = np.zeros((n * m * m * p, rank))
            for t in range(rank):
                P[:, t] = np.outer(U_mat[:, t], V_mat[:, t]).reshape(-1)
            
            for c in range(p * n):
                rhs = T[:, :, c].reshape(-1)
                try:
                    sol, residuals, rank_ls, sv = np.linalg.lstsq(P, rhs, rcond=None)
                    W_mat[c, :] = sol
                except:
                    pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                        err += (val - 1.0) ** 2
            # Check zero entries (sample)
            import random
            for _ in range(100):
                a = random.randint(0, n * m - 1)
                b = random.randint(0, m * p - 1)
                c = random.randint(0, p * n - 1)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if i == i2 and j == j2 and k == k2:
                    continue
                val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                err += val ** 2
            
            if err < best_err:
                best_err = err
                best_U = U_mat.copy()
                best_V = V_mat.copy()
                best_W = W_mat.copy()
            
            if best_err < 1e-8:
                break
        
        if best_err < 1e-6:
            # Convert to list of lists with rounded values
            U_list = [[float(round(U_mat[a, t], 6)) for t in range(rank)] for a in range(n * m)]
            V_list = [[float(round(V_mat[b, t], 6)) for t in range(rank)] for b in range(m * p)]
            W_list = [[float(round(W_mat[c, t], 6)) for t in range(rank)] for c in range(p * n)]
            
            # Verify exactness
            if verify_exact(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from lower_bound up to schoolbook
    for r in range(lower_bound, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_decomposition(r, n, m, p, T, max_iters=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
