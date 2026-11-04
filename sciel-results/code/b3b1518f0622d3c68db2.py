import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    
    # Verify rank 50
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Precompute unfoldings of T
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def khatri_rao(A, B):
        """
        Compute Khatri-Rao product of A (M x R) and B (N x R).
        Result is (M*N) x R.
        """
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def lstsq_solve(A, B):
        """Fast least squares solve: A @ X = B, returns X."""
        try:
            X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            return X
        except:
            return None

    def als_decompose(U_init, V_init, W_init, max_iters=100, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition using lstsq.
        """
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            # Solve for U
            KR_VW = khatri_rao(V, W)
            U_new = lstsq_solve(KR_VW, T1.T)
            if U_new is None:
                break
            U = U_new.T
            
            # Solve for V
            KR_WU = khatri_rao(W, U)
            V_new = lstsq_solve(KR_WU, T2.T)
            if V_new is None:
                break
            V = V_new.T
            
            # Solve for W
            KR_UV = khatri_rao(U, V)
            W_new = lstsq_solve(KR_UV, T3.T)
            if W_new is None:
                break
            W = W_new.T
            
            # Check convergence
            change = np.linalg.norm(U - U_init) + np.linalg.norm(V - V_init) + np.linalg.norm(W - W_init)
            U_init, V_init, W_init = U.copy(), V.copy(), W.copy()
            
            if change < tol:
                break
        
        return U, V, W

    def polish_decomposition(U, V, W, max_iters=300, tol=1e-14):
        """
        High-precision polish using lstsq-based ALS.
        """
        R = U.shape[1]
        for _ in range(max_iters):
            KR_VW = khatri_rao(V, W)
            U_new = lstsq_solve(KR_VW, T1.T)
            if U_new is None:
                break
            U = U_new.T
            
            KR_WU = khatri_rao(W, U)
            V_new = lstsq_solve(KR_WU, T2.T)
            if V_new is None:
                break
            V = V_new.T
            
            KR_UV = khatri_rao(U, V)
            W_new = lstsq_solve(KR_UV, T3.T)
            if W_new is None:
                break
            W = W_new.T
            
            # Check if we're close
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
        
        return U, V, W

    # Try to reduce rank
    # Start from 49 down to 28
    # Use time budget to decide how many restarts
    
    rng = np.random.default_rng(42)
    
    for target_rank in range(49, 27, -1):
        if time.time() > deadline - 10:
            break
            
        # Determine number of restarts based on time and rank
        time_per_rank = (deadline - 10 - time.time()) / (50 - target_rank)
        num_restarts = max(2, min(20, int(time_per_rank / 2.0)))
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Multiple initialization strategies
            strategy = restart % 5
            
            if strategy == 0:
                # Drop columns from rank-50 and add moderate noise
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                noise_scale = 0.05 * (1 + target_rank / 50)
                U_init += rng.normal(0, noise_scale, U_init.shape)
                V_init += rng.normal(0, noise_scale, V_init.shape)
                W_init += rng.normal(0, noise_scale, W_init.shape)
            elif strategy == 1:
                # Pure random init
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            elif strategy == 2:
                # Drop random subset of columns from rank-50
                cols_to_keep = rng.choice(50, target_rank, replace=False)
                U_init = U50[:, cols_to_keep].copy()
                V_init = V50[:, cols_to_keep].copy()
                W_init = W50[:, cols_to_keep].copy()
                U_init += rng.normal(0, 0.02, U_init.shape)
                V_init += rng.normal(0, 0.02, V_init.shape)
                W_init += rng.normal(0, 0.02, W_init.shape)
            elif strategy == 3:
                # Aggressive perturbation
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.15, U_init.shape)
                V_init += rng.normal(0, 0.15, V_init.shape)
                W_init += rng.normal(0, 0.15, W_init.shape)
            else:
                # Scaled random init
                scale = 0.5
                U_init = rng.normal(0, scale, (n*m, target_rank))
                V_init = rng.normal(0, scale, (m*p, target_rank))
                W_init = rng.normal(0, scale, (p*n, target_rank))
            
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iters=80)
            
            # If close, try to polish with high precision
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-3:
                # High-precision polish
                U_sol, V_sol, W_sol = polish_decomposition(U_sol, V_sol, W_sol, max_iters=400, tol=1e-15)
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)
            
            # Verify with tight tolerance
            if verify_decomp(U_sol, V_sol, W_sol, atol=1e-10):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
                found = True
                break
        
        if found:
            # Found valid decomposition for this rank, continue to try lower
            pass

    # Final high-precision polish on the best solution found
    if best_rank < 50:
        best_U, best_V, best_W = polish_decomposition(best_U, best_V, best_W, max_iters=500, tol=1e-16)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
