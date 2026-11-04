"""Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Check if the decomposition exactly reproduces the tensor."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=atol)

    def round_and_verify(U, V, W):
        """Try rounding to integers/half-integers and verify."""
        for denom in [1, 2, 4]:
            scale = float(denom)
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, atol=1e-9):
                return U_r, V_r, W_r
        return None

    # Schoolbook decomposition: rank = n*m*p = 72
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch), dtype=np.float64)
    V_sch = np.zeros((m * p, rank_sch), dtype=np.float64)
    W_sch = np.zeros((p * n, rank_sch), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0

    assert verify(U_sch, V_sch, W_sch), "Schoolbook verification failed"

    best_rank = rank_sch
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()

    # Pre-compute matricizations for ALS
    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def als_decompose(U_init, V_init, W_init, max_iter=80, tol=1e-10):
        """Run ALS to refine the decomposition."""
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            MW = np.einsum('br,cr->bcr', V, W)
            MW_flat = MW.reshape(m * p * p * n, rank)
            G = MW_flat.T @ MW_flat
            G_inv = np.linalg.pinv(G + tol * np.eye(rank))
            U_new = T_A @ MW_flat @ G_inv
            
            # Update V
            UW = np.einsum('ar,cr->acr', U_new, W)
            UW_flat = UW.reshape(n * m * p * n, rank)
            G = UW_flat.T @ UW_flat
            G_inv = np.linalg.pinv(G + tol * np.eye(rank))
            V_new = T_B @ UW_flat @ G_inv
            
            # Update W
            UV = np.einsum('ar,br->abr', U_new, V_new)
            UV_flat = UV.reshape(n * m * m * p, rank)
            G = UV_flat.T @ UV_flat
            G_inv = np.linalg.pinv(G + tol * np.eye(rank))
            W_new = T_C @ UV_flat @ G_inv
            
            delta = max(np.linalg.norm(U_new - U), np.linalg.norm(V_new - V), np.linalg.norm(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if delta < 1e-12:
                break

        return U, V, W

    # Try to find a lower-rank decomposition
    target_ranks = list(range(rank_sch - 1, 39, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        
        for restart in range(15):
            if time.time() > deadline:
                break
            
            if restart < 5:
                # Random initialization
                np.random.seed(42 + restart)
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            elif restart < 10:
                # Perturb schoolbook
                np.random.seed(100 + restart)
                U_init = U_sch[:, :target_rank].copy() + 0.1 * np.random.randn(n * m, target_rank)
                V_init = V_sch[:, :target_rank].copy() + 0.1 * np.random.randn(m * p, target_rank)
                W_init = W_sch[:, :target_rank].copy() + 0.1 * np.random.randn(p * n, target_rank)
            else:
                # Sparse random initialization
                np.random.seed(200 + restart)
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    idx_u = np.random.choice(n * m, size=3, replace=False)
                    idx_v = np.random.choice(m * p, size=3, replace=False)
                    idx_w = np.random.choice(p * n, size=3, replace=False)
                    U_init[idx_u, r] = 1.0
                    V_init[idx_v, r] = 1.0
                    W_init[idx_w, r] = 1.0
            
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_sol if False else W_init, max_iter=60)
            
            if verify(U_sol, V_sol, W_sol, atol=1e-7):
                rounded = round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
            else:
                rounded = round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break

        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
