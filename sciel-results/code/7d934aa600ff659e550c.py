import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # Safety margin for evaluator's 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T1 = T.reshape(n * m, -1)
    T2 = T.reshape(m * p, -1)
    T3 = T.reshape(p * n, -1)

    def verify_decomp(U, V, W, atol=1e-8):
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
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_solve(U, V, W, max_iters=100, tol=1e-10):
        R = U.shape[1]
        prev_residual = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(recon - T)
            if residual < tol:
                break
            if abs(prev_residual - residual) < 1e-12 * max(1.0, prev_residual):
                break
            prev_residual = residual
        return U, V, W

    def try_eliminate_column(U, V, W, col_idx):
        """Try to eliminate column col_idx by re-solving without it."""
        R = U.shape[1]
        if R <= 1:
            return None
        U_sub = np.delete(U, col_idx, axis=1)
        V_sub = np.delete(V, col_idx, axis=1)
        W_sub = np.delete(W, col_idx, axis=1)
        
        # Re-solve with R-1 columns
        U_new, V_new, W_new = als_solve(U_sub, V_sub, W_sub, max_iters=50, tol=1e-10)
        recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        if np.allclose(recon, T, atol=1e-6):
            return U_new, V_new, W_new
        return None

    def greedy_column_elimination(U, V, W, max_rounds=10):
        """Iteratively try to eliminate columns that are redundant."""
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            R = U.shape[1]
            eliminated = False
            # Try all columns, not just from the end
            for col_idx in range(R):
                if time.time() > deadline - 8:
                    break
                result = try_eliminate_column(U, V, W, col_idx)
                if result is not None:
                    U, V, W = result
                    eliminated = True
                    break
            if not eliminated:
                break
        return U, V, W

    # First, try aggressive column elimination from rank 50
    U_curr, V_curr, W_curr = U50.copy(), V50.copy(), W50.copy()
    U_curr, V_curr, W_curr = greedy_column_elimination(U_curr, V_curr, W_curr, max_rounds=15)
    if verify_decomp(U_curr, V_curr, W_curr) and U_curr.shape[1] < best_rank:
        best_rank = U_curr.shape[1]
        best_U, best_V, best_W = U_curr, V_curr, W_curr

    # Try different column subsets to initialize lower-rank searches
    rng = np.random.default_rng(42)
    R50 = 50
    
    for target_rank in range(best_rank - 1, max(30, best_rank - 20), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(8):
            if time.time() > deadline - 15:
                break
            
            if attempt == 0:
                # Use first target_rank columns
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
            elif attempt == 1:
                # Use last target_rank columns
                U_init = U50[:, -target_rank:].copy()
                V_init = V50[:, -target_rank:].copy()
                W_init = W50[:, -target_rank:].copy()
            elif attempt == 2:
                # Random subset of columns
                cols = rng.choice(R50, target_rank, replace=False)
                U_init = U50[:, cols].copy()
                V_init = V50[:, cols].copy()
                W_init = W50[:, cols].copy()
            elif attempt == 3:
                # Random subset with perturbation
                cols = rng.choice(R50, target_rank, replace=False)
                U_init = U50[:, cols].copy() + rng.normal(0, 0.1, (n*m, target_rank))
                V_init = V50[:, cols].copy() + rng.normal(0, 0.1, (m*p, target_rank))
                W_init = W50[:, cols].copy() + rng.normal(0, 0.1, (p*n, target_rank))
            elif attempt == 4:
                # Sparse random init
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    u_idx = rng.choice(n*m, 3, replace=False)
                    v_idx = rng.choice(m*p, 3, replace=False)
                    w_idx = rng.choice(p*n, 3, replace=False)
                    U_init[u_idx, r] = rng.choice([-1, 1], 3)
                    V_init[v_idx, r] = rng.choice([-1, 1], 3)
                    W_init[w_idx, r] = rng.choice([-1, 1], 3)
            elif attempt == 5:
                # Dense random init
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            elif attempt == 6:
                # Mix of structured and random
                half = target_rank // 2
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(half):
                    i0 = r % n
                    j0 = (r // n) % m
                    k0 = (r // (n*m)) % p
                    U_init[i0*m+j0, r] = 1
                    V_init[j0*p+k0, r] = 1
                    W_init[k0*n+i0, r] = 1
                U_init[:, half:] = rng.normal(0, 1, (n*m, target_rank-half))
                V_init[:, half:] = rng.normal(0, 1, (m*p, target_rank-half))
                W_init[:, half:] = rng.normal(0, 1, (p*n, target_rank-half))
            else:
                # Orthogonal transform of rank-50 columns
                cols = rng.choice(R50, target_rank, replace=False)
                Q = np.linalg.qr(rng.normal(0, 1, (target_rank, target_rank)))[0]
                U_init = U50[:, cols] @ Q
                V_init = V50[:, cols] @ Q
                W_init = W50[:, cols] @ Q

            U_sol, V_sol, W_sol = als_solve(U_init, V_init, W_init, max_iters=80, tol=1e-10)
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            
            if residual < 1e-5:
                # Polish with tighter tolerance
                for _ in range(20):
                    if time.time() > deadline - 10:
                        break
                    U_sol, V_sol, W_sol = als_solve(U_sol, V_sol, W_sol, max_iters=20, tol=1e-12)
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)

            if verify_decomp(U_sol, V_sol, W_sol):
                # Try to eliminate more columns
                U_sol, V_sol, W_sol = greedy_column_elimination(U_sol, V_sol, W_sol, max_rounds=10)
                final_rank = U_sol.shape[1]
                if verify_decomp(U_sol, V_sol, W_sol) and final_rank < best_rank:
                    best_rank = final_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    found = True
                    break

        if found and best_rank <= 35:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
