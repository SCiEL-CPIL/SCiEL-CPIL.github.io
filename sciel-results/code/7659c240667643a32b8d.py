import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS

    # 1. Construct the known Rank-50 Solution (Strassen + Schoolbook)
    # This serves as the fallback and the initialization for ALS.
    
    rank_50 = 50
    U50 = np.zeros((n * m, rank_50), dtype=np.float64)
    V50 = np.zeros((m * p, rank_50), dtype=np.float64)
    W50 = np.zeros((p * n, rank_50), dtype=np.float64)
    
    col = 0
    
    def emit_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A11 + A22) * (B11 + B22)
        U50[i0*m + j0, col] = 1.0
        U50[(i0+1)*m + (j0+1), col] = 1.0
        V50[j0*p + k0, col] = 1.0
        V50[(j0+1)*p + (k0+1), col] = 1.0
        W50[k0*n + i0, col] = 1.0
        W50[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M2 = (A21 + A22) * B11
        U50[(i0+1)*m + j0, col] = 1.0
        U50[(i0+1)*m + (j0+1), col] = 1.0
        V50[j0*p + k0, col] = 1.0
        W50[k0*n + (i0+1), col] = 1.0
        W50[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U50[i0*m + j0, col] = 1.0
        V50[j0*p + (k0+1), col] = 1.0
        V50[(j0+1)*p + (k0+1), col] = -1.0
        W50[(k0+1)*n + i0, col] = 1.0
        W50[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U50[(i0+1)*m + (j0+1), col] = 1.0
        V50[(j0+1)*p + k0, col] = 1.0
        V50[j0*p + k0, col] = -1.0
        W50[k0*n + (i0+1), col] = 1.0
        W50[k0*n + i0, col] = 1.0
        col += 1
        
        # M5 = (A11 + A12) * B22
        U50[i0*m + j0, col] = 1.0
        U50[i0*m + (j0+1), col] = 1.0
        V50[(j0+1)*p + (k0+1), col] = 1.0
        W50[k0*n + i0, col] = -1.0
        W50[(k0+1)*n + i0, col] = 1.0
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U50[(i0+1)*m + j0, col] = 1.0
        U50[i0*m + j0, col] = -1.0
        V50[j0*p + k0, col] = 1.0
        V50[j0*p + (k0+1), col] = 1.0
        W50[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U50[i0*m + (j0+1), col] = 1.0
        U50[(i0+1)*m + (j0+1), col] = -1.0
        V50[(j0+1)*p + k0, col] = 1.0
        V50[(j0+1)*p + (k0+1), col] = 1.0
        W50[k0*n + i0, col] = 1.0
        col += 1

    def emit_schoolbook_2x2x1(i0, j0, k0):
        nonlocal col
        U50[i0*m + j0, col] = 1.0
        V50[j0*p + k0, col] = 1.0
        W50[k0*n + i0, col] = 1.0
        col += 1
        
        U50[i0*m + (j0+1), col] = 1.0
        V50[(j0+1)*p + k0, col] = 1.0
        W50[k0*n + i0, col] = 1.0
        col += 1
        
        U50[(i0+1)*m + j0, col] = 1.0
        V50[j0*p + k0, col] = 1.0
        W50[k0*n + (i0+1), col] = 1.0
        col += 1
        
        U50[(i0+1)*m + (j0+1), col] = 1.0
        V50[(j0+1)*p + k0, col] = 1.0
        W50[k0*n + (i0+1), col] = 1.0
        col += 1

    # Build Rank 50
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 0, 4)
    emit_schoolbook_2x2x1(0, 0, 6)
    
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_strassen(0, 2, 4)
    emit_schoolbook_2x2x1(0, 2, 6)
    
    assert col == 50, f"Rank 50 construction failed, col={col}"

    # 2. Prepare Target Tensor Unfoldings
    # T[i, j, k] = 1
    T1 = np.zeros((n * m, m * p))
    T2 = np.zeros((m * p, p * n))
    T3 = np.zeros((p * n, n * m))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T1[i * m + j, j * p + k] = 1.0
                T2[j * p + k, k * n + i] = 1.0
                T3[k * n + i, i * m + j] = 1.0

    def verify(U, V, W):
        # T = sum_r U[:,r] V[:,r] W[:,r]
        # Check T1
        # T1[i, jk] = sum_r U[i, r] V[j, r] W[k, r]
        # This is U @ (V * W) where * is Khatri-Rao
        # We can just reconstruct T1 and check
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        # Efficient check:
        # T1[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We can use einsum on the 3D reshapes
        # But for speed, we can check the Frobenius norm of the difference
        # T1_recon = np.einsum('ijr,jkr->ijk', U3, V3) * W3? No.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Let's just do the full reconstruction for verification since it's small (8x4x7 = 224 elements)
        T_recon = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            T_recon += U[:, r].reshape(n, m)[:, :, np.newaxis] * \
                       V[:, r].reshape(m, p)[np.newaxis, :, :] * \
                       W[:, r].reshape(p, n)[:, np.newaxis, :]
            
        # Target T is all 1s
        if not np.allclose(T_recon, 1.0, atol=1e-9):
            return False
        return True

    # 3. ALS Optimization
    def khatri_rao(A, B):
        # A: (M, R), B: (N, R) -> (M*N, R)
        M, R = A.shape
        N, _ = B.shape
        # Vectorized Kronecker product for each column
        # np.kron is slow, use broadcasting
        # A[:, r, None] * B[None, :, r] -> (M, N, R) -> reshape to (M*N, R)
        return (A[:, np.newaxis, :] * B[np.newaxis, :, :]).reshape(M * N, R)

    def als_step(T_target, U, V, W, max_iters=200, tol=1e-12):
        R = U.shape[1]
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        for it in range(max_iters):
            # Solve for U
            # T1 = U @ KR(V, W)^T
            KR_VW = khatri_rao(V_curr, W_curr)
            # Solve U @ KR_VW^T = T1  =>  U = T1 @ KR_VW @ (KR_VW^T @ KR_VW)^-1
            # This is a least squares problem: U = argmin || T1 - U KR_VW^T ||
            # Equivalent to solving for each row of U? No, U is (n*m, R).
            # T1 is (n*m, m*p). KR_VW is (m*p, R).
            # We want U (n*m, R) such that U @ KR_VW.T = T1.
            # This is overdetermined if R < m*p.
            # We can use np.linalg.lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(KR_VW.T, T1.T, rcond=None)
                U_new = U_new.T
            except:
                break
                
            # Solve for V
            KR_WU = khatri_rao(W_curr, U_new)
            try:
                V_new, _, _, _ = np.linalg.lstsq(KR_WU.T, T2.T, rcond=None)
                V_new = V_new.T
            except:
                break
                
            # Solve for W
            KR_UV = khatri_rao(U_new, V_new)
            try:
                W_new, _, _, _ = np.linalg.lstsq(KR_UV.T, T3.T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            change = np.linalg.norm(U_new - U_curr) + np.linalg.norm(V_new - V_curr) + np.linalg.norm(W_new - W_curr)
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            if change < tol:
                break
                
        return U_curr, V_curr, W_curr

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50
    
    rng = np.random.default_rng(42)
    
    # Try to reduce rank
    # Start from 49 down to 28
    for target_rank in range(49, 27, -1):
        if time.time() > DEADLINE - 10:
            break
            
        # Limit restarts based on time
        max_restarts = 10
        if time.time() > DEADLINE - 30:
            max_restarts = 2
            
        found = False
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 5:
                break
                
            # Initialization
            if target_rank == 49:
                # Drop 1 column from rank-50 solution
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                # Add small noise
                U_init += rng.normal(0, 0.01, U_init.shape)
                V_init += rng.normal(0, 0.01, V_init.shape)
                W_init += rng.normal(0, 0.01, W_init.shape)
            else:
                # Random init
                U_init = rng.normal(0, 1, (n * m, target_rank))
                V_init = rng.normal(0, 1, (m * p, target_rank))
                W_init = rng.normal(0, 1, (p * n, target_rank))
            
            U_sol, V_sol, W_sol = als_step(T1, U_init, V_init, W_init)
            
            if verify(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                found = True
                break
        
        if found:
            # If we found a lower rank, we might try even lower
            # But we loop downwards, so we just continue
            pass

    # Final Verification and Return
    if verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to Rank 50
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
# EVOLVE-BLOCK-END
