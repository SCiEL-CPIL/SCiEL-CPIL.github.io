"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    dimU, dimV, dimW = n*m, m*p, p*n  # 6, 9, 6
    
    # Build target tensor
    T = np.zeros((dimU, dimV, dimW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank 18
    rank_school = n * m * p
    U_school = np.zeros((dimU, rank_school))
    V_school = np.zeros((dimV, rank_school))
    W_school = np.zeros((dimW, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.zeros((dimU, dimV, dimW))
        for c in range(U.shape[1]):
            T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        return np.allclose(T_recon, T, atol=atol)
    
    def try_rank(target_rank, max_trials=30):
        """Try to find a rank-target_rank decomposition."""
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE - 25:
                return None
            
            np.random.seed(999 + trial * 7)
            U = np.random.randn(dimU, target_rank)
            V = np.random.randn(dimV, target_rank)
            W = np.random.randn(dimW, target_rank)
            
            # ALS optimization
            prev_err = np.inf
            for iteration in range(200):
                if time.time() - start_time > DEADLINE - 25:
                    break
                
                # Update U: for each a, U[a,:] solves least squares
                for a in range(dimU):
                    # T[a,:,:] = sum_c U[a,c] * V[:,c] * W[:,c]
                    # For fixed a: T[a,:,:] = sum_c U[a,c] * (V[:,c] * W[:,c])
                    # Let M_c = V[:,c] * W[:,c] (element-wise product, shape dimV x dimW)
                    # Then T[a,:,:] = sum_c U[a,c] * M_c
                    # This is a linear system in U[a,:]
                    A_mat = np.zeros((dimV * dimW, target_rank))
                    for c in range(target_rank):
                        A_mat[:, c] = (V[:, c] * W[:, c]).flatten()
                    b_vec = T[a, :, :].flatten()
                    # Solve A_mat @ U[a,:] = b_vec
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
                
                # Update V
                for b in range(dimV):
                    A_mat = np.zeros((dimU * dimW, target_rank))
                    for c in range(target_rank):
                        A_mat[:, c] = (U[:, c] * W[:, c]).flatten()
                    b_vec = T[:, b, :].flatten()
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
                
                # Update W
                for c_idx in range(dimW):
                    A_mat = np.zeros((dimU * dimV, target_rank))
                    for c in range(target_rank):
                        A_mat[:, c] = (U[:, c] * V[:, c]).flatten()
                    b_vec = T[:, :, c_idx].flatten()
                    try:
                        W[c_idx, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        pass
                
                # Check error
                T_recon = np.zeros((dimU, dimV, dimW))
                for c in range(target_rank):
                    T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                err = np.linalg.norm(T_recon - T)
                
                if abs(prev_err - err) < 1e-10:
                    break
                prev_err = err
                
                if err < 1e-8:
                    # Try to snap to rationals
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if check_decomposition(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                    # Also try integers
                    U_snap2 = np.round(U)
                    V_snap2 = np.round(V)
                    W_snap2 = np.round(W)
                    if check_decomposition(U_snap2, V_snap2, W_snap2):
                        return U_snap2, V_snap2, W_snap2
                    # Try quarters
                    U_snap3 = np.round(U * 4) / 4
                    V_snap3 = np.round(V * 4) / 4
                    W_snap3 = np.round(W * 4) / 4
                    if check_decomposition(U_snap3, V_snap3, W_snap3):
                        return U_snap3, V_snap3, W_snap3
        
        return None
    
    # Try rank 16 and 17
    for target_rank in [16, 17]:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

