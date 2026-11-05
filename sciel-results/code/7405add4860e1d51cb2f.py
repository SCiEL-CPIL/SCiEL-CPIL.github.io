"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = 3, 3, 7
    rank_sb = n * m * p  # 63
    
    # Schoolbook decomposition - always valid
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def get_residual(U, V, W):
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.linalg.norm(T_approx - T)
    
    # Precompute index sets for efficient ALS
    # For each dimension a in [0, nm), b in [0, mp), c in [0, pn)
    nm, mp, pn = n*m, m*p, p*n
    
    # Try ranks with ALS, using better initialization and more attempts
    # Start from higher ranks and work down
    for target_rank in range(62, 44, -1):
        if time.time() > deadline - 25.0:
            break
        
        # More attempts for higher ranks, fewer for lower
        max_attempts = 6 if target_rank >= 58 else 4
        found_at_rank = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 25.0:
                break
            
            # Vary the random seed
            seed = 2000 + target_rank * 100 + attempt
            np.random.seed(seed)
            
            # Structured initialization based on schoolbook
            if target_rank >= rank_sb:
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                U += np.random.randn(n*m, target_rank) * 0.05
                V += np.random.randn(m*p, target_rank) * 0.05
                W += np.random.randn(p*n, target_rank) * 0.05
            else:
                # Random init with moderate scale, try a few different scales
                scale = 0.3 + 0.1 * (attempt % 3)
                U = np.random.randn(n*m, target_rank) * scale
                V = np.random.randn(m*p, target_rank) * scale
                W = np.random.randn(p*n, target_rank) * scale
            
            # ALS iterations
            max_iters = 60
            prev_res = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 25.0:
                    break
                
                # ALS: Update U
                # T_flat[a, b*c_idx] = T[a, b, c]
                # For fixed V, W: U[a, :] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # This is a linear least squares problem in U[a, :]
                # Let M_VW be a matrix where M_VW[:, r] = kron(V[:, r], W[:, r])
                # Then T_flat[a, :] = U[a, :] @ M_VW
                # U[a, :] = T_flat[a, :] @ pinv(M_VW)
                
                Vkr = np.zeros((mp * pn, target_rank))
                for r in range(target_rank):
                    Vkr[:, r] = np.kron(V[:, r], W[:, r])
                T_flat = T.reshape(nm, -1)
                VkrT_Vkr = Vkr.T @ Vkr
                # Add small regularization for numerical stability
                VkrT_Vkr_reg = VkrT_Vkr + 1e-10 * np.eye(target_rank)
                U = (T_flat @ Vkr) @ np.linalg.pinv(VkrT_Vkr_reg)
                
                # ALS: Update V
                # For each b: T[:, b, :].reshape(-1) = V[b, :] @ M_UW
                # M_UW[:, r] = kron(U[:, r], W[:, r])
                UW = np.zeros((nm * pn, target_rank))
                for r in range(target_rank):
                    UW[:, r] = np.kron(U[:, r], W[:, r])
                UW_T_UW = UW.T @ UW
                UW_T_UW_reg = UW_T_UW + 1e-10 * np.eye(target_rank)
                V = np.empty((mp, target_rank))
                for b in range(mp):
                    V[b, :] = (T[:, b, :].reshape(-1) @ UW) @ np.linalg.pinv(UW_T_UW_reg)
                
                # ALS: Update W
                # For each c: T[:, :, c].reshape(-1) = W[c, :] @ M_UV
                # M_UV[:, r] = kron(U[:, r], V[:, r])
                UV = np.zeros((nm * mp, target_rank))
                for r in range(target_rank):
                    UV[:, r] = np.kron(U[:, r], V[:, r])
                UV_T_UV = UV.T @ UV
                UV_T_UV_reg = UV_T_UV + 1e-10 * np.eye(target_rank)
                W = np.empty((pn, target_rank))
                for c in range(pn):
                    W[c, :] = (T[:, :, c].reshape(-1) @ UV) @ np.linalg.pinv(UV_T_UV_reg)
                
                # Check convergence
                res = get_residual(U, V, W)
                if res < 1e-14:
                    if verify(U, V, W, tol=1e-10):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = target_rank
                        found_at_rank = True
                    break
                elif abs(prev_res - res) < 1e-12 and it > 10:
                    # Converged but not to zero - try a few more iterations with different strategy
                    if res < 1e-8:
                        # Very close, try a polish pass
                        for polish_it in range(20):
                            if time.time() > deadline - 25.0:
                                break
                            # Same ALS steps
                            Vkr = np.zeros((mp * pn, target_rank))
                            for r in range(target_rank):
                                Vkr[:, r] = np.kron(V[:, r], W[:, r])
                            VkrT_Vkr = Vkr.T @ Vkr + 1e-10 * np.eye(target_rank)
                            U = (T_flat @ Vkr) @ np.linalg.pinv(VkrT_Vkr)
                            
                            UW = np.zeros((nm * pn, target_rank))
                            for r in range(target_rank):
                                UW[:, r] = np.kron(U[:, r], W[:, r])
                            UW_T_UW = UW.T @ UW + 1e-10 * np.eye(target_rank)
                            for b in range(mp):
                                V[b, :] = (T[:, b, :].reshape(-1) @ UW) @ np.linalg.pinv(UW_T_UW)
                            
                            UV = np.zeros((nm * mp, target_rank))
                            for r in range(target_rank):
                                UV[:, r] = np.kron(U[:, r], V[:, r])
                            UV_T_UV = UV.T @ UV + 1e-10 * np.eye(target_rank)
                            for c in range(pn):
                                W[c, :] = (T[:, :, c].reshape(-1) @ UV) @ np.linalg.pinv(UV_T_UV)
                            
                            res = get_residual(U, V, W)
                            if res < 1e-14:
                                if verify(U, V, W, tol=1e-10):
                                    best_U = U
                                    best_V = V
                                    best_W = W
                                    best_rank = target_rank
                                    found_at_rank = True
                                break
                    break
                prev_res = res
            
            # Early exit if found at this rank
            if found_at_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
