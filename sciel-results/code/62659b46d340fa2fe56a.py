import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        """Return schoolbook decomposition as dict of lists."""
        rank = n * m * p  # 50
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)

    def try_rank_als(rank, max_iters=100, tol=1e-10):
        """Try to find a decomposition using ALS."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + rank)
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # Normalize to prevent divergence
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 0:
                scale = (1.0 / norm) ** (1.0/3.0)
                U[:, r] *= scale
                V[:, r] *= scale
                W[:, r] *= scale
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # T_u = einsum('abc,br,cr->ar', target, V, W)
            T_u = np.einsum('abc,br,cr->ar', target, V, W)
            # Gram matrix for U: G_u = (V^T V) * (W^T W) element-wise
            G_u = (V.T @ V) * (W.T @ W)
            G_u += 1e-8 * np.eye(rank)
            U_new = np.linalg.solve(G_u, T_u.T).T
            
            # Update V
            T_v = np.einsum('abc,ar,cr->br', target, U_new, W)
            G_v = (U_new.T @ U_new) * (W.T @ W)
            G_v += 1e-8 * np.eye(rank)
            V_new = np.linalg.solve(G_v, T_v.T).T
            
            # Update W
            T_w = np.einsum('abc,ar,br->cr', target, U_new, V_new)
            G_w = (U_new.T @ U_new) * (V_new.T @ V_new)
            G_w += 1e-8 * np.eye(rank)
            W_new = np.linalg.solve(G_w, T_w.T).T
            
            # Check convergence
            err = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if err < tol:
                break
        
        # Snap to nearest half-integer
        u_snap = np.round(U * 2.0) / 2.0
        v_snap = np.round(V * 2.0) / 2.0
        w_snap = np.round(W * 2.0) / 2.0
        
        if verify_decomposition(u_snap, v_snap, w_snap):
            return (u_snap, v_snap, w_snap)
            
        # Try cleaning small values
        u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
        v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
        w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
        if verify_decomposition(u_clean, v_clean, w_clean):
            return (u_clean, v_clean, w_clean)
            
        # Try snapping to integers only
        u_int = np.round(U)
        v_int = np.round(V)
        w_int = np.round(W)
        if verify_decomposition(u_int, v_int, w_int):
            return (u_int, v_int, w_int)
            
        return None

    best_result = None
    
    # Try ranks
    for target_rank in [30, 32, 35, 40]:
        if time.time() > deadline:
            break
        
        result = try_rank_als(target_rank, max_iters=200)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
