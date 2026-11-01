import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Try to find a decomposition with rank < 50
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Try ranks from 45 down to 40
    for rank in range(45, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for restart in range(20):
            if time.time() > deadline:
                break
                
            np.random.seed(rank * 100 + restart)
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Alternating Least Squares
            for it in range(50):
                # Solve for U
                # T_a,b,c = sum_r U_a,r V_b,r W_c,r
                # For fixed a, T_a = sum_r U_a,r (V_r \otimes W_r)
                # Flatten T to 2D: T_flat (dim_u, dim_v * dim_w)
                T_flat = T.reshape(dim_u, dim_v * dim_w)
                VW = np.einsum('br,cr->bc', V, W).T # Shape (rank, dim_v * dim_w) ?
                # V is (dim_v, rank), W is (dim_w, rank)
                # V_r \otimes W_r is vector of length dim_v * dim_w
                # Let's construct M_VW of shape (dim_v * dim_w, rank)
                M_VW = np.einsum('br,cr->bc', V, W).reshape(dim_v * dim_w, rank)
                # T_flat = U @ M_VW.T ?
                # T_a,b,c = sum_r U_a,r M_VW_bc,r
                # So T_flat = U @ M_VW.T
                U, residuals, rank_u, s = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)
                U = U.T
                
                # Solve for V
                T_flat_v = T.reshape(dim_v, dim_u * dim_w)
                # T_b,a,c = sum_r V_b,r U_a,r W_c,r
                # M_UW of shape (dim_u * dim_w, rank)
                M_UW = np.einsum('ar,cr->ac', U, W).reshape(dim_u * dim_w, rank)
                V, residuals, rank_v, s = np.linalg.lstsq(M_UW.T, T_flat_v.T, rcond=None)
                V = V.T
                
                # Solve for W
                T_flat_w = T.reshape(dim_w, dim_u * dim_v)
                # T_c,a,b = sum_r W_c,r U_a,r V_b,r
                # M_UV of shape (dim_u * dim_v, rank)
                M_UV = np.einsum('ar,br->ab', U, V).reshape(dim_u * dim_v, rank)
                W, residuals, rank_w, s = np.linalg.lstsq(M_UV.T, T_flat_w.T, rcond=None)
                W = W.T
                
            # Check exactness
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_recon, T, atol=1e-9):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == rank:
            break

    if best_U is None:
        # Fallback to schoolbook
        rank = n * m * p
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
        
    # Round to nearest 0.5
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify again after rounding
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback
        rank = n * m * p
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
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
