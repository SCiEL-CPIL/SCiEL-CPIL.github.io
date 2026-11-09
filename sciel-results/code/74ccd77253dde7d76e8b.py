import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 130
    
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    for rank in range(rank_sb - 1, 30, -1):
        if time.time() > deadline - 40:
            break
        
        found = False
        for attempt in range(15):
            if time.time() > deadline - 40:
                break
            
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            for iteration in range(200):
                if time.time() > deadline - 40:
                    break
                
                # Solve for U: M_u @ U[a,:] = T[a,:,:].flatten()
                M_u = np.zeros((dim_v * dim_w, rank))
                for r in range(rank):
                    M_u[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                # M_u is (dim_v*dim_w, rank), we want U (dim_u, rank)
                # For each a: M_u @ U[a,:] = T[a,:,:].flatten()
                # Stack all targets: T_flat (dim_u, dim_v*dim_w)
                T_flat = T.reshape(dim_u, dim_v * dim_w)
                # U = T_flat @ pinv(M_u)
                U = T_flat @ np.linalg.pinv(M_u)
                
                # Solve for V
                M_v = np.zeros((dim_u * dim_w, rank))
                for r in range(rank):
                    M_v[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                T_flat_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                V = T_flat_v @ np.linalg.pinv(M_v)
                
                # Solve for W
                M_w = np.zeros((dim_u * dim_v, rank))
                for r in range(rank):
                    M_w[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                T_flat_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                W = T_flat_w @ np.linalg.pinv(M_w)
            
            # Check
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((recon - T) ** 2)
            
            if error < 1e-10:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
