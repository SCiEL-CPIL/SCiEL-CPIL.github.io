import numpy as np
import time
import random

def construct():
    n, m, p = 2, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                entries.append((u_idx, v_idx, w_idx))
                
    T = np.zeros((nm, mp, pn), dtype=float)
    for u, v, w in entries:
        T[u, v, w] = 1.0
        
    def verify(U, V, W):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        U_round = np.round(U_np * 2) / 2
        V_round = np.round(V_np * 2) / 2
        W_round = np.round(W_np * 2) / 2
        
        T_recon = np.einsum('ur,vr,wr->uvw', U_round, V_round, W_round)
        
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        return True

    def als_decompose(rank, max_iters=50, tol=1e-6):
        U = np.random.rand(nm, rank) * 2 - 1
        V = np.random.rand(mp, rank) * 2 - 1
        W = np.random.rand(pn, rank) * 2 - 1
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            # Update U
            # B shape: (mp*pn, rank)
            B = np.zeros((mp * pn, rank))
            for r in range(rank):
                B[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
            # Precompute L = (B^T B)^{-1} B^T
            # Use pinv for stability? Or just solve.
            # B^T B is rank x rank.
            BtB = B.T @ B
            # Add small regularization to avoid singularity
            BtB += 1e-8 * np.eye(rank)
            Bt = B.T
            # L = inv(BtB) @ Bt
            try:
                L = np.linalg.solve(BtB, Bt)
            except np.linalg.LinAlgError:
                L = np.linalg.pinv(BtB) @ Bt
                
            for u in range(nm):
                t_u = T[u, :, :].flatten()
                U[u, :] = L @ t_u
                
            # Update V
            B_v = np.zeros((nm * pn, rank))
            for r in range(rank):
                B_v[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
            BtB_v = B_v.T @ B_v + 1e-8 * np.eye(rank)
            Bt_v = B_v.T
            try:
                L_v = np.linalg.solve(BtB_v, Bt_v)
            except np.linalg.LinAlgError:
                L_v = np.linalg.pinv(BtB_v) @ Bt_v
                
            for v in range(mp):
                t_v = T[:, v, :].flatten()
                V[v, :] = L_v @ t_v
                
            # Update W
            B_w = np.zeros((nm * mp, rank))
            for r in range(rank):
                B_w[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
            BtB_w = B_w.T @ B_w + 1e-8 * np.eye(rank)
            Bt_w = B_w.T
            try:
                L_w = np.linalg.solve(BtB_w, Bt_w)
            except np.linalg.LinAlgError:
                L_w = np.linalg.pinv(BtB_w) @ Bt_w
                
            for w in range(pn):
                t_w = T[:, :, w].flatten()
                W[w, :] = L_w @ t_w
                
            # Calculate error
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T - T_recon)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W

    best_rank = nm * mp * pn
    best_U, best_V, best_W = None, None, None
    
    deadline = time.time() + 110
    
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
            
        max_restarts = 5 if rank < 35 else 2
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            U, V, W = als_decompose(rank, max_iters=40)
            
            if verify(U, V, W):
                best_rank = rank
                best_U = np.round(U * 2) / 2
                best_V = np.round(V * 2) / 2
                best_W = np.round(W * 2) / 2
                break
        else:
            continue
        break
        
    if best_U is None:
        best_U = np.zeros((nm, best_rank))
        best_V = np.zeros((mp, best_rank))
        best_W = np.zeros((pn, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    best_U[u_idx, col] = 1
                    best_V[v_idx, col] = 1
                    best_W[w_idx, col] = 1
                    col += 1
        best_rank = 50

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
