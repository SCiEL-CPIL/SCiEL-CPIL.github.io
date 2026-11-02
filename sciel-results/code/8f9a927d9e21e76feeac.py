"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build the multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def try_rank(r, seed=42, max_iters=300):
        if time.time() > deadline - 5:
            return None
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Alternating minimization with smooth objective
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: fix V and W
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            TW = V[:, None, :] * W[None, :, :]  # (m*p, p*n, r)
            h_u = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            G_u = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            # Add small regularization for numerical stability
            G_u += 1e-10 * np.eye(r)
            try:
                U_new = np.linalg.solve(G_u, h_u.T).T
            except np.linalg.LinAlgError:
                U_new = h_u
            U = U_new
            
            # Update V: fix U and W
            UW = U[:, None, :] * W[None, :, :]  # (n*m, p*n, r)
            h_v = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_v = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            G_v += 1e-10 * np.eye(r)
            try:
                V_new = np.linalg.solve(G_v, h_v.T).T
            except np.linalg.LinAlgError:
                V_new = h_v
            V = V_new
            
            # Update W: fix U and V
            UV = U[:, None, :] * V[None, :, :]  # (n*m, m*p, r)
            h_w = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_w = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            G_w += 1e-10 * np.eye(r)
            try:
                W_new = np.linalg.solve(G_w, h_w.T).T
            except np.linalg.LinAlgError:
                W_new = h_w
            W = W_new
        
        # Final high-precision polish pass
        for it in range(50):
            if time.time() > deadline - 3:
                break
            
            TW = V[:, None, :] * W[None, :, :]
            h_u = np.einsum('abc,bcd->ad', T, TW)
            G_u = np.einsum('bcd,bce->de', TW, TW) + 1e-12 * np.eye(r)
            try:
                U = np.linalg.solve(G_u, h_u.T).T
            except np.linalg.LinAlgError:
                pass
            
            UW = U[:, None, :] * W[None, :, :]
            h_v = np.einsum('abc,acd->bd', T, UW)
            G_v = np.einsum('acd,ace->de', UW, UW) + 1e-12 * np.eye(r)
            try:
                V = np.linalg.solve(G_v, h_v.T).T
            except np.linalg.LinAlgError:
                pass
            
            UV = U[:, None, :] * V[None, :, :]
            h_w = np.einsum('abc,abd->cd', T, UV)
            G_w = np.einsum('abd,abe->de', UV, UV) + 1e-12 * np.eye(r)
            try:
                W = np.linalg.solve(G_w, h_w.T).T
            except np.linalg.LinAlgError:
                pass
        
        # Check if the decomposition is already exact (within tolerance)
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        if np.allclose(T_recon, T, atol=1e-8):
            # Round to nearest half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_recon_r = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
            if np.allclose(T_recon_r, T, atol=1e-8):
                return (r, U_r, V_r, W_r)
            
            # Try integer rounding
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            T_recon_i = np.einsum('ad,bd,cd->abc', U_i, V_i, W_i)
            if np.allclose(T_recon_i, T, atol=1e-8):
                return (r, U_i, V_i, W_i)
        
        return None
    
    # Try various ranks from lower bound up to schoolbook
    # Lower bound for matrix multiplication rank is at least n*p = 15
    # But for 3x3x5, we expect it to be higher
    ranks_to_try = list(range(15, rank_sb + 1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        # Try multiple seeds for each rank
        seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456]
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found valid for this rank, move to next rank
    
    return best_result
# EVOLVE-BLOCK-END
