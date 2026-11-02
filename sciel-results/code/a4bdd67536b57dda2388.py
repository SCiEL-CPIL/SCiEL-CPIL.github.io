"""Matrix multiplication 2,3,5 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    N = n * m   # 6 (A indices)
    M = m * p   # 15 (B indices)
    P = p * n   # 10 (C indices, W[k*n+i])
    
    # Build tensor T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    # Schoolbook
    rank_sb = N * M // m  # = n*m*p = 30
    U_sb = np.zeros((N, rank_sb))
    V_sb = np.zeros((M, rank_sb))
    W_sb = np.zeros((P, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank
    # Use ALS with multiple restarts
    def als_decompose(rank, max_iter=300, seed=0):
        np.random.seed(seed)
        U = np.random.randn(N, rank)
        V = np.random.randn(M, rank)
        W = np.random.randn(P, rank)
        
        for it in range(max_iter):
            # Update U
            VW = np.einsum('br,cr->bcr', V, W).reshape(M*P, rank)
            G = VW.T @ VW
            rhs = VW.T @ T.reshape(N, M*P)
            try:
                U = np.linalg.solve(G, rhs).T
            except:
                U = np.linalg.lstsq(VW, T.reshape(N, M*P), rcond=None)[0].T
            
            # Update V
            UW = np.einsum('ar,cr->acr', U, W).reshape(N*P, rank)
            G = UW.T @ UW
            rhs = np.array([UW.T @ T[:, b, :].reshape(-1) for b in range(M)])
            try:
                V = np.linalg.solve(G, rhs).T
            except:
                V = np.linalg.lstsq(UW, T.reshape(M, N*P).T, rcond=None)[0].T
            
            # Update W
            UV = np.einsum('ar,br->abr', U, V).reshape(N*M, rank)
            G = UV.T @ UV
            rhs = np.array([UV.T @ T[:, :, c].reshape(-1) for c in range(P)])
            try:
                W = np.linalg.solve(G, rhs).T
            except:
                W = np.linalg.lstsq(UV, T.reshape(P, N*M).T, rcond=None)[0].T
        
        return U, V, W
    
    # Try ranks from lower bound upward
    # Lower bound: at least max(N, M, P) / min(n,m,p) roughly, but let's just try
    for R in range(12, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        n_tries = 40
        
        for seed in range(n_tries):
            if time.time() > deadline - 10:
                break
            
            U, V, W = als_decompose(R, max_iter=200, seed=seed)
            
            if verify(U, V, W, tol=1e-8):
                if verify(U, V, W, tol=1e-10):
                    found = True
                    best_rank = R
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
        
        if found:
            continue
    
    # Try to snap to half-integers
    def try_snap(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify(U_s, V_s, W_s, tol=1e-9):
            return U_s, V_s, W_s
        return None
    
    snapped = try_snap(best_U, best_V, best_W)
    if snapped:
        best_U, best_V, best_W = snapped
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
