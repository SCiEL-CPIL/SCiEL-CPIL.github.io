"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
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
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute index mappings for fast tensor operations
    # For each a (index in n*m), which (b,c) pairs have T[a,b,c]=1?
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Given a=i*m+j, b=j*p+k, c=k*n+i:
    #   j = a % m, i = a // m
    #   k = b % p
    #   c must equal k*n + i
    
    # Build sparse structure: for each a, list of (b, c) where T=1
    T_sparse = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if a not in T_sparse:
                    T_sparse[a] = []
                T_sparse[a].append((b, c))
    
    # Also build for each b: list of (a, c) where T=1
    T_sparse_b = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if b not in T_sparse_b:
                    T_sparse_b[b] = []
                T_sparse_b[b].append((a, c))
    
    # And for each c: list of (a, b) where T=1
    T_sparse_c = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if c not in T_sparse_c:
                    T_sparse_c[c] = []
                T_sparse_c[c].append((a, b))
    
    def compute_residual_norm(U, V, W):
        """Compute ||T - sum_d U[:,d]⊗V[:,d]⊗W[:,d]||_F^2 efficiently"""
        res = 0.0
        for a in range(n * m):
            for b, c in T_sparse[a]:
                val = 0.0
                for d in range(U.shape[1]):
                    val += U[a, d] * V[b, d] * W[c, d]
                res += (1.0 - val) ** 2
        return res
    
    def try_rank(r, seed=42, max_iter=200):
        if time.time() > deadline - 5:
            return None
        np.random.seed(seed)
        
        # Initialize with random values, scaled
        scale = 1.0 / (r ** 0.5)
        U = np.random.randn(n * m, r) * scale
        V = np.random.randn(m * p, r) * scale
        W = np.random.randn(p * n, r) * scale
        
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            # For each a: minimize sum_{(b,c) in T_sparse[a]} (T[a,b,c] - sum_d U[a,d]*V[b,d]*W[c,d])^2
            # Linear in U[a,:]: 
            #   G_U[d1,d2] = sum_{(b,c) in T_sparse[a]} V[b,d1]*W[c,d1]*V[b,d2]*W[c,d2]
            #   h_U[d] = sum_{(b,c) in T_sparse[a]} T[a,b,c] * V[b,d] * W[c,d]
            #   G_U @ U[a,:] = h_U
            # But G_U depends on a! So we need per-a Gram matrices.
            # For efficiency, since T is sparse, each a has few (b,c) pairs.
            
            for a in range(n * m):
                pairs = T_sparse[a]
                if not pairs:
                    continue
                # Build G_U (r x r) and h_U (r,)
                G_U = np.zeros((r, r))
                h_U = np.zeros(r)
                for b, c in pairs:
                    vw = V[b, :] * W[c, :]  # (r,)
                    G_U += np.outer(vw, vw)
                    h_U += T[a, b, c] * vw
                try:
                    U[a, :] = np.linalg.solve(G_U + 1e-10 * np.eye(r), h_U)
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(G_U + 1e-10 * np.eye(r), h_U, rcond=None)[0]
            
            # Update V
            for b in range(m * p):
                pairs = T_sparse_b[b]
                if not pairs:
                    continue
                G_V = np.zeros((r, r))
                h_V = np.zeros(r)
                for a, c in pairs:
                    uw = U[a, :] * W[c, :]  # (r,)
                    G_V += np.outer(uw, uw)
                    h_V += T[a, b, c] * uw
                try:
                    V[b, :] = np.linalg.solve(G_V + 1e-10 * np.eye(r), h_V)
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(G_V + 1e-10 * np.eye(r), h_V, rcond=None)[0]
            
            # Update W
            for c in range(p * n):
                pairs = T_sparse_c[c]
                if not pairs:
                    continue
                G_W = np.zeros((r, r))
                h_W = np.zeros(r)
                for a, b in pairs:
                    uv = U[a, :] * V[b, :]  # (r,)
                    G_W += np.outer(uv, uv)
                    h_W += T[a, b, c] * uv
                try:
                    W[c, :] = np.linalg.solve(G_W + 1e-10 * np.eye(r), h_W)
                except np.linalg.LinAlgError:
                    W[c, :] = np.linalg.lstsq(G_W + 1e-10 * np.eye(r), h_W, rcond=None)[0]
            
            # Check convergence
            if it % 10 == 0 or it == max_iter - 1:
                res = compute_residual_norm(U, V, W)
                if res < best_residual:
                    best_residual = res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if res < 1e-12:
                    break
        
        U, V, W = best_U, best_V, best_W
        
        # Try rounding to integers first
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        T_recon_i = np.einsum('ad,bd,cd->abc', U_i, V_i, W_i)
        if np.allclose(T_recon_i, T, atol=1e-9):
            return (r, U_i, V_i, W_i)
        
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_recon_r = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
        if np.allclose(T_recon_r, T, atol=1e-9):
            return (r, U_r, V_r, W_r)
        
        # Try local search: perturb and re-round
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            np.random.seed(seed * 1000 + attempt)
            U_p = U_i + np.random.randn(n * m, r) * 0.1
            V_p = V_i + np.random.randn(m * p, r) * 0.1
            W_p = W_i + np.random.randn(p * n, r) * 0.1
            U_p = np.round(U_p)
            V_p = np.round(V_p)
            W_p = np.round(W_p)
            T_recon_p = np.einsum('ad,bd,cd->abc', U_p, V_p, W_p)
            if np.allclose(T_recon_p, T, atol=1e-9):
                return (r, U_p, V_p, W_p)
        
        # Try half-integer local search
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            np.random.seed(seed * 2000 + attempt)
            U_p = U_r + np.random.randn(n * m, r) * 0.1
            V_p = V_r + np.random.randn(m * p, r) * 0.1
            W_p = W_r + np.random.randn(p * n, r) * 0.1
            U_p = np.round(U_p * 2) / 2
            V_p = np.round(V_p * 2) / 2
            W_p = np.round(W_p * 2) / 2
            T_recon_p = np.einsum('ad,bd,cd->abc', U_p, V_p, W_p)
            if np.allclose(T_recon_p, T, atol=1e-9):
                return (r, U_p, V_p, W_p)
        
        return None
    
    # Try various ranks
    # Lower bound: max(n, m, p) * min(n, m, p) but for matmul it's at least n*p=15
    # Known: 3x3x3 has rank 23 (Strassen-like), so 3x3x5 likely needs more
    # Let's try from 15 upward
    
    ranks_to_try = list(range(15, rank_sb + 1))
    
    seeds_list = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 101, 202, 303, 404, 505]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        for seed in seeds_list:
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
                break
    
    return best_result
# EVOLVE-BLOCK-END
