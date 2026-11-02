"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    
    # Precompute index mappings for efficient updates
    # For each (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
    idx_ijk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_ijk.append((i * m + j, j * p + k, k * n + i))
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose T"""
        # Compute full tensor reconstruction
        # T_recon[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        
        # Check entries that should be 1
        for (a, b, c) in idx_ijk:
            if abs(T_recon[a, b, c] - 1.0) > tol:
                return False
        
        # Check entries that should be 0
        # Create a mask for the 1s
        mask = np.zeros(T.shape, dtype=bool)
        for (a, b, c) in idx_ijk:
            mask[a, b, c] = True
            
        # All other entries must be close to 0
        zero_vals = T_recon[~mask]
        if np.any(np.abs(zero_vals) > tol):
            return False
            
        return True
    
    def try_rank(r, seed=42, max_iters=200):
        if time.time() > deadline - 3:
            return None
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                return None
            
            # Update U: for each a, solve linear system
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # For each a: sum_{b,c} T[a,b,c] * V[b,d] * W[c,d] = sum_{b,c} T[a,b,c] * sum_d' U[a,d'] * V[b,d'] * W[c,d']
            # => G_U @ U[a,:] = h_U[a,:]
            # where G_U[d,d'] = sum_{b,c} T[a,b,c] * V[b,d] * W[c,d] * V[b,d'] * W[c,d']
            #       h_U[a,d] = sum_{b,c} T[a,b,c] * V[b,d] * W[c,d]
            
            # Precompute V*W products: TW[b,c,d] = V[b,d]*W[c,d]
            # Efficient: TW = V[:,None,:] * W[None,:,:] shape (m*p, p*n, r)
            TW = V[:, None, :] * W[None, :, :]  # (m*p, p*n, r)
            
            # h_U[a,d] = sum_{b,c} T[a,b,c] * TW[b,c,d]
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            
            # G_U[d,d'] = sum_{a,b,c} T[a,b,c] * TW[b,c,d] * TW[b,c,d']
            # = sum_{b,c} TW[b,c,d] * TW[b,c,d'] * (sum_a T[a,b,c])
            # sum_a T[a,b,c] = 1 if (b,c) corresponds to valid (j,k) with some i, else 0
            # Actually T[a,b,c] is 1 only for specific (a,b,c) triples
            # G_U[d,d'] = sum_{b,c} TW[b,c,d] * TW[b,c,d'] * S[b,c]
            # where S[b,c] = sum_a T[a,b,c]
            
            S_bc = T.sum(axis=0)  # (m*p, p*n)
            # G_U[d,d'] = sum_{b,c} S[b,c] * TW[b,c,d] * TW[b,c,d']
            G_U = np.einsum('bc,bcd,bce->de', S_bc, TW, TW)  # (r, r)
            
            try:
                U_new = np.linalg.solve(G_U, h_U.T).T
            except np.linalg.LinAlgError:
                U_new = h_U
            U = U_new
            
            # Update V
            UW = U[:, None, :] * W[None, :, :]  # (n*m, p*n, r)
            S_ac = T.sum(axis=1)  # (n*m, p*n)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_V = np.einsum('ac,acd,ace->de', S_ac, UW, UW)  # (r, r)
            try:
                V_new = np.linalg.solve(G_V, h_V.T).T
            except np.linalg.LinAlgError:
                V_new = h_V
            V = V_new
            
            # Update W
            UV = U[:, None, :] * V[None, :, :]  # (n*m, m*p, r)
            S_ab = T.sum(axis=2)  # (n*m, m*p)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_W = np.einsum('ab,abd,abe->de', S_ab, UV, UV)  # (r, r)
            try:
                W_new = np.linalg.solve(G_W, h_W.T).T
            except np.linalg.LinAlgError:
                W_new = h_W
            W = W_new
        
        # Try rounding to half-integers and integers
        for scale, offset in [(2, 0), (1, 0)]:
            U_r = np.round(U * scale + offset) / scale - offset / scale
            V_r = np.round(V * scale + offset) / scale - offset / scale
            W_r = np.round(W * scale + offset) / scale - offset / scale
            
            if check_exact(U_r, V_r, W_r):
                return (r, U_r, V_r, W_r)
        
        # Try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_exact(U_i, V_i, W_i):
            return (r, U_i, V_i, W_i)
        
        return None
    
    # Try various ranks
    # For 3x3x5, known lower bounds suggest rank ~ 15-20.
    # We will try ranks from 15 to 44.
    # To increase chances, we use more seeds for lower ranks.
    
    # Strategy: Try a few ranks with many seeds, or one rank with many seeds?
    # Since time is limited, we'll iterate ranks and use a fixed number of seeds per rank.
    # If we find a solution, we return immediately (since we are searching for minimal rank, 
    # and we go from low to high, the first one found is the best for that rank, 
    # but we should check lower ranks first).
    
    ranks_to_try = list(range(15, rank_sb))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
            
        # Use more seeds for lower ranks to ensure we find a solution if it exists
        num_seeds = 10 if r < 20 else 5
        seeds = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        found = False
        for seed in seeds[:num_seeds]:
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
                found = True
                break
        
        # If we found a solution for rank r, we can stop because we are looking for minimal rank
        # and we are iterating r from low to high.
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
