"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # Build the multiplication tensor T[a, b, c] where:
    # a = i*m + j (row of A, col of A)
    # b = j*p + k (row of B, col of B)  
    # c = k*n + i (row of output, col of output)
    # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
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
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T (after rounding to half-integers)."""
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)
    
    def try_als(r, seed, max_iters=300):
        """Alternating least squares for rank r decomposition."""
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                return None
            
            # Update U: for each a, solve G_U @ U[a,:] = h_U[a,:]
            # where G_U[d1,d2] = sum_{b,c} V[b,d1]*W[c,d1]*V[b,d2]*W[c,d2]
            # and h_U[a,d] = sum_{b,c} T[a,b,c]*V[b,d]*W[c,d]
            
            # Efficient computation using einsum
            # TW[b,c,d] = V[b,d] * W[c,d]
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (m*p, p*n, r)
            
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            try:
                U_new = np.linalg.solve(G_U + 1e-10 * np.eye(r), h_U.T).T
            except np.linalg.LinAlgError:
                U_new = h_U
            U = U_new
            
            # Update V
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            
            try:
                V_new = np.linalg.solve(G_V + 1e-10 * np.eye(r), h_V.T).T
            except np.linalg.LinAlgError:
                V_new = h_V
            V = V_new
            
            # Update W
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            
            try:
                W_new = np.linalg.solve(G_W + 1e-10 * np.eye(r), h_W.T).T
            except np.linalg.LinAlgError:
                W_new = h_W
            W = W_new
        
        # Try rounding to half-integers
        for scale in [2, 1]:  # half-integers first, then integers
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for tensor rank of matrix multiplication: at least max(n*m, m*p, p*n)
    # but really the bilinear complexity lower bound
    # For 3x3x5: n*m=9, m*p=15, p*n=15
    # Start from a reasonable lower bound
    
    lower_bound = max(n * m, m * p, p * n)  # 15
    # Actually, the rank must be at least the "slice rank" lower bound
    # Let's just try from lower_bound upward
    
    ranks_to_try = list(range(lower_bound, rank_sb + 1))
    
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000]
    
    for r in ranks_to_try:
        if time.time() > deadline - 3:
            break
        if r >= best_rank:
            break
        
        for seed in seeds:
            if time.time() > deadline - 3:
                break
            
            result = try_als(r, seed)
            if result is not None:
                U_f, V_f, W_f = result
                if r < best_rank:
                    best_rank = r
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found valid for this rank, move to next rank
    
    return best_result
# EVOLVE-BLOCK-END
