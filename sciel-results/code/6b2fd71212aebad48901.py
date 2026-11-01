"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, permutations

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
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return bool(np.allclose(T_recon, T, atol=tol))
    
    def polish(U, V, W, r, max_iters=50):
        """Local polish: small perturbations to help rounding converge."""
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            # Small random perturbation
            eps = 0.01
            U = U + eps * np.random.randn(n * m, r)
            V = V + eps * np.random.randn(m * p, r)
            W = W + eps * np.random.randn(p * n, r)
            
            # One ALS step
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            h_U = np.einsum('abc,bcd->ad', T, TW)
            G_U = np.einsum('bcd,bce->de', TW, TW)
            try:
                U = np.linalg.solve(G_U + 1e-8 * np.eye(r), h_U.T).T
            except np.linalg.LinAlgError:
                pass
            
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            h_V = np.einsum('abc,acd->bd', T, UW)
            G_V = np.einsum('acd,ace->de', UW, UW)
            try:
                V = np.linalg.solve(G_V + 1e-8 * np.eye(r), h_V.T).T
            except np.linalg.LinAlgError:
                pass
            
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            h_W = np.einsum('abc,abd->cd', T, UV)
            G_W = np.einsum('abd,abe->de', UV, UV)
            try:
                W = np.linalg.solve(G_W + 1e-8 * np.eye(r), h_W.T).T
            except np.linalg.LinAlgError:
                pass
        
        # Try rounding to half-integers and integers
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        return None
    
    def try_als(r, seed, max_iters=200):
        """Alternating least squares for rank r decomposition."""
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                return None
            
            # Update U
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            h_U = np.einsum('abc,bcd->ad', T, TW)
            G_U = np.einsum('bcd,bce->de', TW, TW)
            try:
                U = np.linalg.solve(G_U + 1e-10 * np.eye(r), h_U.T).T
            except np.linalg.LinAlgError:
                U = h_U
            
            # Update V
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            h_V = np.einsum('abc,acd->bd', T, UW)
            G_V = np.einsum('acd,ace->de', UW, UW)
            try:
                V = np.linalg.solve(G_V + 1e-10 * np.eye(r), h_V.T).T
            except np.linalg.LinAlgError:
                V = h_V
            
            # Update W
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            h_W = np.einsum('abc,abd->cd', T, UV)
            G_W = np.einsum('abd,abe->de', UV, UV)
            try:
                W = np.linalg.solve(G_W + 1e-10 * np.eye(r), h_W.T).T
            except np.linalg.LinAlgError:
                W = h_W
            
            # Check convergence and try rounding every 20 iterations
            if it % 20 == 19:
                for scale in [2, 1]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if verify_decomposition(U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
        
        # Final rounding attempt
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        
        # Try polish
        result = polish(U, V, W, r, max_iters=30)
        if result is not None:
            return result
        
        return None
    
    def try_structured_init(r, seed):
        """Try structured initializations with sparse patterns."""
        np.random.seed(seed)
        
        # Generate random sparse initializations
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for d in range(r):
            # Randomly assign a few non-zero entries per factor
            for _ in range(np.random.randint(1, 4)):
                U[np.random.randint(n * m), d] = np.random.choice([-1, 1])
            for _ in range(np.random.randint(1, 4)):
                V[np.random.randint(m * p), d] = np.random.choice([-1, 1])
            for _ in range(np.random.randint(1, 4)):
                W[np.random.randint(p * n), d] = np.random.choice([-1, 1])
        
        # Run ALS from this structured start
        for it in range(150):
            if time.time() > deadline - 3:
                return None
            
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            h_U = np.einsum('abc,bcd->ad', T, TW)
            G_U = np.einsum('bcd,bce->de', TW, TW)
            try:
                U = np.linalg.solve(G_U + 1e-10 * np.eye(r), h_U.T).T
            except np.linalg.LinAlgError:
                U = h_U
            
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            h_V = np.einsum('abc,acd->bd', T, UW)
            G_V = np.einsum('acd,ace->de', UW, UW)
            try:
                V = np.linalg.solve(G_V + 1e-10 * np.eye(r), h_V.T).T
            except np.linalg.LinAlgError:
                V = h_V
            
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            h_W = np.einsum('abc,abd->cd', T, UV)
            G_W = np.einsum('abd,abe->de', UV, UV)
            try:
                W = np.linalg.solve(G_W + 1e-10 * np.eye(r), h_W.T).T
            except np.linalg.LinAlgError:
                W = h_W
            
            if it % 25 == 24:
                for scale in [2, 1]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if verify_decomposition(U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
        
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        
        return polish(U, V, W, r, max_iters=20)
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n * m, m * p, p * n)  # 15
    
    ranks_to_try = list(range(lower_bound, min(rank_sb + 1, lower_bound + 10)))
    
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 2024, 31415, 27182, 16180, 9999, 8888, 7777, 6666]
    
    for r in ranks_to_try:
        if time.time() > deadline - 3:
            break
        if r >= best_rank:
            break
        
        # Try both random and structured initializations
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
                break
        
        if r >= best_rank:
            continue
        
        # Also try structured initializations
        for seed in seeds[:10]:
            if time.time() > deadline - 3:
                break
            
            result = try_structured_init(r, seed)
            if result is not None:
                U_f, V_f, W_f = result
                if r < best_rank:
                    best_rank = r
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result
# EVOLVE-BLOCK-END
