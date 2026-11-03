"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Build the target tensor T
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Function to check validity and snap to half-integers
    def check_and_snap(U, V, W, r, tol=1e-7):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        # Check exactness
        if not np.allclose(T_recon, T_target, atol=tol):
            return None
            
        # Convert back to lists
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Alternating Least Squares for tensor decomposition
    def als_decompose(r, max_iter=200, tol=1e-10, seed=42):
        if time.time() > deadline:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each row a, solve (V @ W.T) X = T[a,:,:].reshape(-1)
            VW = (V * W).sum(axis=0)  # shape (m*p, p*n)
            # Actually: sum_r V[b,r] * W[c,r] for each b,c
            VW_mat = V @ W.T  # shape (m*p, p*n)
            
            U_new = np.zeros_like(U)
            for a in range(n * m):
                # T[a, :, :] reshaped to (m*p, p*n)
                T_a = T_target[a, :, :].reshape(m * p, p * n)
                # Solve: U[a, :] @ VW_mat = T_a
                # i.e., VW_mat.T @ U[a, :].T = T_a.T
                try:
                    U_new[a, :] = np.linalg.lstsq(VW_mat.T, T_a.T, rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Update V
            UW_mat = U @ W.T  # shape (n*m, p*n)
            V_new = np.zeros_like(V)
            for b in range(m * p):
                T_b = T_target[:, b, :].reshape(n * m, p * n)
                try:
                    V_new[b, :] = np.linalg.lstsq(UW_mat.T, T_b.T, rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Update W
            UV_mat = U @ V.T  # shape (n*m, m*p)
            W_new = np.zeros_like(W)
            for c in range(p * n):
                T_c = T_target[:, :, c].reshape(n * m, m * p)
                try:
                    W_new[c, :] = np.linalg.lstsq(UV_mat.T, T_c.T, rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T_target) / np.linalg.norm(T_target)
            if err < tol:
                break
        
        # Final check
        result = check_and_snap(U.tolist(), V.tolist(), W.tolist(), r)
        if result:
            return result
        
        # Try polishing with more iterations if close
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T_target) / np.linalg.norm(T_target)
        if err < 1e-4:
            # Do more ALS iterations
            for it in range(500):
                if time.time() > deadline:
                    break
                
                VW_mat = V @ W.T
                U_new = np.zeros_like(U)
                for a in range(n * m):
                    T_a = T_target[a, :, :].reshape(m * p, p * n)
                    try:
                        U_new[a, :] = np.linalg.lstsq(VW_mat.T, T_a.T, rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                U = U_new
                
                UW_mat = U @ W.T
                V_new = np.zeros_like(V)
                for b in range(m * p):
                    T_b = T_target[:, b, :].reshape(n * m, p * n)
                    try:
                        V_new[b, :] = np.linalg.lstsq(UW_mat.T, T_b.T, rcond=None)[0]
                    except:
                        V_new[b, :] = V[b, :]
                V = V_new
                
                UV_mat = U @ V.T
                W_new = np.zeros_like(W)
                for c in range(p * n):
                    T_c = T_target[:, :, c].reshape(n * m, m * p)
                    try:
                        W_new[c, :] = np.linalg.lstsq(UV_mat.T, T_c.T, rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
                W = W_new
            
            result = check_and_snap(U.tolist(), V.tolist(), W.tolist(), r)
            if result:
                return result
        
        return None
    
    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try ranks from low to high
    # For M_{2,5,5}, known bounds suggest rank around 25-40
    ranks_to_try = [25, 27, 28, 30, 32, 34, 35, 36, 38, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        # Try multiple seeds
        seeds = [42, 123, 999, 777, 2024, 1111, 5555, 8888]
        found = False
        for seed in seeds:
            if time.time() > deadline:
                break
            result = als_decompose(r, max_iter=100, tol=1e-12, seed=seed)
            if result:
                best_result = result
                best_rank = r
                found = True
                break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we failed at this rank, higher ranks are less likely to be minimal
            # but we still try them as fallback
            pass
    
    return best_result
# EVOLVE-BLOCK-END
