import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 160s limit
    
    n, m, p = 2, 2, 5
    
    # Build target tensor T[i,j,k] = 1 if i,j,k all in range else 0
    # U has n*m = 4 rows, V has m*p = 10 rows, W has p*n = 10 rows
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Known rank-18 solution
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            # Strassen products M1..M7
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}
    
    fallback = get_rank_18_solution()
    
    # Try to find rank 17 using ALS
    try:
        import numpy as np
        from scipy.optimize import minimize
    except ImportError:
        return fallback
    
    def build_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_target_tensor()
    
    def check_decomposition(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Compute sum_r U[i,r] V[j,r] W[k,r] for all i,j,k
        result = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            result += np.einsum('i,j,k->ijk', u_col, v_col, w_col)
        return np.allclose(result, T, atol=tol)
    
    def snap_to_half_integers(U, V, W):
        def snap(mat):
            mat = np.array(mat, dtype=float)
            snapped = np.round(mat * 2) / 2.0
            # Check if close
            if np.allclose(mat, snapped, atol=1e-6):
                return snapped
            return mat
        return snap(U), snap(V), snap(W)
    
    def try_als_rank(rank, max_iters=500, tol=1e-10, seed=None):
        if time.time() > deadline:
            return None
        
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 0:
                U[:, r] /= np.linalg.norm(U[:, r])
                V[:, r] /= np.linalg.norm(V[:, r])
                W[:, r] /= np.linalg.norm(W[:, r])
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for r in range(rank):
                # Minimize ||T - sum_r' U[:,r'] ⊗ V[:,r'] ⊗ W[:,r']||
                # For each r, we want to update U[:,r]
                residual = T.copy()
                for r2 in range(rank):
                    if r2 != r:
                        residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                
                # residual = sum_r' U[:,r'] ⊗ V[:,r'] ⊗ W[:,r'] - sum_{r'!=r} ...
                # We need: U[:,r] such that U[:,r] ⊗ V[:,r] ⊗ W[:,r'] approximates residual
                # This is: minimize ||residual - U_new ⊗ V[:,r] ⊗ W[:,r]||
                # = minimize ||residual.flatten() - np.kron(np.kron(U_new, V[:,r]), W[:,r])||
                # This is a linear system: A * U_new = b where A = kron(V[:,r], W[:,r])
                # Actually, using einsum: residual[i,j,k] = sum_r' U[i,r'] V[j,r'] W[k,r']
                # For fixed r, we want U[i,r] = argmin sum_i (residual[i,:,:] - U[i,r] V[:,r] ⊗ W[:,r])^2
                # = argmin sum_i [sum_jk (residual[i,j,k] - U[i,r] V[j,r] W[k,r])]^2 ... no
                # It's: sum_jk (residual[i,j,k] - U[i,r] V[j,r] W[k,r])^2
                # = sum_jk residual[i,j,k]^2 - 2 U[i,r] sum_jk residual[i,j,k] V[j,r] W[k,r] + U[i,r]^2 sum_jk V[j,r]^2 W[k,r]^2
                # d/dU[i,r]: -2 sum_jk residual[i,j,k] V[j,r] W[k,r] + 2 U[i,r] sum_jk V[j,r]^2 W[k,r]^2 = 0
                # U[i,r] = sum_jk residual[i,j,k] V[j,r] W[k,r] / sum_jk V[j,r]^2 W[k,r]^2
                
                denom = np.sum(V[:, r]**2) * np.sum(W[:, r]**2)
                if abs(denom) < 1e-15:
                    continue
                # numerator[i] = sum_jk residual[i,j,k] V[j,r] W[k,r]
                # = sum_j V[j,r] * (residual[i,j,:] dot W[:,r])
                temp = np.einsum('ij k, j, k -> i', residual, V[:, r], W[:, r])
                U[:, r] = temp / denom
            
            # Update V
            for r in range(rank):
                residual = T.copy()
                for r2 in range(rank):
                    if r2 != r:
                        residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                
                denom = np.sum(U[:, r]**2) * np.sum(W[:, r]**2)
                if abs(denom) < 1e-15:
                    continue
                temp = np.einsum('i j k, i, k -> j', residual, U[:, r], W[:, r])
                V[:, r] = temp / denom
            
            # Update W
            for r in range(rank):
                residual = T.copy()
                for r2 in range(rank):
                    if r2 != r:
                        residual -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                
                denom = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
                if abs(denom) < 1e-15:
                    continue
                temp = np.einsum('i j k, i, j -> k', residual, U[:, r], V[:, r])
                W[:, r] = temp / denom
            
            # Check convergence
            error = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        val = 0
                        for r in range(rank):
                            val += U[i,r] * V[j,r] * W[k,r]
                        error += (val - T[i,j,k])**2
            error = np.sqrt(error)
            if error < tol:
                # Try to snap to half-integers
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                if check_decomposition(U_s, V_s, W_s, tol=1e-8):
                    # Convert to lists
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                # If snap fails, check if original is exact enough
                if check_decomposition(U, V, W, tol=1e-10):
                    # Try snapping with more tolerance
                    U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        return None
    
    # Try rank 17
    best_result = fallback
    
    for seed in range(50):
        if time.time() > deadline - 20:
            break
        
        result = try_als_rank(17, max_iters=300, seed=seed)
        if result is not None:
            best_result = result
            break
    
    if time.time() > deadline - 10:
        return best_result
    
    # If rank 17 failed, we still have rank 18
    return best_result
