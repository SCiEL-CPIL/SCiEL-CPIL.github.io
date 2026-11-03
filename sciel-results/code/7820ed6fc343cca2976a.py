"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s kill
    
    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2x2 matrix multiplication.
        Returns U, V, W matrices of shape (4,7) each.
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = A11(B00-B10)
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W, 7
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs T."""
        recon = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # For each i,k: sum_j u[i,j]*v[j,k]*w[k,i]
            # This is equivalent to: (u * w.T) @ v ... no
            # Let's do it directly but vectorized
            for i in range(n):
                for k in range(p):
                    recon[i, :, k] = u_col[i, :] * v_col[:, k] * w_col[k, i]
        return np.allclose(recon, T, atol=1e-8)
    
    def peel_greedy_vectorized(T_res, n, m, p, max_iters=30, seed=0):
        """
        Vectorized greedy rank-1 peeling.
        """
        rng = np.random.RandomState(seed)
        residual = T_res.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for it in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            if np.allclose(residual, 0, atol=1e-10):
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            n_trials = 50
            for t in range(n_trials):
                u = rng.randn(n * m)
                v = rng.randn(m * p)
                
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                
                # Vectorized computation of optimal w
                # For each (i,k): w[k,i] = sum_j residual[i,j,k] * u[i,j] * v[j,k] / sum_j (u[i,j]*v[j,k])^2
                num = np.zeros((n, p))
                den = np.zeros((n, p))
                
                for i in range(n):
                    for k in range(p):
                        ujvk = u_mat[i, :] * v_mat[:, k]
                        num[i, k] = np.dot(residual[i, :, k], ujvk)
                        den[i, k] = np.dot(ujvk, ujvk)
                
                mask = den > 1e-12
                w_flat = np.zeros(p * n)
                for i in range(n):
                    for k in range(p):
                        if mask[i, k]:
                            w_flat[k * n + i] = num[i, k] / den[i, k]
                
                # Compute reconstruction
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for k in range(p):
                        recon[i, :, k] = u_mat[i, :] * v_mat[:, k] * w_flat[k * n + i]
                
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
            
            if best_u is None:
                break
            
            recon = np.zeros((n, m, p))
            for i in range(n):
                for k in range(p):
                    recon[i, :, k] = best_u[i * m + 0:(i+1)*m].reshape(m) * 0  # placeholder
                    # Actually let me redo this properly
            # Redo reconstruction properly
            best_u_mat = best_u.reshape(n, m)
            best_v_mat = best_v.reshape(m, p)
            recon = np.zeros((n, m, p))
            for i in range(n):
                for k in range(p):
                    recon[i, :, k] = best_u_mat[i, :] * best_v_mat[:, k] * best_w[k * n + i]
            
            new_residual = residual - recon
            if np.sum(new_residual * new_residual) >= np.sum(residual * residual) - 1e-12:
                break
            
            U_cols.append(best_u)
            V_cols.append(best_v)
            W_cols.append(best_w)
            residual = new_residual
        
        if len(U_cols) == 0:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        if verify(U, V, W, rank):
            return U, V, W, rank
        else:
            return None
    
    # Try peeling with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 20.0:
            break
        
        result = peel_greedy_vectorized(T, n, m, p, max_iters=15, seed=seed)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W)
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
