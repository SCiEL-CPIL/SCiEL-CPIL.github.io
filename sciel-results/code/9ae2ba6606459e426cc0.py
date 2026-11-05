"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def compute_residual_fast(U, V, W):
        """Compute the residual tensor efficiently using vectorized operations."""
        # Residual[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    residual[i, j, k] = np.dot(u_row * V[j*p+k], W[k*n+i])
        return residual
    
    def compute_error(U, V, W):
        """Compute the error norm efficiently."""
        residual = compute_residual_fast(U, V, W)
        return np.sum((residual - 1.0) ** 2)
    
    def try_rank_structured(rank, max_iters=2000, seed=42):
        """Try to find a valid decomposition using structured initialization and alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        target = np.ones((n, m, p))
        
        # Structured initialization: start with schoolbook-like structure and perturb
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize first n*m*p columns with schoolbook pattern (if rank >= n*m*p)
        schoolbook_rank = n * m * p
        if rank >= schoolbook_rank:
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            # Perturb remaining columns
            for r in range(schoolbook_rank, rank):
                U[:, r] = np.random.randn(n*m) * 0.1
                V[:, r] = np.random.randn(m*p) * 0.1
                W[:, r] = np.random.randn(p*n) * 0.1
        else:
            # If rank < schoolbook, use random initialization with structure
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = compute_error(U, V, W)
        
        # Alternating minimization with vectorized updates
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve M @ u = b where M[k,r] = V[j*p+k,r]*W[k*n+i,r]
            for i in range(n):
                for j in range(m):
                    # Build M: shape (p, rank)
                    M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                    b = np.ones(p)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U[i*m+j] = u_sol
                    except:
                        pass
            
            # Update V: for each (j,k), solve M @ v = b where M[i,r] = U[i*m+j,r]*W[k*n+i,r]
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                    b = np.ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V[j*p+k] = v_sol
                    except:
                        pass
            
            # Update W: for each (k,i), solve M @ w = b where M[j,r] = U[i*m+j,r]*V[j*p+k,r]
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                    b = np.ones(m)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W[k*n+i] = w_sol
                    except:
                        pass
            
            # Compute error
            err = compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-12:
                    break
            elif err > best_err * 1.5:
                # If error increased significantly, revert to best
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        
        if best_err < 1e-8:
            # Try to round to rational numbers with small denominators
            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                U_round = np.round(best_U * denom) / denom
                V_round = np.round(best_V * denom) / denom
                W_round = np.round(best_W * denom) / denom
                
                residual = compute_residual_fast(U_round, V_round, W_round)
                if np.allclose(residual, target, atol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        return None
    
    def try_rank_random(rank, max_iters=1000, seed=42):
        """Try random initialization with alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        target = np.ones((n, m, p))
        
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = compute_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            for i in range(n):
                for j in range(m):
                    M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                    b = np.ones(p)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U[i*m+j] = u_sol
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                    b = np.ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V[j*p+k] = v_sol
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                    b = np.ones(m)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W[k*n+i] = w_sol
                    except:
                        pass
            
            err = compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-12:
                    break
        
        if best_err < 1e-8:
            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                U_round = np.round(best_U * denom) / denom
                V_round = np.round(best_V * denom) / denom
                W_round = np.round(best_W * denom) / denom
                
                residual = compute_residual_fast(U_round, V_round, W_round)
                if np.allclose(residual, target, atol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        return None
    
    # Try ranks in increasing order with both structured and random initialization
    for rank in range(18, 40):
        if time.time() > deadline - 20:
            break
        
        # Try structured initialization first
        for seed in [42, 123, 456]:
            if time.time() > deadline - 10:
                break
            result = try_rank_structured(rank, max_iters=1500, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try random initialization
        for seed in [789, 1024, 2048]:
            if time.time() > deadline - 10:
                break
            result = try_rank_random(rank, max_iters=800, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook if we couldn't find better
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
