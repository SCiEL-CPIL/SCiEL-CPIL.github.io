import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin before 150s kill
    
    def schoolbook():
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
    
    def build_A_mat(U, V, rank):
        """Build the linear system matrix A such that A @ W_flat = target_flat"""
        # A has shape (n*m*p, p*n*rank)
        # Row index: i*m*p + j*p + k
        # Col index: k*n*rank + i*rank + r
        A_mat = np.zeros((n * m * p, p * n * rank))
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    col_base = k * n * rank + i * rank + r
                    for j in range(m):
                        row = i * m * p + j * p + k
                        A_mat[row, col_base] = U[i * m + j, r] * V[j * p + k, r]
        return A_mat
    
    def check_solution(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(W.shape[1]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank_direct(rank, max_trials=500):
        """Direct random search with least squares"""
        if time.time() > deadline - 30:
            return None
        
        target = np.ones(n * m * p)
        
        for trial in range(max_trials):
            if time.time() > deadline - 30:
                return None
            
            # Random initialization with +/-1
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Build A_mat and solve
            A_mat = build_A_mat(U, V, rank)
            
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target, rcond=None)
            except:
                continue
            
            # Check if W_flat is close to integers or half-integers
            W_int = np.round(W_flat)
            error_int = np.linalg.norm(A_mat @ W_int - target)
            
            if error_int < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
            
            W_half = np.round(W_flat * 2) / 2
            error_half = np.linalg.norm(A_mat @ W_half - target)
            
            if error_half < 1e-9:
                W = W_half.reshape(p * n, rank)
                return U, V, W
        
        return None
    
    # Try ranks from lower to higher
    # Known lower bound for 3x4x4 is likely around 19 (from 4x4x4)
    # Schoolbook is 48
    for rank in range(19, 49):
        if time.time() > deadline - 35:
            break
        
        # More trials for lower ranks
        trials = 1000 if rank < 25 else 500
        result = try_rank_direct(rank, max_trials=trials)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
