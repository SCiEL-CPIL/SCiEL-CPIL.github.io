import numpy as np
import time
from itertools import product

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
        A_mat = np.zeros((n * m * p, p * n * rank))
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    col_base = k * n * rank + i * rank + r
                    for j in range(m):
                        row = i * m * p + j * p + k
                        A_mat[row, col_base] = U[i * m + j, r] * V[j * p + k, r]
        return A_mat
    
    def build_A_mat_for_W(U, V, rank):
        """Alternative: build A for solving W given U,V"""
        # For each (i,j,k) entry of output: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # So A[row, col] where row=(i,j,k), col=(k,i,r) -> U[i*m+j,r]*V[j*p+k,r]
        A_mat = np.zeros((n * m * p, p * n * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for r in range(rank):
                        col = k * n + i
                        A_mat[row, col * rank + r] = U[i * m + j, r] * V[j * p + k, r]
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
    
    def try_rank_direct(rank, max_trials=200):
        """Direct random search with least squares"""
        if time.time() > deadline - 30:
            return None
        
        target = np.ones(n * m * p)
        
        for trial in range(max_trials):
            if time.time() > deadline - 30:
                return None
            
            # Various initialization strategies
            strategy = trial % 4
            
            if strategy == 0:
                # Pure random ±1
                U = np.random.randint(-1, 2, (n * m, rank))
                V = np.random.randint(-1, 2, (m * p, rank))
            elif strategy == 1:
                # Sparse: mostly 0 with some ±1
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                for _ in range(int(n * m * rank * 0.3)):
                    U[np.random.randint(n * m), np.random.randint(rank)] = np.random.choice([-1, 1])
                for _ in range(int(m * p * rank * 0.3)):
                    V[np.random.randint(m * p), np.random.randint(rank)] = np.random.choice([-1, 1])
            elif strategy == 2:
                # Block structure
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                block_size = max(1, rank // 4)
                for b in range(0, rank, block_size):
                    be = min(b + block_size, rank)
                    rows_u = np.random.choice(n * m, min(block_size, n * m), replace=False)
                    rows_v = np.random.choice(m * p, min(block_size, m * p), replace=False)
                    U[rows_u, b:be] = np.random.randint(-1, 2, (len(rows_u), be - b))
                    V[rows_v, b:be] = np.random.randint(-1, 2, (len(rows_v), be - b))
            else:
                # Mixed: random with some structure
                U = np.random.randint(-1, 2, (n * m, rank))
                V = np.random.randint(-1, 2, (m * p, rank))
                # Zero out some entries
                for _ in range(int(rank * 0.2)):
                    col = np.random.randint(rank)
                    if np.random.random() < 0.5:
                        U[np.random.choice(n * m, int(n * m * 0.5), replace=False), col] = 0
                    else:
                        V[np.random.choice(m * p, int(m * p * 0.5), replace=False), col] = 0
            
            # Build A_mat and solve
            A_mat = build_A_mat(U, V, rank)
            
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target, rcond=None)
            except:
                continue
            
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
            
            # If very close, try rounding with different offsets
            if error < 0.5:
                for offset in [0.1, -0.1, 0.2, -0.2]:
                    W_alt = np.round(W_flat + offset)
                    err_alt = np.linalg.norm(A_mat @ W_alt - target)
                    if err_alt < 1e-9:
                        W = W_alt.reshape(p * n, rank)
                        return U, V, W
        
        return None
    
    def try_rank_fix_U(rank, max_trials=100):
        """Fix U, solve for V and W jointly"""
        if time.time() > deadline - 30:
            return None
        
        target = np.ones(n * m * p)
        
        for trial in range(max_trials):
            if time.time() > deadline - 30:
                return None
            
            U = np.random.randint(-1, 2, (n * m, rank))
            
            # For fixed U, we need to find V and W such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
            # This is still bilinear in V and W, so we can't solve directly
            # But we can try: fix V too, solve W, check
            V = np.random.randint(-1, 2, (m * p, rank))
            
            A_mat = build_A_mat(U, V, rank)
            
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target, rcond=None)
            except:
                continue
            
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
        
        return None
    
    def try_rank_with_svd_polish(rank, max_trials=50):
        """Use SVD to polish near-integer solutions"""
        if time.time() > deadline - 30:
            return None
        
        target = np.ones(n * m * p)
        
        for trial in range(max_trials):
            if time.time() > deadline - 30:
                return None
            
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            A_mat = build_A_mat(U, V, rank)
            
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target, rcond=None)
            except:
                continue
            
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
            
            # SVD-based polish: try to find integer W close to W_flat
            # by projecting onto integer lattice
            if error < 2.0:
                # Try local search around W_int
                best_W = W_int.copy()
                best_err = error
                
                for _ in range(20):
                    if time.time() > deadline - 30:
                        return None
                    
                    # Perturb a random entry
                    idx = np.random.randint(W_flat.size)
                    old_val = best_W[idx]
                    best_W[idx] = old_val + np.random.choice([-1, 1])
                    
                    new_err = np.linalg.norm(A_mat @ best_W - target)
                    
                    if new_err < best_err:
                        best_err = new_err
                    else:
                        best_W[idx] = old_val
                
                if best_err < 1e-9:
                    W = best_W.reshape(p * n, rank)
                    return U, V, W
        
        return None
    
    # Try ranks from lower to higher
    # For 3x4x4, known lower bounds suggest rank >= 30
    # Schoolbook is 48, so we search 30-47
    for rank in range(30, 48):
        if time.time() > deadline - 35:
            break
        
        # First pass: direct search
        trials = 100 if rank < 38 else 60
        result = try_rank_direct(rank, max_trials=trials)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Second pass: with SVD polish
        if time.time() > deadline - 35:
            break
        trials_polish = 40 if rank < 38 else 25
        result = try_rank_with_svd_polish(rank, max_trials=trials_polish)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
