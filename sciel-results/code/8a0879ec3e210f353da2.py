import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Leave 5s safety margin before 150s kill
    
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
        A_mat = np.zeros((n * m * p, p * n * rank))
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    col_base = k * n * rank + i * rank + r
                    for j in range(m):
                        row = i * m * p + j * p + k
                        A_mat[row, col_base] = U[i * m + j, r] * V[j * p + k, r]
        return A_mat

    def try_rank(rank, max_trials=500):
        if time.time() > deadline - 20:
            return None
        
        target = np.ones((n, m, p))
        target_flat = target.flatten()
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                return None
            
            # Random initialization with structured patterns
            # Try different distributions
            if trial % 2 == 0:
                U = np.random.randint(-1, 2, (n * m, rank))
                V = np.random.randint(-1, 2, (m * p, rank))
            else:
                U = np.random.choice([-1, 1], (n * m, rank))
                V = np.random.choice([-1, 1], (m * p, rank))
            
            # Build A_mat
            A_mat = build_A_mat(U, V, rank)
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
            except:
                continue
            
            # Check if solution is close to integer
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target_flat)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
            
            # If close but not exact, try local search polishing
            if error < 0.5:
                # Local search: try flipping a few entries in U or V
                best_U = U.copy()
                best_V = V.copy()
                best_error = error
                
                for polish_iter in range(20):
                    if time.time() > deadline - 20:
                        return None
                    
                    improved = False
                    # Try flipping random entries in U
                    for _ in range(5):
                        idx = np.random.randint(0, n * m)
                        col = np.random.randint(0, rank)
                        old_val = best_U[idx, col]
                        # Try flipping sign or changing value
                        new_val = -old_val if old_val != 0 else (1 if np.random.random() < 0.5 else -1)
                        if new_val == old_val: continue
                        
                        best_U[idx, col] = new_val
                        A_mat_pert = build_A_mat(best_U, best_V, rank)
                        try:
                            W_flat_pert, _, _, _ = np.linalg.lstsq(A_mat_pert, target_flat, rcond=None)
                            W_int_pert = np.round(W_flat_pert)
                            error_pert = np.linalg.norm(A_mat_pert @ W_int_pert - target_flat)
                            if error_pert < best_error:
                                best_error = error_pert
                                if error_pert < 1e-9:
                                    W = W_int_pert.reshape(p * n, rank)
                                    return best_U, best_V, W
                                improved = True
                            else:
                                best_U[idx, col] = old_val # revert
                        except:
                            best_U[idx, col] = old_val

                    # Try flipping random entries in V
                    for _ in range(5):
                        idx = np.random.randint(0, m * p)
                        col = np.random.randint(0, rank)
                        old_val = best_V[idx, col]
                        new_val = -old_val if old_val != 0 else (1 if np.random.random() < 0.5 else -1)
                        if new_val == old_val: continue
                        
                        best_V[idx, col] = new_val
                        A_mat_pert = build_A_mat(best_U, best_V, rank)
                        try:
                            W_flat_pert, _, _, _ = np.linalg.lstsq(A_mat_pert, target_flat, rcond=None)
                            W_int_pert = np.round(W_flat_pert)
                            error_pert = np.linalg.norm(A_mat_pert @ W_int_pert - target_flat)
                            if error_pert < best_error:
                                best_error = error_pert
                                if error_pert < 1e-9:
                                    W = W_int_pert.reshape(p * n, rank)
                                    return best_U, best_V, W
                                improved = True
                            else:
                                best_V[idx, col] = old_val # revert
                        except:
                            best_V[idx, col] = old_val
                            
                    if not improved:
                        break

        return None

    # Try ranks from 47 down to 45
    # Ranks below 45 are highly unlikely for 3x4x4
    for rank in range(47, 44, -1):
        if time.time() > deadline - 25:
            break
        
        # Allocate more time for higher ranks (closer to schoolbook)
        trials = 300 if rank == 47 else 100
        result = try_rank(rank, max_trials=trials)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
