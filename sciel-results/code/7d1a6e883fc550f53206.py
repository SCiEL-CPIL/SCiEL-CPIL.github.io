import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def calc_S(U, V, W, rank):
        """Compute S[i,j,k] = sum_s U[s,i,j] * V[s,j,k] * W[s,i,k]"""
        return np.einsum('sij,sjk,sik->ijk', U, V, W)
    
    def verify_solution(U_list, V_list, W_list, rank, n, m, p):
        """Verify that the solution exactly decomposes the tensor"""
        U = np.array(U_list, dtype=float).reshape(n, m, rank)
        V = np.array(V_list, dtype=float).reshape(m, p, rank)
        W = np.array(W_list, dtype=float).reshape(p, n, rank)
        # Transpose to get (rank, n, m), (rank, m, p), (rank, n, p)
        U_t = U.transpose(2, 0, 1)
        V_t = V.transpose(2, 0, 1)
        W_t = W.transpose(2, 0, 1)
        S = calc_S(U_t, V_t, W_t, rank)
        return np.allclose(S, 1.0, atol=1e-10)
    
    def get_rank_30_solution():
        """Trivial rank-30 solution"""
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def try_reduce_rank(U_list, V_list, W_list, rank, target_rank, max_time=30):
        """Try to reduce rank from current to target_rank using local search"""
        start = time.time()
        deadline_local = start + max_time
        
        U = np.array(U_list, dtype=float).reshape(n, m, rank).transpose(2, 0, 1)
        V = np.array(V_list, dtype=float).reshape(m, p, rank).transpose(2, 0, 1)
        W = np.array(W_list, dtype=float).reshape(p, n, rank).transpose(2, 0, 1)
        
        S = calc_S(U, V, W, rank)
        error = np.sum((S - 1.0)**2)
        
        # If already at target rank, verify
        if rank == target_rank:
            if error < 1e-10:
                return True, U, V, W, rank
            return False, U, V, W, rank
        
        # For now, just verify if we're at the right rank
        # A more sophisticated approach would try to merge terms
        return error < 1e-10, U, V, W, rank
    
    # Strategy: Try to find solutions at various ranks
    # Start with rank 30 (known to work) and try to reduce
    
    # First, let's try a direct search for lower ranks
    # Use a smarter initialization
    
    best_result = None
    best_rank = 30
    
    # Try ranks from 20 to 30
    for target_rank in range(20, 31):
        if time.time() > deadline - 15:
            break
        
        # Try multiple restarts
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 7 + target_rank * 13)
            
            # Initialize with random half-integers
            vals = np.arange(-2, 2.5, 0.5)
            
            U = np.random.choice(vals, size=(target_rank, n, m))
            V = np.random.choice(vals, size=(target_rank, m, p))
            W = np.random.choice(vals, size=(target_rank, n, p))
            
            S = calc_S(U, V, W, target_rank)
            error = np.sum((S - 1.0)**2)
            
            # Local search with batch updates
            max_iters = 2000
            iter_count = 0
            stuck_count = 0
            best_error = error
            
            while error > 1e-9 and iter_count < max_iters and time.time() < deadline - 5:
                iter_count += 1
                
                # Choose which tensor to update
                tensor_id = np.random.choice(3)
                
                if tensor_id == 0:
                    # Update U
                    s, i, j = np.random.randint(target_rank), np.random.randint(n), np.random.randint(m)
                    old_val = U[s, i, j]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    current_S_row = S[i, j, :]
                    new_S_row = current_S_row + delta * (V[s, j, :] * W[s, i, :])
                    
                    old_err_part = np.sum((current_S_row - 1.0)**2)
                    new_err_part = np.sum((new_S_row - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        U[s, i, j] = new_val
                        S[i, j, :] = new_S_row
                        error += (new_err_part - old_err_part)
                        stuck_count = 0
                        if error < best_error:
                            best_error = error
                    else:
                        stuck_count += 1
                        
                elif tensor_id == 1:
                    # Update V
                    s, j, k = np.random.randint(target_rank), np.random.randint(m), np.random.randint(p)
                    old_val = V[s, j, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    current_S_col = S[:, j, k]
                    new_S_col = current_S_col + delta * (U[s, :, j] * W[s, :, k])
                    
                    old_err_part = np.sum((current_S_col - 1.0)**2)
                    new_err_part = np.sum((new_S_col - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        V[s, j, k] = new_val
                        S[:, j, k] = new_S_col
                        error += (new_err_part - old_err_part)
                        stuck_count = 0
                        if error < best_error:
                            best_error = error
                    else:
                        stuck_count += 1
                        
                else:
                    # Update W
                    s, i, k = np.random.randint(target_rank), np.random.randint(n), np.random.randint(p)
                    old_val = W[s, i, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5:
                        continue
                    
                    delta = new_val - old_val
                    current_S_vec = S[i, :, k]
                    new_S_vec = current_S_vec + delta * (U[s, i, :] * V[s, :, k])
                    
                    old_err_part = np.sum((current_S_vec - 1.0)**2)
                    new_err_part = np.sum((new_S_vec - 1.0)**2)
                    
                    if new_err_part < old_err_part - 1e-12:
                        W[s, i, k] = new_val
                        S[i, :, k] = new_S_vec
                        error += (new_err_part - old_err_part)
                        stuck_count = 0
                        if error < best_error:
                            best_error = error
                    else:
                        stuck_count += 1
                
                # If stuck too long, break early
                if stuck_count > 500:
                    break
            
            if error < 1e-9:
                # Found a solution!
                U_list = [[U[s, i, j] for s in range(target_rank)] for i in range(n) for j in range(m)]
                V_list = [[V[s, j, k] for s in range(target_rank)] for j in range(m) for k in range(p)]
                W_list = [[W[s, i, k] for s in range(target_rank)] for k in range(p) for i in range(n)]
                
                # Verify
                if verify_solution(U_list, V_list, W_list, target_rank, n, m, p):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to rank 30
    if best_result is not None:
        return best_result
    return get_rank_30_solution()
