"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook as fallback
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
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Precompute index arrays for fast evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def compute_total(U_arr, V_arr, W_arr, rank):
        """Compute the total tensor sum efficiently"""
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        return total
    
    def check_exact_integer(U, V, W, rank):
        """Check if integer decomposition is exact"""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        if np.max(np.abs(total - 1.0)) > 1e-9:
            return False
        return True
    
    def gradient_descent_step(U_arr, V_arr, W_arr, rank, lr=0.01):
        """One step of gradient descent on the smooth squared-error objective"""
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        error = total - 1.0
        
        # Compute gradients
        dU = np.zeros_like(U_arr)
        dV = np.zeros_like(V_arr)
        dW = np.zeros_like(W_arr)
        
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            
            # Gradient w.r.t. U
            dU[:, r] = lr * error * (v_col[v_idx] * w_col[w_idx])
            # Gradient w.r.t. V
            dV[:, r] = lr * error * (u_col[u_idx] * w_col[w_idx])
            # Gradient w.r.t. W
            dW[:, r] = lr * error * (u_col[u_idx] * v_col[v_idx])
        
        U_arr -= dU
        V_arr -= dV
        W_arr -= dW
        
        return U_arr, V_arr, W_arr
    
    def polish_solution(U_arr, V_arr, W_arr, rank, max_iters=100):
        """High-precision polish pass with decreasing learning rate"""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            lr = 0.01 * (0.95 ** it)
            U_arr, V_arr, W_arr = gradient_descent_step(U_arr, V_arr, W_arr, rank, lr)
            
            # Check convergence
            total = compute_total(U_arr, V_arr, W_arr, rank)
            if np.max(np.abs(total - 1.0)) < 1e-10:
                break
        
        # Round to nearest integer/half-integer
        U_int = np.round(U_arr * 2) / 2
        V_int = np.round(V_arr * 2) / 2
        W_int = np.round(W_arr * 2) / 2
        
        # Verify
        total = compute_total(U_int, V_int, W_int, rank)
        if np.max(np.abs(total - 1.0)) < 1e-9:
            U_out = [[float(x) for x in row] for row in U_int.tolist()]
            V_out = [[float(x) for x in row] for row in V_int.tolist()]
            W_out = [[float(x) for x in row] for row in W_int.tolist()]
            return U_out, V_out, W_out
        
        return None
    
    # Try ranks from 50 to 71
    for rank_target in range(50, min(rank_school, 72)):
        if time.time() > deadline - 20:
            break
            
        success = False
        
        # Multiple random restarts for each rank
        max_restarts = 8 if rank_target > 60 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Structured initialization: exploit the 8 independent 3x3x3 sub-problems
            # Each sub-problem (for fixed k) is a 3x3x3 tensor
            U_arr = np.zeros((n * m, rank_target), dtype=np.float64)
            V_arr = np.zeros((m * p, rank_target), dtype=np.float64)
            W_arr = np.zeros((p * n, rank_target), dtype=np.float64)
            
            # For each of the p=8 sub-problems, assign a subset of rank dimensions
            rank_per_sub = rank_target // p
            rank_remainder = rank_target % p
            
            for k in range(p):
                # Start index for this sub-problem's rank dimensions
                start_r = k * rank_per_sub + min(k, rank_remainder)
                end_r = start_r + rank_per_sub + (1 if k < rank_remainder else 0)
                
                for r in range(start_r, end_r):
                    # Random ±1 pattern for U (rows corresponding to this k don't matter directly)
                    # But we need to ensure coverage
                    num_nonzero_u = random.randint(3, 9)
                    rows_u = random.sample(range(n * m), num_nonzero_u)
                    for row in rows_u:
                        U_arr[row, r] = random.choice([-1, 1])
                    
                    # V: rows for this k
                    v_rows_for_k = [j * p + k for j in range(m)]
                    num_nonzero_v = random.randint(2, 4)
                    if num_nonzero_v <= len(v_rows_for_k):
                        rows_v = random.sample(v_rows_for_k, num_nonzero_v)
                    else:
                        rows_v = v_rows_for_k
                    for row in rows_v:
                        V_arr[row, r] = random.choice([-1, 1])
                    
                    # W: rows for this k
                    w_rows_for_k = [k * n + i for i in range(n)]
                    num_nonzero_w = random.randint(2, 3)
                    if num_nonzero_w <= len(w_rows_for_k):
                        rows_w = random.sample(w_rows_for_k, num_nonzero_w)
                    else:
                        rows_w = w_rows_for_k
                    for row in rows_w:
                        W_arr[row, r] = random.choice([-1, 1])
            
            # Gradient descent local search
            best_err = np.inf
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                    
                # Compute current error
                total = compute_total(U_arr, V_arr, W_arr, rank_target)
                err = np.sum((total - 1.0)**2)
                
                if err < best_err:
                    best_err = err
                
                # Check for near-exact solution
                if err < 1e-8:
                    polished = polish_solution(U_arr.copy(), V_arr.copy(), W_arr.copy(), rank_target)
                    if polished is not None:
                        U_int, V_int, W_int = polished
                        if check_exact_integer(U_int, V_int, W_int, rank_target):
                            best_rank = rank_target
                            best_result = {"U": U_int, "V": V_int, "W": W_int}
                            success = True
                            break
                
                # Gradient step
                U_arr, V_arr, W_arr = gradient_descent_step(U_arr, V_arr, W_arr, rank_target, lr=0.005)
                
                # Occasionally add random perturbation to escape local minima
                if step % 50 == 0 and step > 0:
                    mat_idx = random.choice([0, 1, 2])
                    if mat_idx == 0:
                        row = random.randint(0, n * m - 1)
                        col = random.randint(0, rank_target - 1)
                        U_arr[row, col] += random.choice([-0.5, 0.5])
                    elif mat_idx == 1:
                        row = random.randint(0, m * p - 1)
                        col = random.randint(0, rank_target - 1)
                        V_arr[row, col] += random.choice([-0.5, 0.5])
                    else:
                        row = random.randint(0, p * n - 1)
                        col = random.randint(0, rank_target - 1)
                        W_arr[row, col] += random.choice([-0.5, 0.5])
            
            if success:
                break
        
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
