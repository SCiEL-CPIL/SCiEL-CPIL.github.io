import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    
    # We want to find U, V, W with integer/half-integer entries
    # Let's use a structured random search with local optimization
    # Values to consider: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    # But maybe we should focus on {-1, 0, 1} or {-1, 0, 1, 2} first
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    def check_decomposition(U, V, W):
        # T_rec = U @ V^T @ W^T is not quite right for 3-way tensor
        # T_rec[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_rec)
        return err < 1e-9

    def random_search(rank, time_limit):
        end_time = time.time() + time_limit
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Try a few random restarts
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > end_time:
                break
                
            # Initialize with sparse random matrices
            # Many entries should be 0
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Fill with random values from vals
            # Density: maybe 1/3 of entries non-zero?
            for r in range(rank):
                # Choose a few rows to have non-zero entries
                nz_U = random.randint(1, 3)
                nz_V = random.randint(1, 3)
                nz_W = random.randint(1, 3)
                
                rows_U = random.sample(range(n*m), nz_U)
                for row in rows_U:
                    U[row, r] = random.choice(vals)
                    
                rows_V = random.sample(range(m*p), nz_V)
                for row in rows_V:
                    V[row, r] = random.choice(vals)
                    
                rows_W = random.sample(range(p*n), nz_W)
                for row in rows_W:
                    W[row, r] = random.choice(vals)
            
            err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-9:
                return U, V, W
        
        # Local search from best solution found
        if best_U is not None:
            U, V, W = best_U, best_V, best_W
            # Perform local search: perturb one entry at a time
            # We need to be careful about time
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                    
                # Pick a random matrix and entry
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    r, c = random.randint(0, n*m-1), random.randint(0, rank-1)
                    old_val = U[r, c]
                    new_val = random.choice(vals)
                    if old_val != new_val:
                        U[r, c] = new_val
                        err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
                        if err < 1e-9:
                            return U, V, W
                        # Accept if it improves, or with some probability (simulated annealing-like)
                        # For now, just accept if it improves
                        if err < best_err:
                            best_err = err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        else:
                            U[r, c] = old_val
                elif mat_idx == 1:
                    r, c = random.randint(0, m*p-1), random.randint(0, rank-1)
                    old_val = V[r, c]
                    new_val = random.choice(vals)
                    if old_val != new_val:
                        V[r, c] = new_val
                        err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
                        if err < 1e-9:
                            return U, V, W
                        if err < best_err:
                            best_err = err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        else:
                            V[r, c] = old_val
                else:
                    r, c = random.randint(0, p*n-1), random.randint(0, rank-1)
                    old_val = W[r, c]
                    new_val = random.choice(vals)
                    if old_val != new_val:
                        W[r, c] = new_val
                        err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
                        if err < 1e-9:
                            return U, V, W
                        if err < best_err:
                            best_err = err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        else:
                            W[r, c] = old_val
        
        return None

    # Search for lower ranks
    # Start from rank 20 down to 15
    # Give more time to lower ranks? Or less?
    # Let's give each rank about 15 seconds
    for rank in range(20, 14, -1):
        if time.time() > start_time + 110: # Safety margin
            break
        result = random_search(rank, 15.0)
        if result:
            U, V, W = result
            # Convert to lists of floats
            # Ensure they are exact half-integers
            # numpy floats should be fine for the evaluator
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
