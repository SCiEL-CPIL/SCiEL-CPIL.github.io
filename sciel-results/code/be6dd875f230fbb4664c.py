import time
import numpy as np
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds limit to be safe
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def check_solution(U, V, W):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True

    def round_to_half_int(x):
        return int(x * 2 + 0.5) / 2.0

    # Target ranks to try, from low to high
    # Known rank for 2x3x4 is 20. Try 19, 20, 21.
    target_ranks = [19, 20, 21]
    
    best_solution = None
    best_rank = rank_schoolbook
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Multiple restarts for each rank
        num_restarts = 5 if target_rank >= 20 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random integers -1, 0, 1
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Iterative refinement
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U
                A_u = np.zeros((n * m * p, target_rank * n * m))
                b_u = np.ones(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_u[idx, r * n * m + i * m + j] = V_np[j * p + k, r] * W_np[k * n + i, r]
                            idx += 1
                
                try:
                    U_sol, residuals, rank, s = np.linalg.lstsq(A_u, b_u, rcond=None)
                except:
                    continue
                    
                U_new = U_sol.reshape(target_rank, n * m).T
                U_new = np.round(U_new * 2) / 2
                U = U_new.tolist()
                U_np = U_new
                
                # Solve for V
                A_v = np.zeros((n * m * p, target_rank * m * p))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_v[idx, r * m * p + j * p + k] = U_np[i * m + j, r] * W_np[k * n + i, r]
                            idx += 1
                
                try:
                    V_sol, residuals, rank, s = np.linalg.lstsq(A_v, b_u, rcond=None)
                except:
                    continue
                    
                V_new = V_sol.reshape(target_rank, m * p).T
                V_new = np.round(V_new * 2) / 2
                V = V_new.tolist()
                V_np = V_new
                
                # Solve for W
                A_w = np.zeros((n * m * p, target_rank * p * n))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_w[idx, r * p * n + k * n + i] = U_np[i * m + j, r] * V_np[j * p + k, r]
                            idx += 1
                
                try:
                    W_sol, residuals, rank, s = np.linalg.lstsq(A_w, b_u, rcond=None)
                except:
                    continue
                    
                W_new = W_sol.reshape(target_rank, p * n).T
                W_new = np.round(W_new * 2) / 2
                W = W_new.tolist()
                W_np = W_new
                
                # Check if exact
                if check_solution(U, V, W):
                    return {"U": U, "V": V, "W": W}
                
                # Optional: Local search polish
                # If not exact, try small perturbations
                # This is expensive, so only do it if we are close
                # Calculate error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(target_rank):
                                val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                            error += (val - 1) ** 2
                
                if error < 1e-6:
                    # Try to fix by adjusting one entry
                    # Skip for now to save time, rely on next iteration
                    pass

    # If no better solution found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
