import time
import numpy as np
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
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
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check exactness
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
        # Round to nearest 0.5
        return int(x * 2 + 0.5) / 2.0

    # Try to find a better rank
    for target_rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        
        # Initialize
        # Use random initialization with small integers
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Convert to numpy for linear algebra
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Iterative refinement
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # 1. Solve for U
            # Equations: sum_r u_{r, im+j} v_{r, jp+k} w_{r, kn+i} = 1
            # This is linear in u.
            # We can form a matrix A_u where each row corresponds to an (i,j,k) equation.
            # A_u[row, col] = v_{r, jp+k} * w_{r, kn+i}
            # But u is indexed by (r, im+j).
            # So the unknowns are u_{r, im+j}.
            # Let's flatten U to length R * n * m.
            
            A_u = np.zeros((n * m * p, target_rank * n * m))
            b_u = np.ones(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            # u_{r, im+j} is at index r * n * m + i * m + j
                            A_u[idx, r * n * m + i * m + j] = V_np[j * p + k, r] * W_np[k * n + i, r]
                        idx += 1
            
            # Solve least squares
            try:
                U_sol, residuals, rank, s = np.linalg.lstsq(A_u, b_u, rcond=None)
            except:
                continue
                
            # Reshape and round
            U_new = U_sol.reshape(target_rank, n * m).T
            U_new = np.round(U_new * 2) / 2
            U = U_new.tolist()
            U_np = U_new
            
            # 2. Solve for V
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
            
            # 3. Solve for W
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
        
    # If no better solution found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
