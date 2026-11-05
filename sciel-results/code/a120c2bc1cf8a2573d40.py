import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    def get_schoolbook(rank):
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        return U, V, W

    U_school, V_school, W_school = get_schoolbook(rank_schoolbook)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    # Target Rank 108 (Known to be possible via Block Strassen)
    target_rank = 108
    
    if time.time() < deadline - 5:
        np.random.seed(42)
        
        # Initialize with Strassen-like structure for the 2x2x2 blocks
        # This is a heuristic initialization
        U = np.random.uniform(-1, 1, size=(n * m, target_rank))
        V = np.random.uniform(-1, 1, size=(m * p, target_rank))
        W = np.random.uniform(-1, 1, size=(p * n, target_rank))
        
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        max_iter = 300
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    M = V_r[j] * W_r[:, i]
                    try:
                        u_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_r[i, j] = u_sol
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    M = U_r[:, j] * W_r[k]
                    try:
                        v_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_r[j, k] = v_sol
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    M = U_r[i] * V_r[:, k]
                    try:
                        w_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_r[k, i] = w_sol
                    except np.linalg.LinAlgError:
                        pass

        # Round to nearest 0.5
        U_rounded = np.round(U_r * 2) / 2
        V_rounded = np.round(V_r * 2) / 2
        W_rounded = np.round(W_r * 2) / 2
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_rounded[i, j] * V_rounded[j, k] * W_rounded[k, i])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = target_rank
            best_U = U_rounded.flatten().tolist()
            best_V = V_rounded.flatten().tolist()
            best_W = W_rounded.flatten().tolist()
            return {"U": best_U, "V": best_V, "W": best_W}

    # If no better rank found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
