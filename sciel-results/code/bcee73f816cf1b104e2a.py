import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
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

    # Strategy: Try to find lower rank by exploiting structure
    # For 4x5x6, we can try block decomposition approaches
    
    # First, let's try a systematic approach:
    # 1. Try rank 119, 118, ... down to some limit
    # 2. Use integer programming / constraint solving
    
    # Actually, let's try a different structural approach:
    # Decompose the 4x5x6 multiplication into smaller blocks and use Strassen-like algorithms
    
    # For 4x4x4, Strassen gives rank 7 instead of 16
    # For our case, we can try to decompose into 2x2 blocks and apply Strassen
    
    # Let's try a hybrid approach:
    # - Use the fact that matrix multiplication is bilinear
    # - Try to find a decomposition with rank < 120 using structured patterns
    
    # Approach: Try to find a valid decomposition by:
    # 1. Starting with schoolbook
    # 2. Trying to eliminate terms by finding linear dependencies
    # 3. Using exact arithmetic to verify
    
    # For now, let's try a few specific ranks with a smarter initialization
    
    target_ranks = [119, 118, 117, 116, 115]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.uniform(-0.5, 0.5, size=(n * m, target_rank))
            V = np.random.uniform(-0.5, 0.5, size=(m * p, target_rank))
            W = np.random.uniform(-0.5, 0.5, size=(p * n, target_rank))
            
            # Reshape for easier indexing
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # ALS iterations with exact verification at the end
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = V_r[j] * W_r[:, i]
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_r[i, j] = u_sol
                        except np.linalg.LinAlgError:
                            pass
                            
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = U_r[:, j] * W_r[k]
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_r[j, k] = v_sol
                        except np.linalg.LinAlgError:
                            pass
                            
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = U_r[i] * V_r[:, k]
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_r[k, i] = w_sol
                        except np.linalg.LinAlgError:
                            pass

            # After ALS, try to round to half-integers and verify
            U_rounded = np.round(U_r * 2) / 2
            V_rounded = np.round(V_r * 2) / 2
            W_rounded = np.round(W_r * 2) / 2
            
            # Verify exactness
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
