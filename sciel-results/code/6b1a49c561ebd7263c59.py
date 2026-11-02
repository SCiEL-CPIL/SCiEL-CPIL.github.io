import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    n, m, p = 2, 3, 7
    
    # Try ranks from 41 down to some minimum
    best_rank = n * m * p  # 42
    best_U = None
    best_V = None
    best_W = None
    
    for rank in range(41, 20, -1):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        for trial in range(100):
            if time.time() > deadline - 30:
                    break
            
            np.random.seed(trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            
            # Check if we can solve for W
            W = np.zeros((p * n, rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # For this (i,k), we have m=3 equations in rank unknowns
                    # M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # Solve M @ w = [1, 1, 1]^T
                    try:
                        # Check consistency
                        rank_M = np.linalg.matrix_rank(M)
                        if rank_M == 0:
                            valid = False
                            break
                        # Use least squares and check residual
                        w, residuals, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        if np.linalg.norm(M @ w - np.ones(m)) > 1e-8:
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Found a solution! Convert to half-integers if needed
                best_rank = rank
                # Round to nearest half-integer
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                # Verify the rounded solution
                valid_rounded = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(rank):
                                val += U_half[i*m+j, c] * V_half[j*p+k, c] * W_half[k*n+i, c]
                            if abs(val - 1) > 1e-6:
                                valid_rounded = False
                                break
                        if not valid_rounded:
                            break
                    if not valid_rounded:
                        break
                
                if valid_rounded:
                    # Convert to lists
                    U_list = U_half.tolist()
                    V_list = V_half.tolist()
                    W_list = W_half.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If rounding doesn't work, still record but continue
                best_U = U
                best_V = V
                best_W = W
    
    # Fallback: return schoolbook
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
