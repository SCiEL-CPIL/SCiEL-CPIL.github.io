import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    best_rank = rank_seed
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe
    
    # Target ranks to try
    target_ranks = list(range(rank_seed - 1, 40, -1))
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random integer U, V with small entries
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            
            # Solve for W
            A = np.zeros((n * m * p, p * n * rank))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A[idx, c * (p * n) + k * n + i] = U[i * m + j, c] * V[j * p + k, c]
                        idx += 1
            
            # Solve A W_flat = b
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(rank, p * n).T
                # Check if the solution is exact
                if np.allclose(A @ W_flat, b, atol=1e-9):
                    # Check if W has integer or half-integer entries
                    W_rounded = np.round(W * 2) / 2.0
                    if np.allclose(A @ (W_rounded.T.reshape(-1)), b, atol=1e-9):
                        # Found a valid decomposition
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W_rounded
                        break
            except np.linalg.LinAlgError:
                pass
    
    if best_U is None:
        # Fallback to schoolbook
        rank = rank_seed
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
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
