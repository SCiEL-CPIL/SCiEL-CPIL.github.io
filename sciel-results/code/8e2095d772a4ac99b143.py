import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    rank_school = n * m * p
    
    # Schoolbook decomposition
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def verify(U, V, W):
        n, m, p = 3, 3, 6
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_school, V_school, W_school):
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Try to find a lower rank decomposition using random search
    # This is a heuristic and may not succeed, but it's safe to try
    start_time = time.time()
    deadline = start_time + 100
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 14 (theoretical min for 2x 3x3x3) up to 53
    # We focus on a few likely candidates
    candidate_ranks = [14, 15, 16, 18, 20, 24, 27, 30, 36, 40, 45, 50, 53]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
            
        # Try a few random initializations
        max_attempts = 5
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 7919 + rank * 31)
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each rank term picks random indices
            for r in range(rank):
                ui = np.random.randint(0, n*m)
                vj = np.random.randint(0, m*p)
                wk = np.random.randint(0, p*n)
                U[ui, r] = np.random.choice([-1.0, 1.0])
                V[vj, r] = np.random.choice([-1.0, 1.0])
                W[wk, r] = np.random.choice([-1.0, 1.0])
                
            if verify(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
                
        if found:
            # Found a lower rank, continue to try even lower
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
