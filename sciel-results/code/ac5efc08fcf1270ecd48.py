"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_numpy(U, V, W, n, m, p):
    """Verify exact decomposition using numpy for speed."""
    rank = U.shape[1]
    # Build the tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution as baseline
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try to find lower-rank decompositions
    # Strategy: For each target rank, try structured initializations
    # and verify. Use numpy for fast verification.
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(200):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.RandomState(trial * 7919 + target_rank * 137)
            
            # Initialize with sparse random values from {0, ±0.5, ±1, ±1.5, ±2}
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            vals = np.array([0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])
            
            for idx in range(n*m):
                mask = rng.random(target_rank) < 0.4
                U[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(m*p):
                mask = rng.random(target_rank) < 0.4
                V[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(p*n):
                mask = rng.random(target_rank) < 0.4
                W[idx, mask] = rng.choice(vals, size=np.sum(mask))
            
            if _verify_numpy(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
