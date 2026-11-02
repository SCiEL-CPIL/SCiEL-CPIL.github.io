import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build target tensor as sparse dict of (I,J,K) -> value
    # I = i*m + j, J = j*p + k, K = k*n + i
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                target[(I, J, K)] = 1.0
    
    start_time = time.time()
    deadline = 120.0  # 120s wall clock, use 118s to be safe
    
    # Try decreasing ranks from 41 down
    for target_rank in range(rank_schoolbook - 1, 0, -1):
        if time.time() - start_time > deadline - 2:
            break
        
        # Strategy: generate structured random matrices
        # Use small integer/half-integer entries
        # Try multiple random seeds and structures
        
        found = False
        max_trials = 50000
        trial = 0
        
        while trial < max_trials and time.time() - start_time < deadline - 2:
            trial += 1
            
            # Random generation with small entries
            # Use a mix: mostly ±1, sometimes 0, rarely ±2
            scale = 2 if target_rank > 30 else 1
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with random entries from {-1, 0, 1} mostly
            U[:] = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V[:] = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W[:] = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # For lower ranks, allow ±2 sometimes
            if target_rank <= 35 and np.random.random() < 0.3:
                idx_u = np.random.randint(0, n * m * target_rank)
                U.flat[idx_u] = np.random.choice([-2, 2])
                idx_v = np.random.randint(0, m * p * target_rank)
                V.flat[idx_v] = np.random.choice([-2, 2])
                idx_w = np.random.randint(0, p * n * target_rank)
                W.flat[idx_w] = np.random.choice([-2, 2])
            
            # Vectorized verification using einsum
            # Compute T[I,J,K] = sum_r U[I,r] * V[J,r] * W[K,r]
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            # Check all target entries
            valid = True
            for (I, J, K), val in target.items():
                if T[I, J, K] != val:
                    valid = False
                    break
            
            if valid:
                # Check all non-target entries are zero
                # T shape is (6, 21, 14)
                # Create a mask of target positions
                T_masked = T.copy()
                for (I, J, K), val in target.items():
                    T_masked[I, J, K] = 0.0
                if not np.allclose(T_masked, 0.0, atol=0.0):
                    valid = False
            
            if valid:
                found = True
                break
        
        if found:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 42
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
