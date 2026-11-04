import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin

    def validate(U, V, W, n, m, p, rank):
        """
        Validate that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all i in [0,n), j in [0,m), k in [0,p).
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        s += U[ui, r] * V[vj, r] * W[wk, r]
                    if s != 1:
                        return False
        return True

    # Schoolbook decomposition
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try to find a lower rank decomposition using random search
    # We try ranks from 20 up to base_rank - 1
    for target_rank in range(20, base_rank):
        if time.time() > deadline - 5.0:
            break
            
        # If we find a valid decomposition, we can stop early if it's very low
        # But we want the minimum, so we should try lower ranks first.
        # Since we are iterating up, we should break on first success?
        # No, we want the MINIMUM rank. So we should try 20, then 21, etc.
        # If we find one at 20, that's likely optimal or near optimal.
        
        found = False
        # Limit attempts per rank to avoid spending too much time on one rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
                
            np.random.seed(attempt * 1000 + target_rank)
            
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            W = np.zeros((p * n, target_rank), dtype=np.int64)
            
            # Initialize with sparse random integers
            for r in range(target_rank):
                # Each term has a few non-zeros
                nz_u = np.random.choice(n * m, size=np.random.randint(2, 6), replace=False)
                nz_v = np.random.choice(m * p, size=np.random.randint(2, 6), replace=False)
                nz_w = np.random.choice(p * n, size=np.random.randint(2, 6), replace=False)
                
                U[nz_u, r] = np.random.choice([-2, -1, 1, 2], size=len(nz_u))
                V[nz_v, r] = np.random.choice([-2, -1, 1, 2], size=len(nz_v))
                W[nz_w, r] = np.random.choice([-2, -1, 1, 2], size=len(nz_w))
            
            if validate(U, V, W, n, m, p, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            # If we found a low rank, we can stop searching higher ranks
            # as we want the minimum.
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
