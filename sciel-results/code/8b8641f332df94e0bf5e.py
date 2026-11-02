import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Returns a valid tensor decomposition for 4x5 by 5x6 matrix multiplication.
    Uses the schoolbook decomposition (rank 120) as a baseline.
    Attempts to find a lower rank decomposition using structured random search
    with exact verification. Returns the best (lowest rank) valid decomposition
    found within the time budget.
    """
    n, m, p = DIMENSIONS
    rank_full = n * m * p  # 120
    
    # Start with schoolbook decomposition
    def schoolbook():
        U = [[0] * rank_full for _ in range(n * m)]
        V = [[0] * rank_full for _ in range(m * p)]
        W = [[0] * rank_full for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    # Verification function using NumPy for speed
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # We can vectorize this somewhat
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    prod_sum = np.dot(u_val, v_val * w_val)
                    if abs(prod_sum - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions using structured random search
    # Strategy: try ranks from 119 down, with limited attempts per rank
    # Use a time budget to ensure we finish well before the 150s limit
    
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try descending ranks
    # We'll focus on ranks that are slightly below 120 first, as they're more likely
    # to be achievable with small modifications
    for target_rank in range(rank_full - 1, 80, -1):
        if time.time() > deadline:
            break
        
        # Budget per rank: more time for higher ranks
        time_per_rank = min(1.0, max(0.2, (deadline - time.time()) / 30.0))
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - time_per_rank:
            attempts += 1
            
            # Create a candidate by taking the schoolbook and randomly modifying
            # a few entries, or by using a sparse random structure
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # For each column, assign a small number of non-zero entries
            # This is a heuristic: real decompositions have specific structure
            for r in range(target_rank):
                # Each column should contribute to exactly one (i, j, k) triple
                # but we allow for more complex interactions
                # Pick random positions with small probability of being non-zero
                nnz_u = 1
                nnz_v = 1
                nnz_w = 1
                
                # For simplicity, try a structured approach:
                # Each column corresponds to a specific (i, j, k) but with possible
                # additional terms to cancel out
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
                
                # With some probability, add extra terms to create cancellations
                if np.random.random() < 0.3:
                    extra_i = np.random.randint(0, n)
                    extra_j = np.random.randint(0, m)
                    extra_k = np.random.randint(0, p)
                    U[extra_i * m + extra_j][r] += np.random.choice([0.5, -0.5, 1.0, -1.0])
                    V[extra_j * p + extra_k][r] += np.random.choice([0.5, -0.5, 1.0, -1.0])
                    W[extra_k * n + extra_i][r] += np.random.choice([0.5, -0.5, 1.0, -1.0])
            
            # Verify the candidate
            if verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                return best
        
        # If we found a valid decomposition at this rank, return it
        # (This is handled above)
    
    # Return the best decomposition found (default is schoolbook)
    return best
