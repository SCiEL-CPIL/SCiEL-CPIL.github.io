import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Deadline guard: although this specific construction is O(nmp) and very fast,
    # we include a check as per directive to ensure safety if the logic were expanded.
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin before 150s kill
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time only occasionally to avoid overhead, 
                # but for such small N, checking every step is fine too.
                if time.time() > deadline:
                    # Return partial result if we ever exceeded time (unlikely here)
                    # We must return valid matrices. If we stopped early, 
                    # the remaining columns are 0, which is a valid (but higher rank 
                    # effectively, or just incomplete) decomposition? 
                    # No, incomplete decomposition is invalid.
                    # Since the loop is tiny, this won't trigger.
                    pass
                
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
