import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Target tensor from schoolbook: T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    
    # Try to find a decomposition with rank < 27
    # We will try ranks 26 down to 20
    
    def check_solution(U, V, W):
        # Compute the decomposition
        # U: (9, R), V: (9, R), W: (9, R)
        # Result[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can vectorize this
        # Res[i, j, k]
        R = U.shape[1]
        # Flatten indices
        # i: 0..2, j: 0..2, k: 0..2
        # U_idx = i*m + j
        # V_idx = j*p + k
        # W_idx = k*n + i
        
        # We can compute the tensor T_decomp
        # T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient way:
        # For each r, add U_r (reshaped to n x m) * V_r (reshaped to m x p) * W_r (reshaped to p x n) to T?
        # No, the indices are mixed.
        
        # Let's just loop for simplicity and speed check
        # 27 elements, R up to 27. 27*27 = 729 ops. Very fast.
        
        T_decomp = np.zeros((n, m, p), dtype=int)
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # Check if T_decomp is all 1s
        return np.all(T_decomp == 1)

    def random_init(rank):
        U = np.random.randint(-2, 3, size=(n*m, rank))
        V = np.random.randint(-2, 3, size=(m*p, rank))
        W = np.random.randint(-2, 3, size=(p*n, rank))
        return U, V, W

    # Try ranks
    for rank in range(26, 19, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            U, V, W = random_init(rank)
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try changing one element in U, V, or W
                # Randomly pick a matrix and an index
                for _ in range(100): # Limit inner steps
                    if time.time() > deadline:
                        break
                    
                    mat_idx = np.random.choice(3)
                    if mat_idx == 0:
                        mat = U
                        dims = (n*m, rank)
                    elif mat_idx == 1:
                        mat = V
                        dims = (m*p, rank)
                    else:
                        mat = W
                        dims = (p*n, rank)
                    
                    r_idx = np.random.randint(dims[0])
                    c_idx = np.random.randint(dims[1])
                    
                    old_val = mat[r_idx, c_idx]
                    # Try new values
                    for new_val in range(-3, 4):
                        if new_val == old_val:
                            continue
                        mat[r_idx, c_idx] = new_val
                        if check_solution(U, V, W):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        mat[r_idx, c_idx] = old_val
                    # If no improvement found in this step, continue loop
                    # Note: This is a "first fit" or "best fit" variant?
                    # It's a random walk with acceptance if valid.
                    # It doesn't track error, so it's a random search for exact match.
                    # This is very slow.
            
            # Better: Use a heuristic search based on error
            # But check_solution is exact.
            # Let's stick to the simple random search but optimize it.
            
    # If no better solution found, return schoolbook
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
