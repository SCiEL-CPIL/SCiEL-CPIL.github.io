import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def calculate_error_fast(U, V, W):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((R - T) ** 2)

    def local_search(U, V, W, max_steps=50000):
        current_error = calculate_error_fast(U, V, W)
        if current_error == 0:
            return U, V, W, True
            
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        rank = U.shape[1]
        
        for step in range(max_steps):
            if time.time() > deadline:
                return None, None, None, False
                
            # Pick a random matrix and coordinate
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
                
            row = np.random.randint(mat.shape[0])
            col = np.random.randint(mat.shape[1])
            
            original_val = mat[row, col]
            
            # Try +1
            mat[row, col] = original_val + 1
            new_error = calculate_error_fast(U, V, W)
            if new_error < current_error:
                current_error = new_error
                if current_error == 0:
                    return U, V, W, True
            else:
                mat[row, col] = original_val
                
            # Try -1
            mat[row, col] = original_val - 1
            new_error = calculate_error_fast(U, V, W)
            if new_error < current_error:
                current_error = new_error
                if current_error == 0:
                    return U, V, W, True
            else:
                mat[row, col] = original_val
                
        return U, V, W, False

    # Try rank 23 with structured initialization
    rank = 23
    
    # Pattern 1: Random
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        np.random.seed(42 + attempt)
        U = np.random.randint(-2, 3, (size_u, rank))
        V = np.random.randint(-2, 3, (size_v, rank))
        W = np.random.randint(-2, 3, (size_w, rank))
        U, V, W, found = local_search(U, V, W, max_steps=8000)
        if found and calculate_error_fast(U, V, W) == 0:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Pattern 2: Sparse initialization
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        np.random.seed(100 + attempt)
        U = np.zeros((size_u, rank), dtype=int)
        V = np.zeros((size_v, rank), dtype=int)
        W = np.zeros((size_w, rank), dtype=int)
        for r in range(rank):
            # Each column has 1-2 non-zero entries
            u_rows = np.random.choice(size_u, size=np.random.randint(1, 3), replace=False)
            v_rows = np.random.choice(size_v, size=np.random.randint(1, 3), replace=False)
            w_rows = np.random.choice(size_w, size=np.random.randint(1, 3), replace=False)
            for row in u_rows:
                U[row, r] = np.random.choice([-1, 1])
            for row in v_rows:
                V[row, r] = np.random.choice([-1, 1])
            for row in w_rows:
                W[row, r] = np.random.choice([-1, 1])
        U, V, W, found = local_search(U, V, W, max_steps=8000)
        if found and calculate_error_fast(U, V, W) == 0:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
