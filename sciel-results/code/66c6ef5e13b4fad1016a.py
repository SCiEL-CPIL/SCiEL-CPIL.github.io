import time
import random

def _schoolbook():
    n, m, p = 3, 5, 6
    rank = n * m * p
    # Initialize U, V, W as lists of lists of zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1.0
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1.0
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _to_list(x):
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    if isinstance(x, (int, float)):
        # Ensure integer values are represented as int for cleaner output
        if isinstance(x, float) and x == int(x):
            return int(x)
        return x
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    # Return the schoolbook decomposition which is rank 90.
    # This is the trivial, guaranteed valid solution.
    result = _schoolbook()
    
    # If we had time and a better search algorithm, we would try to improve it here.
    # Currently, we stick to the known valid rank 90.
    return _to_list(result)
