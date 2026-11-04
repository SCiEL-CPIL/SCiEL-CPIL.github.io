import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Strassen 2x2 decomposition
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    # Cols: M1..M7
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Convert to numpy for efficient Kronecker product
    U2_np = np.array(U2, dtype=float)
    V2_np = np.array(V2, dtype=float)
    W2_np = np.array(W2, dtype=float)
    
    # Kronecker product
    U4_np = np.kron(U2_np, U2_np)
    V4_np = np.kron(V2_np, V2_np)
    W4_np = np.kron(W2_np, W2_np)
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook if somehow too slow (unlikely)
        n, m, p = DIMENSIONS
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

    # Convert back to lists of integers
    U4 = [[int(x) for x in row] for row in U4_np]
    V4 = [[int(x) for x in row] for row in V4_np]
    W4 = [[int(x) for x in row] for row in W4_np]
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
