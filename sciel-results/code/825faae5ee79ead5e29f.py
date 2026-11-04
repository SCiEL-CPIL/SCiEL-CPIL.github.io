import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = 2, 3, 6
    rank = 33

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    # Define Strassen factors for 2x2x2
    # U factors (A side)
    S_U = [
        [1, 0], [0, 1],
        [0, 0], [1, 1],
        [1, 0], [0, 0],
        [0, 0], [0, 1],
        [1, 1], [0, 0],
        [-1, 0], [1, 0],
        [0, 1], [0, -1]
    ]
    S_U = [np.array(S_U[i:i+4]).reshape(2,2) for i in range(0, 28, 4)]

    # V factors (B side)
    S_V = [
        [1, 0], [0, 1],
        [1, 0], [0, 0],
        [0, 1], [0, -1],
        [-1, 0], [1, 0],
        [0, 0], [0, 1],
        [1, 1], [0, 0],
        [0, 0], [1, 1]
    ]
    S_V = [np.array(S_V[i:i+4]).reshape(2,2) for i in range(0, 28, 4)]

    # W factors (C side)
    S_W = [
        [1, 0], [0, 1],
        [0, 1], [0, -1],
        [0, 0], [1, 1],
        [1, 1], [0, 0],
        [-1, 0], [1, 0],
        [0, 0], [0, 1],
        [1, 0], [0, 0]
    ]
    S_W = [np.array(S_W[i:i+4]).reshape(2,2) for i in range(0, 28, 4)]

    # Standard factors for 2x1x2
    Std_U = [
        [1], [0],
        [1], [0],
        [0], [1],
        [0], [1]
    ]
    Std_U = [np.array(Std_U[i:i+2]).reshape(2,1) for i in range(0, 8, 2)]

    Std_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    Std_V = [np.array(Std_V[i:i+2]).reshape(1,2) for i in range(0, 8, 2)]

    Std_W = [
        [1, 0], [0, 0],
        [0, 0], [1, 0],
        [0, 1], [0, 0],
        [0, 0], [0, 1]
    ]
    Std_W = [np.array(Std_W[i:i+4]).reshape(2,2) for i in range(0, 16, 4)]

    col = 0
    for b in range(3):
        # Strassen part
        for r in range(7):
            if time.time() > _deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i, j]
            
            # Fill V
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j, kp]
            
            # Fill W
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp, i]
            
            col += 1

        # Standard part
        for r in range(4):
            if time.time() > _deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Fill U
            for i in range(2):
                U[i * m + 2, col] = Std_U[r][i, 0]
            
            # Fill V
            for kp in range(2):
                k_global = 2 * b + kp
                V[2 * p + k_global, col] = Std_V[r][0, kp]
            
            # Fill W
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = Std_W[r][kp, i]
            
            col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
