import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 21

    # Strassen coefficients for 2x2 matrix multiplication
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]

    U = np.zeros((4, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))

    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]

            u_vals = sc['U']
            for idx in range(4):
                if u_vals[idx] != 0:
                    U[idx, col] = u_vals[idx]

            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            v_vals = sc['V']
            for idx in range(4):
                if v_vals[idx] != 0:
                    V[v_idx[idx], col] = v_vals[idx]

            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            w_vals = sc['W']
            for idx in range(4):
                if w_vals[idx] != 0:
                    W[w_idx[idx], col] = w_vals[idx]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
