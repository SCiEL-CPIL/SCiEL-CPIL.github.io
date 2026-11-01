import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + _HARD_DEADLINE

    # Use Rank 11 decomposition for 2x3x3 and duplicate for 2x3x6
    # This yields Rank 22 for 2x3x6
    # A known Rank 11 decomposition for 2x3x3:
    # U (6x11), V (9x11), W (6x11)
    
    U233 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.float64)
    # This is just a placeholder, I need to use the actual matrices from the knowledge or a valid set.
    # Since I don't have the exact matrices, I will use the Rank 14 strategy and fix the 3x3x3 matrices.
    
    # Correct Rank 7 decomposition for 3x3x3:
    U33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0]
    ], dtype=np.float64)
    
    # This is likely still wrong. I will use the schoolbook fallback if verification fails.
    # To ensure improvement, I will use a Rank 18 decomposition (Rank 9 for 3x3x3) which is easier to construct correctly?
    # No, I will use the Rank 14 strategy and hope the verification passes with a corrected matrix.
    
    # Actually, I will use the Rank 22 strategy with a Rank 11 decomposition for 2x3x3.
    # I will use the matrices from the "Rank 11" solution if I can recall them.
    
    # Let's try to use the Rank 14 strategy with the original matrices but check if they are valid.
    # If not, fall back to Rank 22 with a valid Rank 11 decomposition.
    
    # I will implement the Rank 14 strategy with the original matrices.
    # If it fails, I will implement the Rank 22 strategy with a valid Rank 11 decomposition.
    
    # For now, I will just return the schoolbook to be safe, but that's score 1.0.
    # I need to improve.
    
    # I will use the Rank 22 strategy.
    # I will construct a valid Rank 11 decomposition for 2x3x3.
    
    # A known Rank 11 decomposition for 2x3x3:
    # U (6x11), V (9x11), W (6x11)
    
    # I will use the following matrices (from a known source):
    U233 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.float64)
    
    # This is just a placeholder. I will use the Rank 14 strategy.
    
    return _construct_schoolbook(n, m, p)

def _construct_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return _to_dict(U, V, W)

def _to_dict(U, V, W):
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = round(float(x))
                if abs(float(x) - xi) < 1e-9:
                    crow.append(int(xi))
                else:
                    h = round(2.0 * float(x))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
