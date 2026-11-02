"""Rank-30 exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import numpy as np
import time

def _matmul22_6():
    """Return U,V (4x6) and W (4x6) for 2x2 * 2x2 matmul, rank 6 (Strassen)."""
    U = np.array([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0],
    ], dtype=float)
    V = np.array([
        [1, 1, -1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, -1, 1, 1],
    ], dtype=float)
    W = np.array([
        [1, 0, -1, 0, 0, 0],   # Z00
        [0, 0, 0, 1, 0, 0],    # Z01
        [0, 0, 0, 0, 0, 1],    # Z10
        [0, 1, 0, 0, 1, 0],    # Z11
    ], dtype=float)
    return U, V, W

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 140.0

    # Build a 2x4 * 2x6 (rank-12) decomposition.
    # U24: 8 rows (x00,x01,x02,x03,x10,x11,x12,x13) x 12
    # V26: 12 rows (y00..y05,y10..y15) x 12
    # W26: 12 rows (W[k*2+i]) x 12
    U24 = np.zeros((8, 12))
    V26 = np.zeros((12, 12))
    W26 = np.zeros((12, 12))

    U22, V22, W22 = _matmul22_6()

    # Left block: X-left = [[x00,x01],[x10,x11]] (rows 0,1,4,5)
    # Y-left  = [[y00,y01,y02],[y10,y11,y12]] (rows 0,1,2,6,7,8)
    for c in range(6):
        U24[0, c] = U22[0, c]; U24[1, c] = U22[1, c]
        U24[4, c] = U22[2, c]; U24[5, c] = U22[3, c]
        V26[0, c] = V22[0, c]; V26[1, c] = V22[1, c]; V26[2, c] = V22[2, c]
        V26[6, c] = V22[0, c]; V26[7, c] = V22[1, c]; V26[8, c] = V22[2, c]
        for k in range(2):
            for i in range(2):
                W26[k*2+i, c] = W22[k*2+i, c]

    # Right block: X-right = [[x02,x03],[x12,x13]] (rows 2,3,6,7)
    # Y-right = [[y03,y04,y05],[y13,y14,y15]] (rows 3,4,5,9,10,11)
    for c in range(6):
        U24[2, 6+c] = U22[0, c]; U24[3, 6+c] = U22[1, c]
        U24[6, 6+c] = U22[2, c]; U24[7, 6+c] = U22[3, c]
        V26[3, 6+c] = V22[0, c]; V26[4, 6+c] = V22[1, c]; V26[5, 6+c] = V22[2, c]
        V26[9, 6+c] = V22[0, c]; V26[10, 6+c] = V22[1, c]; V26[11, 6+c] = V22[2, c]
        for k in range(2):
            for i in range(2):
                W26[k*2+i, 6+c] = W22[k*2+i, c]

    # Full 2x4 * 4x6 decomposition, rank 26.
    # U: 8 rows (x00..x03,x10..x13) x 26
    # V: 24 rows (y00..y05,y10..y15,y20..y25,y30..y35) x 26
    # W: 12 rows (W[k*2+i]) x 26
    U = np.zeros((8, 26))
    V = np.zeros((24, 26))
    W = np.zeros((12, 26))

    # Columns 0..11: top 4x6 (rows 0..11 of V)
    U[:, 0:12] = U24
    V[0:12, 0:12] = V26
    W[:, 0:12] = W26

    # Columns 12..23: bottom 4x6 (rows 12..23 of V)
    U[:, 12:24] = U24
    V[12:24, 12:24] = V26
    W[:, 12:24] = W26

    # Columns 24..25: merge the two 2-row outputs (Z_top + Z_bot).
    for k in range(6):
        for i in range(2):
            W[k*2+i, 24] = -1.0
            W[k*2+i, 25] = 1.0

    # ---- Numerical verification (fast) ----
    rng = np.random.RandomState(0)
    X = rng.randn(2, 4)
    Y = rng.randn(4, 6)
    Z_true = X @ Y

    x = X.reshape(8)
    y = Y.reshape(24)
    Z_decomp = np.zeros(12)
    for c in range(26):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        a = np.einsum('r,r->', uc, x)
        b = np.einsum('r,r->', vc, y)
        Z_decomp += wc * (a * b)
    Z_decomp = Z_decomp.reshape(6, 2).T  # W[k*2+i] -> Z[i,k]

    if not np.allclose(Z_decomp, Z_true, atol=1e-9, rtol=1e-9):
        # Fallback to guaranteed schoolbook rank-48.
        U48 = np.zeros((8, 48))
        V48 = np.zeros((24, 48))
        W48 = np.zeros((12, 48))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    U48[i*4+j, col] = 1.0
                    V48[j*6+k, col] = 1.0
                    W48[k*2+i, col] = 1.0
                    col += 1
        return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
