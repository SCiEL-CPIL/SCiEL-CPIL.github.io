import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 2x2 by 2x14 matrix multiplication.
    
    Decomposes as direct sum of 7 independent 2x2 x 2x2 multiplication tensors,
    each using Strassen's rank-7 algorithm. Total rank = 7 * 7 = 49.
    """
    _t0 = time.time()
    _DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    # Strassen U rows: U[i, s] for i in 0..3 (a,b,c,d), s in 0..6
    U_rows = np.array([
        [1,  0,  1,  0,  1, -1,  0],  # a
        [0,  0,  0,  0,  1,  0,  1],  # b
        [0,  1,  0,  0,  0,  1,  0],  # c
        [1,  1,  0,  1,  0,  0, -1],  # d
    ], dtype=np.int64)

    # V local: V_local[s, v] where v indexes (e,f,g,h)
    V_local = np.array([
        [1,  0,  0,  1],   # M1: e+h
        [1,  0,  0,  0],   # M2: e
        [0,  1,  0, -1],   # M3: f-h
        [-1, 0,  1,  0],   # M4: g-e
        [0,  0,  0,  1],   # M5: h
        [1,  1,  0,  0],   # M6: e+f
        [0,  0,  1,  1],   # M7: g+h
    ], dtype=np.int64)

    # W local: W_local[s, w] where w indexes (C11, C12, C21, C22)
    W_local = np.array([
        [1,  0,  0,  1],   # M1
        [0,  0,  1, -1],   # M2
        [0,  1,  0,  1],   # M3
        [1,  0,  1,  0],   # M4
        [-1, 1,  0,  0],   # M5
        [0,  0,  0,  1],   # M6
        [1,  0,  0,  0],   # M7
    ], dtype=np.int64)

    # Build U: (n*m, rank) = (4, 49)
    U = np.zeros((n * m, rank), dtype=np.int64)
    for block in range(num_blocks):
        U[:, block * 7:(block + 1) * 7] = U_rows

    # Build V: (m*p, rank) = (28, 49)
    # Row index: local_row * p + 2*block + local_col
    # local_row in {0,1}, local_col in {0,1}
    V = np.zeros((m * p, rank), dtype=np.int64)
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            # v_idx 0=e (row0,col0), 1=f (row0,col1), 2=g (row1,col0), 3=h (row1,col1)
            V[0 * p + 2 * block + 0, col] = V_local[s, 0]  # e
            V[0 * p + 2 * block + 1, col] = V_local[s, 1]  # f
            V[1 * p + 2 * block + 0, col] = V_local[s, 2]  # g
            V[1 * p + 2 * block + 1, col] = V_local[s, 3]  # h

    # Build W: (p*n, rank) = (28, 49)
    # W[k*n + i, col] where k = output column, i = output row
    # C11: k=2*block, i=0; C12: k=2*block+1, i=0
    # C21: k=2*block, i=1; C22: k=2*block+1, i=1
    W = np.zeros((p * n, rank), dtype=np.int64)
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            W[2 * block * n + 0, col] = W_local[s, 0]       # C11
            W[(2 * block + 1) * n + 0, col] = W_local[s, 1] # C12
            W[2 * block * n + 1, col] = W_local[s, 2]       # C21
            W[(2 * block + 1) * n + 1, col] = W_local[s, 3] # C22

    if time.time() - _t0 > _DEADLINE:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
