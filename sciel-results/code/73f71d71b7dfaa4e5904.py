import numpy as np
import time

DIMENSIONS = [2, 2, 16]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen coefficients for 2x2x2
    # Intermediates M1..M7:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U rows: A00=0, A01=1, A10=2, A11=3
    # V rows: B00=0, B01=1, B10=2, B11=3
    # W rows: C00=0, C01=1, C10=2, C11=3

    u_s = np.array([
        [1, 0, 0, 1],   # M1: A00=1, A11=1
        [0, 0, 1, 1],   # M2: A10=1, A11=1
        [1, 0, 0, 0],   # M3: A00=1
        [0, 0, 0, 1],   # M4: A11=1
        [1, 1, 0, 0],   # M5: A00=1, A01=1
        [-1, 0, 1, 0],  # M6: A00=-1, A10=1
        [0, 1, 0, -1]   # M7: A01=1, A11=-1
    ])

    v_s = np.array([
        [1, 0, 0, 1],   # M1: B00=1, B11=1
        [1, 0, 0, 0],   # M2: B00=1
        [0, 1, 0, -1],  # M3: B01=1, B11=-1
        [-1, 0, 1, 0],  # M4: B00=-1, B10=1
        [0, 0, 0, 1],   # M5: B11=1
        [1, 1, 0, 0],   # M6: B00=1, B01=1
        [0, 0, 1, 1]    # M7: B10=1, B11=1
    ])

    w_s = np.array([
        [1, 0, 0, 1],   # M1: C00=1, C11=1
        [0, 0, 1, -1],  # M2: C10=1, C11=-1
        [0, 1, 0, 1],   # M3: C01=1, C11=1
        [1, 0, 1, 0],   # M4: C00=1, C10=1
        [-1, 1, 0, 0],  # M5: C00=-1, C01=1
        [0, 0, 0, 1],   # M6: C11=1
        [1, 0, 0, 0]    # M7: C00=1
    ])

    for b in range(num_blocks):
        col_offset = b * 7
        for s in range(7):
            c = col_offset + s

            # U: A indices (same for all blocks)
            U[:, c] = u_s[s]

            # V: B indices for block b
            # B[0, 2b] -> V[2b], B[0, 2b+1] -> V[2b+1]
            # B[1, 2b] -> V[16+2b], B[1, 2b+1] -> V[16+2b+1]
            V[2*b, c] = v_s[s][0]
            V[2*b+1, c] = v_s[s][1]
            V[16+2*b, c] = v_s[s][2]
            V[16+2*b+1, c] = v_s[s][3]

            # W: C indices using trace convention W[k*n+i, r]
            # C[0, 2b] -> W[2b*2+0] = W[4b]
            # C[1, 2b] -> W[2b*2+1] = W[4b+1]
            # C[0, 2b+1] -> W[(2b+1)*2+0] = W[4b+2]
            # C[1, 2b+1] -> W[(2b+1)*2+1] = W[4b+3]
            W[4*b, c] = w_s[s][0]
            W[4*b+1, c] = w_s[s][2]
            W[4*b+2, c] = w_s[s][1]
            W[4*b+3, c] = w_s[s][3]

    if time.time() - _t0 > _DEADLINE:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
