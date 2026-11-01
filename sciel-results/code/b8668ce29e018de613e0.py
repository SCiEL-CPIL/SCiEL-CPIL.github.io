import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set ranges
    def set_range(mat, start, end, val, col):
        for idx in range(start, end):
            mat[idx][col] = val

    # r=0: M1 = (A00+A11)(B00+B11)
    # U: A00 (idx 0), A11 (idx 3)
    U[0][0] = 1
    U[3][0] = 1
    # V: B00 (j=0, k=0..4), B11 (j=1, k=5..9)
    set_range(V, 0, 5, 1, 0)
    set_range(V, 15, 20, 1, 0)
    # W: C00 (i=0, k=0..4), C11 (i=1, k=5..9)
    # C00: W[2k+0] for k=0..4 -> 0,2,4,6,8
    for k in range(5):
        W[2*k + 0][0] = 1
    # C11: W[2k+1] for k=5..9 -> 11,13,15,17,19
    for k in range(5, 10):
        W[2*k + 1][0] = 1

    # r=1: M2 = (A10+A11)B00
    # U: A10 (idx 2), A11 (idx 3)
    U[2][1] = 1
    U[3][1] = 1
    # V: B00 (j=0, k=0..4)
    set_range(V, 0, 5, 1, 1)
    # W: C10 (i=1, k=0..4)
    for k in range(5):
        W[2*k + 1][1] = 1

    # r=2: M3 = A00(B01-B11)
    # U: A00 (idx 0)
    U[0][2] = 1
    # V: B01 (j=0, k=5..9), B11 (j=1, k=5..9)
    set_range(V, 5, 10, 1, 2)
    set_range(V, 15, 20, -1, 2)
    # W: C01 (i=0, k=5..9), C11 (i=1, k=5..9)
    for k in range(5, 10):
        W[2*k + 0][2] = 1
        W[2*k + 1][2] = 1

    # r=3: M4 = A11(B10-B00)
    # U: A11 (idx 3)
    U[3][3] = 1
    # V: B10 (j=1, k=0..4), B00 (j=0, k=0..4)
    set_range(V, 10, 15, 1, 3)
    set_range(V, 0, 5, -1, 3)
    # W: C00 (i=0, k=0..4), C10 (i=1, k=0..4)
    for k in range(5):
        W[2*k + 0][3] = 1
        W[2*k + 1][3] = 1

    # r=4: M5 = (A00+A01)B11
    # U: A00 (idx 0), A01 (idx 1)
    U[0][4] = 1
    U[1][4] = 1
    # V: B11 (j=1, k=5..9)
    set_range(V, 15, 20, 1, 4)
    # W: -C00 (i=0, k=0..4), C01 (i=0, k=5..9)
    for k in range(5):
        W[2*k + 0][4] = -1
    for k in range(5, 10):
        W[2*k + 0][4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    # U: A10 (idx 2), A00 (idx 0)
    U[2][5] = 1
    U[0][5] = -1
    # V: B00 (j=0, k=0..4), B01 (j=0, k=5..9) -> j=0, k=0..9
    set_range(V, 0, 10, 1, 5)
    # W: C11 (i=1, k=5..9)
    for k in range(5, 10):
        W[2*k + 1][5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    # U: A01 (idx 1), A11 (idx 3)
    U[1][6] = 1
    U[3][6] = -1
    # V: B10 (j=1, k=0..4), B11 (j=1, k=5..9) -> j=1, k=0..9
    set_range(V, 10, 20, 1, 6)
    # W: C00 (i=0, k=0..4)
    for k in range(5):
        W[2*k + 0][6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
