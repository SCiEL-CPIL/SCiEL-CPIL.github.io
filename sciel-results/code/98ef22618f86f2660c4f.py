"""Rewritten program: hybrid rank-18 decomposition using two Strassen blocks and one schoolbook block."""
DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    # We split the output columns k=0,1 (Strassen block A) and k=2,3 (Strassen block B)
    # and k=4 (schoolbook block).
    #
    # Block A: A(2x2) @ B[:, 0:2] -> C[:, 0:2]
    #   Use Strassen's 7-term formula.
    # Block B: A(2x2) @ B[:, 2:4] -> C[:, 2:4]
    #   Use Strassen's 7-term formula.
    # Block C: A(2x2) @ B[:, 4:5] -> C[:, 4:5]
    #   Use schoolbook: 2*2*1 = 4 terms.
    #
    # Total rank = 7 + 7 + 4 = 18.

    # Strassen factors for 2x2x2 (U: 4x7, V: 4x7, W: 4x7)
    # Convention: U[i*m+j, c] for A element (i,j); V[j*p+k, c] for B element (j,k); W[k*n+i, c] for C element (k,i)
    #
    # Strassen's P's:
    # P1 = (A11)(B11+B22)
    # P2 = (A11+A22)(B22)
    # P3 = (A21+A22)(B11)
    # P4 = (A22)(B21+B22)
    # P5 = (A11)(B12-B22)
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # U for block (A factors): columns correspond to P1..P7
    # A row index = i*2 + j (since m=2 for the 2x2 block)
    # P1: A11 -> (0,0)=0; also (1,1) is not in P1. So U for P1: [1,0,0,0] for A11... wait.
    # Actually each P has a specific A factor and B factor.
    # P1 = A11 * (B11+B22): U column has 1 at A11 (index 0). V column has 1 at B11 (index 0) and 1 at B22 (index 3).
    # P2 = (A11+A22) * B22: U has 1 at A11 (index 0) and 1 at A22 (index 3). V has 1 at B22 (index 3).
    # P3 = (A21+A22) * B11: U has 1 at A21 (index 2) and 1 at A22 (index 3). V has 1 at B11 (index 0).
    # P4 = A22 * (B21+B22): U has 1 at A22 (index 3). V has 1 at B21 (index 2) and 1 at B22 (index 3).
    # P5 = A11 * (B12-B22): U has 1 at A11 (index 0). V has 1 at B12 (index 1) and -1 at B22 (index 3).
    # P6 = (A21-A11) * (B11+B12): U has 1 at A21 (index 2) and -1 at A11 (index 0). V has 1 at B11 (index 0) and 1 at B12 (index 1).
    # P7 = (A12-A22) * (B21+B22): U has 1 at A12 (index 1) and -1 at A22 (index 3). V has 1 at B21 (index 2) and 1 at B22 (index 3).

    strassen_U = [
        # col P1, P2, P3, P4, P5, P6, P7
        [1,  1,  0,  0,  1, -1,  0],  # A11 (index 0)
        [0,  0,  0,  0,  0,  0,  1],  # A12 (index 1)
        [0,  0,  1,  0,  0,  1,  0],  # A21 (index 2)
        [0,  1,  1,  1,  0,  0, -1],  # A22 (index 3)
    ]
    strassen_V = [
        [1,  0,  1,  0,  0,  1,  0],  # B11 (index 0)
        [0,  0,  0,  0,  1,  1,  0],  # B12 (index 1)
        [0,  0,  0,  1,  0,  0,  1],  # B21 (index 2)
        [1,  1,  0,  1, -1, 0,  1],  # B22 (index 3)
    ]
    # W: C factors. C row index = k*2 + i (since n=2, p=2 for the block; output columns k=0,1 and rows i=0,1)
    # C11 = P1 + P4 - P5 - P7 -> row (k=0,i=0)=0: [1, 0, 0, 1, -1, 0, -1]
    # C12 = P3 + P5 -> row (k=1,i=0)=1: [0, 0, 1, 0, 1, 0, 0]
    # C21 = P2 + P4 -> row (k=0,i=1)=2: [0, 1, 0, 1, 0, 0, 0]
    # C22 = P1 - P2 + P3 + P6 -> row (k=1,i=1)=3: [1, -1, 1, 0, 0, 1, 0]
    strassen_W = [
        [1,  0,  0,  1, -1,  0, -1],  # C(k=0,i=0)=C11 -> index 0*2+0=0
        [0,  0,  1,  0,  1,  0,  0],  # C(k=1,i=0)=C12 -> index 1*2+0=2
        [0,  1,  0,  1,  0,  0,  0],  # C(k=0,i=1)=C21 -> index 0*2+1=1
        [1, -1,  1,  0,  0,  1,  0],  # C(k=1,i=1)=C22 -> index 1*2+1=3
    ]
    # Reorder strassen_W to match index k*2+i:
    # index 0: k=0,i=0 -> C11 -> [1, 0, 0, 1, -1, 0, -1]
    # index 1: k=0,i=1 -> C21 -> [0, 1, 0, 1, 0, 0, 0]
    # index 2: k=1,i=0 -> C12 -> [0, 0, 1, 0, 1, 0, 0]
    # index 3: k=1,i=1 -> C22 -> [1, -1, 1, 0, 0, 1, 0]
    strassen_W = [
        [1,  0,  0,  1, -1,  0, -1],  # idx 0
        [0,  1,  0,  1,  0,  0,  0],  # idx 1
        [0,  0,  1,  0,  1,  0,  0],  # idx 2
        [1, -1,  1,  0,  0,  1,  0],  # idx 3
    ]

    # Build full U, V, W
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 4 x 18
    V = [[0] * rank for _ in range(m * p)]  # 10 x 18
    W = [[0] * rank for _ in range(p * n)]  # 10 x 18

    # Block A: uses columns 0-6 (Strassen)
    # A columns: j=0,1 -> rows 0,1,2,3 (i*m+j: i in 0..1, j in 0..1)
    for col in range(7):
        for i in range(2):
            for j in range(2):
                idx = i * m + j
                U[idx][col] = strassen_U[i * 2 + j][col]
    # V for block A: B columns k=0,1 -> rows j*p+k for j in 0..1, k in 0..1
    for col in range(7):
        for j in range(2):
            for k in range(2):
                idx = j * p + k
                V[idx][col] = strassen_V[j * 2 + k][col]
    # W for block A: C columns k=0,1 -> rows k*n+i for k in 0..1, i in 0..1
    for col in range(7):
        for k in range(2):
            for i in range(2):
                idx = k * n + i
                W[idx][col] = strassen_W[k * 2 + i][col]

    # Block B: uses columns 7-13 (Strassen)
    for col in range(7):
        offset = 7 + col
        for i in range(2):
            for j in range(2):
                idx = i * m + j
                U[idx][offset] = strassen_U[i * 2 + j][col]
    # V for block B: B columns k=2,3 -> rows j*p+k for j in 0..1, k in 2,3
    for col in range(7):
        offset = 7 + col
        for j in range(2):
            for k in range(2):
                idx = j * p + (2 + k)
                V[idx][offset] = strassen_V[j * 2 + k][col]
    # W for block B: C columns k=2,3 -> rows k*n+i for k in 2,3, i in 0..1
    for col in range(7):
        offset = 7 + col
        for k in range(2):
            for i in range(2):
                idx = (2 + k) * n + i
                W[idx][offset] = strassen_W[k * 2 + i][col]

    # Block C: uses columns 14-17 (Schoolbook, 4 terms for 2x2x1)
    # Schoolbook: 4 terms, each (i,j,k=4)
    # U: A(i,j) -> 1
    # V: B(j,4) -> 1
    # W: C(4,i) -> 1
    term = 0
    for i in range(2):
        for j in range(2):
            col = 14 + term
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            term += 1

    return {"U": U, "V": V, "W": W}
