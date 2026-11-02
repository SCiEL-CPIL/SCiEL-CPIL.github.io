import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    terms = []

    def add_term(U, V, W):
        terms.append((U, V, W))

    # Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    # M1 = (a11+a22)(b11+b22) -> +c11, +c22
    add_term(
        {(0*m+0):1, (1*m+1):1},
        {(0*p+0):1, (1*p+1):1},
        {(0*n+0):1, (1*n+1):1}
    )
    # M2 = (a21+a22)(b11) -> +c21, -c22
    add_term(
        {(1*m+0):1, (1*m+1):1},
        {(0*p+0):1},
        {(1*n+0):1, (1*n+1):-1}
    )
    # M3 = (a11)(b12-b22) -> +c12, +c22
    add_term(
        {(0*m+0):1},
        {(0*p+1):1, (1*p+1):-1},
        {(0*n+1):1, (1*n+1):1}
    )
    # M4 = (a22)(b21-b11) -> +c11, +c21
    add_term(
        {(1*m+1):1},
        {(1*p+0):1, (0*p+0):-1},
        {(0*n+0):1, (1*n+0):1}
    )
    # M5 = (a11+a12)(b22) -> -c11, +c12
    add_term(
        {(0*m+0):1, (0*m+1):1},
        {(1*p+1):1},
        {(0*n+0):-1, (0*n+1):1}
    )
    # M6 = (a21-a11)(b11+b12) -> +c22
    add_term(
        {(1*m+0):1, (0*m+0):-1},
        {(0*p+0):1, (0*p+1):1},
        {(1*n+1):1}
    )
    # M7 = (a12-a22)(b21+b22) -> +c11
    add_term(
        {(0*m+1):1, (1*m+1):-1},
        {(1*p+0):1, (1*p+1):1},
        {(0*n+0):1}
    )

    # Schoolbook blocks (every (i,j,k) triple not in the Strassen block)
    # A[I1][J2] * B[J2][K1]: i in {0,1}, j in {2,3,4}, k in {0,1}
    for i in [0,1]:
        for j in [2,3,4]:
            for k in [0,1]:
                add_term({(i*m+j):1}, {(j*p+k):1}, {(k*n+i):1})
    # A[I1][J1] * B[J1][K2]: i in {0,1}, j in {0,1}, k in {2,3,4}
    for i in [0,1]:
        for j in [0,1]:
            for k in [2,3,4]:
                add_term({(i*m+j):1}, {(j*p+k):1}, {(k*n+i):1})
    # A[I1][J2] * B[J2][K2]: i in {0,1}, j in {2,3,4}, k in {2,3,4}
    for i in [0,1]:
        for j in [2,3,4]:
            for k in [2,3,4]:
                add_term({(i*m+j):1}, {(j*p+k):1}, {(k*n+i):1})
    # A[I2][J1] * B[J1][K1]: i in {2}, j in {0,1}, k in {0,1}
    for i in [2]:
        for j in [0,1]:
            for k in [0,1]:
                add_term({(i*m+j):1}, {(j*p+k):1}, {(k*n+i):1})
    # A[I2][J2] * B[J2][K1]: i in {2}, j in {2,3,4}, k in {0,1}
    for i in [2]:
        for j in [2,3,4]:
            for k in [0,1]:
                add_term({(i*m+j):1}, {(j*p+k):1}, {(k*n+i):1})
    # A[I2][J1] * B[J1][K2]: i in {2}, j in {0,1}, k in {2,3,4}
    for i in [2]:
        for j in [0,1]:
            for k in [2,3,4]:
                add_term({(i*m+j):1}, {(j*p+k):1}, {(k*n+i):1})
    # A[I2][J2] * B[J2][K2]: i in {2}, j in {2,3,4}, k in {2,3,4}
    for i in [2]:
        for j in [2,3,4]:
            for k in [2,3,4]:
                add_term({(i*m+j):1}, {(j*p+k):1}, {(k*n+i):1})

    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c, (Ud, Vd, Wd) in enumerate(terms):
        for (r, v) in Ud.items():
            U[r][c] = v
        for (r, v) in Vd.items():
            V[r][c] = v
        for (r, v) in Wd.items():
            W[r][c] = v
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
