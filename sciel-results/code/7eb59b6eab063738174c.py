import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val

        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):-1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):-1})

    # 2+3 split
    # C22 = A22*B22 + A23*B32
    strassen_2x2(0, 1, 0, 1, 0, 1)  # 7
    strassen_2x2(0, 1, 2, 3, 0, 1)  # 7
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # 4
    # C22: 18

    # C23 = A22*B23 + A23*B33
    strassen_2x2(0, 1, 0, 1, 2, 3)  # 7
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # 4
    strassen_2x2(0, 1, 2, 3, 2, 3)  # 7
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # 4
    for i in [0, 1]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # 4
    for i in [0, 1]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})  # 2
    # C23: 28

    # C32 = A32*B22 + A33*B32
    strassen_2x2(2, 3, 0, 1, 0, 1)  # 7
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # 4
    strassen_2x2(2, 3, 2, 3, 0, 1)  # 7
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # 4
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # 4
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # 2
    # C32: 28

    # C33 = A32*B23 + A33*B33
    strassen_2x2(2, 3, 0, 1, 2, 3)  # 7
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # 4
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # 4
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # 2
    strassen_2x2(2, 3, 2, 3, 2, 3)  # 7
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # 4
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # 4
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})  # 1
    # C33: 33

    # Total: 18+28+28+33 = 107

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
