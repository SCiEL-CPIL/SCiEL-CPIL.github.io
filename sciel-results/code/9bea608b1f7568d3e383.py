"""
5x5 matrix multiplication — rank 117 via a 2+3 block decomposition.
Uses Strassen for each 2x2x2 sub-block and schoolbook for the remaining
sub-problems.  The 2+3 split gives:
  C22: 7+11=18, C23: 11+17=28, C32: 11+17=28, C33: 17+26=43
Total = 18+28+28+43 = 117
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin: evaluator kills at 150s

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_map, v_map, w_map):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for (i, j), val in u_map.items():
            u[i * m + j] = val
        for (j, k), val in v_map.items():
            v[j * p + k] = val
        for (k, i), val in w_map.items():
            w[k * n + i] = val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def strassen(i0, i1, j0, j1, k0, k1):
        """Strassen 2x2x2 decomposition for rows i0,i1; cols j0,j1; B-cols k0,k1."""
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1,(i1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(k0,i0):1,(k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1,(i1,j1):1}, {(j0,k0):1}, {(k0,i1):1,(k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(k1,i0):1,(k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,i0):1,(k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1,(i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1,(k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1,(i0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1,(i1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(k0,i0):1})

    # C_tl = A_tl * B_tl + A_tr * B_bl
    # A_tl*(2,2)x B_tl*(2,2): indices i={0,1},j={0,1},k={0,1}
    strassen(0, 1, 0, 1, 0, 1)
    # A_tr*(2,3)x B_bl*(3,2): i={0,1},j={2,3,4},k={0,1}
    # Split j: {2,3} and {4}
    strassen(0, 1, 2, 3, 0, 1)  # A(2x2)*B(2x2)
    for i in (0, 1):
        for k in (0, 1):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A(2x1)*B(1x2): 4 terms

    # C_tr = A_tl * B_tr + A_tr * B_br
    # A_tl*(2,2)x B_tr*(2,3): i={0,1},j={0,1},k={2,3,4}
    # Split k: {2,3} and {4}
    strassen(0, 1, 0, 1, 2, 3)  # A(2x2)*B(2x2): 7
    for i in (0, 1):
        for j in (0, 1):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A(2x2)*B(2x1): 4

    # A_tr*(2,3)x B_br*(3,3): i={0,1},j={2,3,4},k={2,3,4}
    # Split j: {2,3}/{4}, k: {2,3}/{4}
    strassen(0, 1, 2, 3, 2, 3)  # (2,2,2): 7
    for i in (0, 1):
        for j in (2, 3):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # (2,2,1): 4
    for i in (0, 1):
        for k in (2, 3):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # (2,1,2): 4
    for i in (0, 1):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})  # (2,1,1): 2

    # C_bl = A_bl * B_tl + A_br * B_bl
    # A_bl*(3,2)x B_tl*(2,2): i={2,3,4},j={0,1},k={0,1}
    # Split i: {2,3}/{4}
    strassen(2, 3, 0, 1, 0, 1)  # (2,2,2): 7
    for j in (0, 1):
        for k in (0, 1):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4

    # A_br*(3,3)x B_bl*(3,2): i={2,3,4},j={2,3,4},k={0,1}
    # Split i: {2,3}/{4}, j: {2,3}/{4}
    strassen(2, 3, 2, 3, 0, 1)  # (2,2,2): 7
    for i in (2, 3):
        for k in (0, 1):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # (2,1,2): 4
    for j in (2, 3):
        for k in (0, 1):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4
    for k in (0, 1):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # (1,1,2): 2

    # C_br = A_bl * B_tr + A_br * B_br
    # A_bl*(3,2)x B_tr*(2,3): i={2,3,4},j={0,1},k={2,3,4}
    # Split i: {2,3}/{4}, k: {2,3}/{4}
    strassen(2, 3, 0, 1, 2, 3)  # (2,2,2): 7
    for i in (2, 3):
        for j in (0, 1):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # (2,2,1): 4
    for j in (0, 1):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4
    for j in (0, 1):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # (1,2,1): 2

    # A_br*(3,3)x B_br*(3,3): i={2,3,4},j={2,3,4},k={2,3,4}
    # Split: {2,3}/{4} for each
    strassen(2, 3, 2, 3, 2, 3)  # (2,2,2): 7
    for i in (2, 3):
        for j in (2, 3):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # (2,2,1): 4
    for i in (2, 3):
        for k in (2, 3):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # (2,1,2): 4
    for i in (2, 3):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})  # (2,1,1): 2
    for j in (2, 3):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4
    for j in (2, 3):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # (1,2,1): 2
    for k in (2, 3):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # (1,1,2): 2
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})  # (1,1,1): 1

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
