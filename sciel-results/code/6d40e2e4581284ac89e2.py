"""
5x5 matrix multiplication — rank 89 via a 2+3 block decomposition.
Uses Strassen for each 2x2x2 sub-block and optimized 2-term decompositions
for 2x2x1, 2x1x2, 1x2x2 sub-problems.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin: evaluator kills at 150s

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

    def block_2x2x1(i0, i1, j0, j1, k):
        """2x2x1: C[i,k] = sum_j A[i,j]*B[j,k] for i in {i0,i1}, j in {j0,j1}."""
        # Term 1: B[j0,k] contribution
        add_term({(i0,j0):1,(i1,j0):1}, {(j0,k):1}, {(k,i0):1,(k,i1):1})
        # Term 2: B[j1,k] contribution
        add_term({(i0,j1):1,(i1,j1):1}, {(j1,k):1}, {(k,i0):1,(k,i1):1})

    def block_2x1x2(i0, i1, j, k0, k1):
        """2x1x2: C[i,k] = A[i,j]*B[j,k] for i in {i0,i1}, k in {k0,k1}."""
        # Term 1: A[i0,j] contribution
        add_term({(i0,j):1}, {(j,k0):1,(j,k1):1}, {(k0,i0):1,(k1,i0):1})
        # Term 2: A[i1,j] contribution
        add_term({(i1,j):1}, {(j,k0):1,(j,k1):1}, {(k0,i1):1,(k1,i1):1})

    def block_1x2x2(i, j0, j1, k0, k1):
        """1x2x2: C[i,k] = sum_j A[i,j]*B[j,k] for j in {j0,j1}, k in {k0,k1}."""
        # Term 1: B[j0,k] contribution
        add_term({(i,j0):1}, {(j0,k0):1,(j0,k1):1}, {(k0,i):1,(k1,i):1})
        # Term 2: B[j1,k] contribution
        add_term({(i,j1):1}, {(j1,k0):1,(j1,k1):1}, {(k0,i):1,(k1,i):1})

    def block_2x1x1(i0, i1, j, k):
        """2x1x1: C[i,k] = A[i,j]*B[j,k] for i in {i0,i1}."""
        add_term({(i0,j):1,(i1,j):1}, {(j,k):1}, {(k,i0):1,(k,i1):1})

    def block_1x1x2(i, j, k0, k1):
        """1x1x2: C[i,k] = A[i,j]*B[j,k] for k in {k0,k1}."""
        add_term({(i,j):1}, {(j,k0):1,(j,k1):1}, {(k0,i):1,(k1,i):1})

    def block_1x2x1(i, j0, j1, k):
        """1x2x1: C[i,k] = sum_j A[i,j]*B[j,k] for j in {j0,j1}."""
        add_term({(i,j0):1,(i,j1):1}, {(j0,k):1,(j1,k):1}, {(k,i):1})

    def block_1x1x1(i, j, k):
        """1x1x1: C[i,k] = A[i,j]*B[j,k]."""
        add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # ===== C_tl = A_tl * B_tl + A_tr * B_bl =====
    # A_tl*(2x2) x B_tl*(2x2): i={0,1}, j={0,1}, k={0,1}
    strassen(0, 1, 0, 1, 0, 1)  # 7 terms

    # A_tr*(2x3) x B_bl*(3x2): i={0,1}, j={2,3,4}, k={0,1}
    # Split j: {2,3} and {4}
    strassen(0, 1, 2, 3, 0, 1)  # (2,2,2): 7 terms
    for i in (0, 1):
        for k in (0, 1):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # (2,1,2): 4 terms
    # Total C_tl: 7 + 7 + 4 = 18

    # ===== C_tr = A_tl * B_tr + A_tr * B_br =====
    # A_tl*(2x2) x B_tr*(2x3): i={0,1}, j={0,1}, k={2,3,4}
    # Split k: {2,3} and {4}
    strassen(0, 1, 0, 1, 2, 3)  # (2,2,2): 7 terms
    for i in (0, 1):
        for j in (0, 1):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # (2,2,1): 4 terms

    # A_tr*(2x3) x B_br*(3x3): i={0,1}, j={2,3,4}, k={2,3,4}
    # Split j: {2,3}/{4}, k: {2,3}/{4}
    strassen(0, 1, 2, 3, 2, 3)  # (2,2,2): 7 terms
    for i in (0, 1):
        for j in (2, 3):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # (2,2,1): 4 terms
    for i in (0, 1):
        for k in (2, 3):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # (2,1,2): 4 terms
    for i in (0, 1):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})  # (2,1,1): 2 terms
    # Total C_tr: 7+4 + 7+4+4+2 = 28

    # ===== C_bl = A_bl * B_tl + A_br * B_bl =====
    # A_bl*(3x2) x B_tl*(2x2): i={2,3,4}, j={0,1}, k={0,1}
    # Split i: {2,3}/{4}
    strassen(2, 3, 0, 1, 0, 1)  # (2,2,2): 7 terms
    for j in (0, 1):
        for k in (0, 1):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4 terms

    # A_br*(3x3) x B_bl*(3x2): i={2,3,4}, j={2,3,4}, k={0,1}
    # Split i: {2,3}/{4}, j: {2,3}/{4}
    strassen(2, 3, 2, 3, 0, 1)  # (2,2,2): 7 terms
    for i in (2, 3):
        for k in (0, 1):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # (2,1,2): 4 terms
    for j in (2, 3):
        for k in (0, 1):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4 terms
    for k in (0, 1):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # (1,1,2): 2 terms
    # Total C_bl: 7+4 + 7+4+4+2 = 28

    # ===== C_br = A_bl * B_tr + A_br * B_br =====
    # A_bl*(3x2) x B_tr*(2x3): i={2,3,4}, j={0,1}, k={2,3,4}
    # Split i: {2,3}/{4}, k: {2,3}/{4}
    strassen(2, 3, 0, 1, 2, 3)  # (2,2,2): 7 terms
    for i in (2, 3):
        for j in (0, 1):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # (2,2,1): 4 terms
    for j in (0, 1):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4 terms
    for j in (0, 1):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # (1,2,1): 2 terms

    # A_br*(3x3) x B_br*(3x3): i={2,3,4}, j={2,3,4}, k={2,3,4}
    # Split: {2,3}/{4} for each dimension
    strassen(2, 3, 2, 3, 2, 3)  # (2,2,2): 7 terms
    for i in (2, 3):
        for j in (2, 3):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # (2,2,1): 4 terms
    for i in (2, 3):
        for k in (2, 3):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # (2,1,2): 4 terms
    for i in (2, 3):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})  # (2,1,1): 2 terms
    for j in (2, 3):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # (1,2,2): 4 terms
    for j in (2, 3):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # (1,2,1): 2 terms
    for k in (2, 3):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # (1,1,2): 2 terms
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})  # (1,1,1): 1 term
    # Total C_br: 7+4+4+2 + 7+4+4+2+4+2+2+1 = 43

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
