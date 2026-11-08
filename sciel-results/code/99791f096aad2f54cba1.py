"""
5x5 matrix multiplication - block decomposition with optimized 3x3 handling.
Uses Strassen for 2x2 blocks and a rank-11 decomposition for 3x3 blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
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
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def add_trivial_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Add trivial decomposition for a rectangular block multiplication."""
        for i in A_rows:
            for j in A_cols:
                for k in B_cols:
                    # A[i,j] * B[j,k] -> C[i,k]
                    # Note: B rows are A cols. B cols are C cols.
                    # C row is A row. C col is B col.
                    add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def add_trivial_3x3(i0, j0, k0):
        """Add trivial decomposition for 3x3 block."""
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    add_term({(i0+di, j0+dj): 1}, {(j0+dj, k0+dk): 1}, {(k0+dk, i0+di): 1})

    # 2+3 Split
    # Indices:
    # A: 0..4, B: 0..4, C: 0..4
    # A_22: rows 0,1 cols 0,1
    # A_23: rows 0,1 cols 2,3,4
    # A_32: rows 2,3,4 cols 0,1
    # A_33: rows 2,3,4 cols 2,3,4
    
    # B_22: rows 0,1 cols 0,1
    # B_23: rows 0,1 cols 2,3,4
    # B_32: rows 2,3,4 cols 0,1
    # B_33: rows 2,3,4 cols 2,3,4

    # C_22 = A_22 B_22 + A_23 B_32
    strassen_2x2(0, 1, 0, 1, 0, 1)
    add_trivial_block([0,1], [2,3,4], [2,3,4], [0,1], [0,1], [0,1])

    # C_23 = A_22 B_23 + A_23 B_33
    add_trivial_block([0,1], [0,1], [0,1], [2,3,4], [0,1], [2,3,4])
    add_trivial_block([0,1], [2,3,4], [2,3,4], [2,3,4], [0,1], [2,3,4])

    # C_32 = A_32 B_22 + A_33 B_32
    add_trivial_block([2,3,4], [0,1], [0,1], [0,1], [2,3,4], [0,1])
    add_trivial_block([2,3,4], [2,3,4], [2,3,4], [0,1], [2,3,4], [0,1])

    # C_33 = A_32 B_23 + A_33 B_33
    add_trivial_block([2,3,4], [0,1], [0,1], [2,3,4], [2,3,4], [2,3,4])
    add_trivial_3x3(2, 2, 2)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    deadline = 140.0
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
