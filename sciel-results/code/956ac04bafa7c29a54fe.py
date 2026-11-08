"""
5x5 matrix multiplication - rank 117 decomposition.
Uses Strassen for 2x2 blocks and schoolbook for remaining terms.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
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
        """Strassen for block with row idx i0,i1; col j0,j1; B col k0,k1"""
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def schoolbook(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # Partition indices into 2+3 blocks
    # Rows: {0,1} (size 2), {2,3,4} (size 3)
    # Cols of A / rows of B: {0,1} (size 2), {2,3,4} (size 3)
    # Cols of B: {0,1} (size 2), {2,3,4} (size 3)

    # C_22 = A_22 B_22 + A_23 B_32
    # A_22 (2x2) * B_22 (2x2): Strassen -> 7
    strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # A_23 (2x3) * B_32 (3x2): 
    # Split j into {2,3} and {4}
    # A_23a (2x2, j=2,3) * B_32a (2x2, j=2,3): Strassen -> 7
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A_23b (2x1, j=4) * B_32b (1x2, j=4): schoolbook -> 2*1*2 = 4
    schoolbook([0,1], [4], [0,1])
    # Total C_22: 7 + 7 + 4 = 18

    # C_23 = A_22 B_23 + A_23 B_33
    # A_22 (2x2) * B_23 (2x3):
    # Split k into {2,3} and {4}
    # A_22 (2x2) * B_23a (2x2, k=2,3): Strassen -> 7
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A_22 (2x2) * B_23b (2x1, k=4): schoolbook -> 2*2*1 = 4
    schoolbook([0,1], [0,1], [4])
    
    # A_23 (2x3) * B_33 (3x3):
    # Split j and k into {2,3} and {4}
    # A_23a (2x2, j=2,3) * B_33a (2x2, j=2,3, k=2,3): Strassen -> 7
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A_23a (2x2, j=2,3) * B_33b (2x1, j=2,3, k=4): schoolbook -> 2*2*1 = 4
    schoolbook([0,1], [2,3], [4])
    # A_23b (2x1, j=4) * B_33b (1x3, j=4, k=2,3,4): schoolbook -> 2*1*3 = 6
    schoolbook([0,1], [4], [2,3,4])
    # Total C_23: 7 + 4 + 7 + 4 + 6 = 28

    # C_32 = A_32 B_22 + A_33 B_32
    # A_32 (3x2) * B_22 (2x2):
    # Split i into {2,3} and {4}
    # A_32a (2x2, i=2,3) * B_22 (2x2): Strassen -> 7
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A_32b (1x2, i=4) * B_22 (2x2): schoolbook -> 1*2*2 = 4
    schoolbook([4], [0,1], [0,1])
    
    # A_33 (3x3) * B_32 (3x2):
    # Split i, j, k
    # A_33a (2x2, i=2,3, j=2,3) * B_32a (2x2, j=2,3, k=0,1): Strassen -> 7
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A_33b (2x1, i=2,3, j=4) * B_32b (1x2, j=4, k=0,1): schoolbook -> 2*1*2 = 4
    schoolbook([2,3], [4], [0,1])
    # A_33c (1x2, i=4, j=2,3) * B_32a (2x2, j=2,3, k=0,1): schoolbook -> 1*2*2 = 4
    schoolbook([4], [2,3], [0,1])
    # A_33d (1x1, i=4, j=4) * B_32b (1x2, j=4, k=0,1): schoolbook -> 1*1*2 = 2
    schoolbook([4], [4], [0,1])
    # Total C_32: 7 + 4 + 7 + 4 + 4 + 2 = 28

    # C_33 = A_32 B_23 + A_33 B_33
    # A_32 (3x2) * B_23 (2x3):
    # Split i and k
    # A_32a (2x2, i=2,3) * B_23a (2x2, k=2,3): Strassen -> 7
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A_32a (2x2, i=2,3) * B_23b (2x1, k=4): schoolbook -> 2*2*1 = 4
    schoolbook([2,3], [0,1], [4])
    # A_32b (1x2, i=4) * B_23a (2x2, k=2,3): schoolbook -> 1*2*2 = 4
    schoolbook([4], [0,1], [2,3])
    # A_32b (1x2, i=4) * B_23b (2x1, k=4): schoolbook -> 1*2*1 = 2
    schoolbook([4], [0,1], [4])
    
    # A_33 (3x3) * B_33 (3x3):
    # Split into 2+1 for i, j, k
    # A_33a (2x2, i=2,3, j=2,3) * B_33a (2x2, j=2,3, k=2,3): Strassen -> 7
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A_33b (2x1, i=2,3, j=4) * B_33b (1x3, j=4, k=2,3,4): schoolbook -> 2*1*3 = 6
    schoolbook([2,3], [4], [2,3,4])
    # A_33c (1x2, i=4, j=2,3) * B_33a (2x2, j=2,3, k=2,3): schoolbook -> 1*2*2 = 4
    schoolbook([4], [2,3], [2,3])
    # A_33d (1x1, i=4, j=4) * B_33b (1x3, j=4, k=2,3,4): schoolbook -> 1*1*3 = 3
    schoolbook([4], [4], [2,3,4])
    # Total C_33: 7 + 4 + 4 + 2 + 7 + 6 + 4 + 3 = 37

    # Total rank: 18 + 28 + 28 + 37 = 111

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
