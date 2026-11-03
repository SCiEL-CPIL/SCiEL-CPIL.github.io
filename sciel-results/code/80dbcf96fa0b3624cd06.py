"""
5x5 matrix multiplication - optimized rank decomposition (2+3 split).
Uses Strassen for 2x2 blocks and a 2+1 split for 3x3 blocks.
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
        """Add Strassen terms for A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k1+1] -> C[i0:i1+1, k0:k1+1]"""
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

    def schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook terms for A[i_list, j_list] * B[j_list, k_list] -> C[i_list, k_list]"""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # 2+3 split: small={0,1}, large={2,3,4}
    s = [0, 1]
    l = [2, 3, 4]

    # C_ss = A_ss B_ss + A_sl B_ls
    strassen_2x2(s[0], s[1], s[0], s[1], s[0], s[1])  # A_ss B_ss: 7
    schoolbook_block(s, l, s)  # A_sl B_ls: 2*3*2=12

    # C_sl = A_ss B_sl + A_sl B_ll
    # A_ss (2x2) * B_sl (2x3): schoolbook 2*2*3=12
    schoolbook_block(s, s, l)
    # A_sl (2x3) * B_ll (3x3): use 2+1 split on B_ll
    # A_sl = [A_sl2 (2x2), A_sl4 (2x1)] where A_sl2 has j in {2,3}, A_sl4 has j=4
    # B_ll = [B_ll2 (2x3), B_ll4 (1x3)] where B_ll2 has j in {2,3}, B_ll4 has j=4
    # A_sl2 * B_ll2: 2x2 * 2x3 -> Strassen on 2x2 part + schoolbook on 2x1 part
    # Actually simpler: just schoolbook 2*3*3=18
    schoolbook_block(s, l, l)

    # C_ls = A_ls B_ss + A_ll B_ls
    # A_ls (3x2) * B_ss (2x2): schoolbook 3*2*2=12
    schoolbook_block(l, s, s)
    # A_ll (3x3) * B_ls (3x2): use 2+1 split
    # A_ll = [A_ll22 (2x2), A_ll24 (2x1), A_ll42 (1x2), A_ll44 (1x1)]
    # B_ls = [B_ls2 (2x2), B_ls4 (1x2)]
    # A_ll22 * B_ls2: 2x2 * 2x2 -> Strassen 7
    strassen_2x2(l[0], l[1], l[0], l[1], s[0], s[1])
    # A_ll24 * B_ls4: 2x1 * 1x2 -> schoolbook 2*1*2=4
    schoolbook_block([l[0], l[1]], [l[2]], s)
    # A_ll42 * B_ls2: 1x2 * 2x2 -> schoolbook 1*2*2=4
    schoolbook_block([l[2]], [l[0], l[1]], s)
    # A_ll44 * B_ls4: 1x1 * 1x2 -> schoolbook 1*1*2=2
    schoolbook_block([l[2]], [l[2]], s)

    # C_ll = A_ls B_sl + A_ll B_ll
    # A_ls (3x2) * B_sl (2x3): schoolbook 3*2*3=18
    schoolbook_block(l, s, l)
    # A_ll (3x3) * B_ll (3x3): use 2+1 split
    # A_ll = [A_ll22 (2x2), A_ll24 (2x1), A_ll42 (1x2), A_ll44 (1x1)]
    # B_ll = [B_ll22 (2x2), B_ll24 (2x1), B_ll42 (1x2), B_ll44 (1x1)]
    # A_ll22 * B_ll22: 2x2 * 2x2 -> Strassen 7
    strassen_2x2(l[0], l[1], l[0], l[1], l[0], l[1])
    # A_ll22 * B_ll24: 2x2 * 2x1 -> schoolbook 2*2*1=4
    schoolbook_block([l[0], l[1]], [l[0], l[1]], [l[2]])
    # A_ll24 * B_ll42: 2x1 * 1x2 -> schoolbook 2*1*2=4
    schoolbook_block([l[0], l[1]], [l[2]], [l[0], l[1]])
    # A_ll24 * B_ll44: 2x1 * 1x1 -> schoolbook 2*1*1=2
    schoolbook_block([l[0], l[1]], [l[2]], [l[2]])
    # A_ll42 * B_ll22: 1x2 * 2x2 -> schoolbook 1*2*2=4
    schoolbook_block([l[2]], [l[0], l[1]], [l[0], l[1]])
    # A_ll42 * B_ll24: 1x2 * 2x1 -> schoolbook 1*2*1=2
    schoolbook_block([l[2]], [l[0], l[1]], [l[2]])
    # A_ll44 * B_ll42: 1x1 * 1x2 -> schoolbook 1*1*2=2
    schoolbook_block([l[2]], [l[2]], [l[0], l[1]])
    # A_ll44 * B_ll44: 1x1 * 1x1 -> schoolbook 1
    schoolbook_block([l[2]], [l[2]], [l[2]])

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
