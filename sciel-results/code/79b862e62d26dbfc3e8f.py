"""
5x5 matrix multiplication - rank 117 block decomposition.
Strategy: 5 = 2 + 2 + 1 split.
  - 2x2x2 blocks: Strassen (rank 7) for 6 such blocks (positions where all three dims are size-2)
  - 2x1x2, 2x2x1, 1x2x2, 1x1x2, 1x2x1, 2x1x1, 1x1x1 blocks: schoolbook

  2x2x2 blocks (rank 7 each): (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1) -> 6 blocks * 7 = 42
  2x2x1 blocks (rank 4 each): (0,1,2), (1,1,2) -> 2 * 4 = 8  (A is 2x2, B is 2x1, C is 2x1... wait)
  
  Let me recount. Split indices into groups: G0={0,1}, G1={2,3}, G2={4}.
  For each (i_group, j_group, k_group) we multiply A[i_group, j_group] * B[j_group, k_group] -> C[i_group, k_group].
  
  i_group has size 2 if in {G0, G1}, size 1 if G2.
  j_group has size 2 if in {G0, G1}, size 1 if G2.
  k_group has size 2 if in {G0, G1}, size 1 if G2.

  Blocks:
  (G0,G0,G0): 2x2x2 -> Strassen, rank 7
  (G0,G0,G1): 2x2x2 -> Strassen, rank 7
  (G0,G0,G2): 2x2x1 -> schoolbook, rank 4
  (G0,G1,G0): 2x2x2 -> Strassen, rank 7
  (G0,G1,G1): 2x2x2 -> Strassen, rank 7
  (G0,G1,G2): 2x2x1 -> schoolbook, rank 4
  (G0,G2,G0): 2x1x2 -> schoolbook, rank 4
  (G0,G2,G1): 2x1x2 -> schoolbook, rank 4
  (G0,G2,G2): 2x1x1 -> schoolbook, rank 2
  (G1,G0,G0): 2x2x2 -> Strassen, rank 7
  (G1,G0,G1): 2x2x2 -> Strassen, rank 7
  (G1,G0,G2): 2x2x1 -> schoolbook, rank 4
  (G1,G1,G0): 2x2x2 -> Strassen, rank 7
  (G1,G1,G1): 2x2x2 -> Strassen, rank 7
  (G1,G1,G2): 2x2x1 -> schoolbook, rank 4
  (G1,G2,G0): 2x1x2 -> schoolbook, rank 4
  (G1,G2,G1): 2x1x2 -> schoolbook, rank 4
  (G1,G2,G2): 2x1x1 -> schoolbook, rank 2
  (G2,G0,G0): 1x2x2 -> schoolbook, rank 4
  (G2,G0,G1): 1x2x2 -> schoolbook, rank 4
  (G2,G0,G2): 1x2x1 -> schoolbook, rank 2
  (G2,G1,G0): 1x2x2 -> schoolbook, rank 4
  (G2,G1,G1): 1x2x2 -> schoolbook, rank 4
  (G2,G1,G2): 1x2x1 -> schoolbook, rank 2
  (G2,G2,G0): 1x1x2 -> schoolbook, rank 2
  (G2,G2,G1): 1x1x2 -> schoolbook, rank 2
  (G2,G2,G2): 1x1x1 -> schoolbook, rank 1

  Total: 6*7 + (4+4+4+4+4+4+4+4+4+4+4+4+2+2+2+2+2+2+1)
        = 42 + (12*4 + 6*2 + 1)
        = 42 + (48 + 12 + 1)
        = 42 + 61
        = 103

  Wait, that's 103, which is better than the current 120. Let me recount the schoolbook terms.

  Actually, let me just implement it carefully and count.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 110.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n * m, dtype=float)
        V = np.zeros(m * p, dtype=float)
        W = np.zeros(p * n, dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook terms for A[i_list, j_list] * B[j_list, k_list] -> C[i_list, k_list]"""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """
        Strassen decomposition for 2x2 block multiplication.
        A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]
        """
        # M1 = (A00+A11)*(B00+B11)
        add_term({(i0, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1, (k1, i1): 1})
        # M2 = (A10+A11)*B00
        add_term({(i1, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1},
                 {(k0, i1): 1, (k1, i1): -1})
        # M3 = A00*(B01-B11)
        add_term({(i0, j0): 1},
                 {(j0, k1): 1, (j1, k1): -1},
                 {(k1, i0): 1, (k1, i1): 1})
        # M4 = A11*(B10-B00)
        add_term({(i1, j1): 1},
                 {(j1, k0): 1, (j0, k0): -1},
                 {(k0, i0): 1, (k0, i1): 1})
        # M5 = (A00+A01)*B11
        add_term({(i0, j0): 1, (i0, j1): 1},
                 {(j1, k1): 1},
                 {(k0, i0): -1, (k1, i0): 1})
        # M6 = (A10-A00)*(B00+B01)
        add_term({(i1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k1): 1},
                 {(k1, i1): 1})
        # M7 = (A01-A11)*(B10+B11)
        add_term({(i0, j1): 1, (i1, j1): -1},
                 {(j1, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1})

    # Block structure: 5 = 2 + 2 + 1
    # G0 = {0, 1}, G1 = {2, 3}, G2 = {4}
    G0 = [0, 1]
    G1 = [2, 3]
    G2 = [4]
    groups = [G0, G1, G2]
    sizes = [2, 2, 1]

    for ig in range(3):
        for jg in range(3):
            for kg in range(3):
                i_list = groups[ig]
                j_list = groups[jg]
                k_list = groups[kg]
                si, sj, sk = sizes[ig], sizes[jg], sizes[kg]
                
                if si == 2 and sj == 2 and sk == 2:
                    # 2x2x2: use Strassen
                    i0, i1 = i_list
                    j0, j1 = j_list
                    k0, k1 = k_list
                    strassen_2x2(i0, i1, j0, j1, k0, k1)
                else:
                    # Schoolbook
                    schoolbook_block(i_list, j_list, k_list)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
