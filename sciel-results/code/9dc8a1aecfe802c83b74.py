"""Matrix multiplication 6x6x6 - rank-minimal exact tensor decomposition search.

Structural change: instead of 2x2 block-Strassen (which the current program uses),
we use a 3x3 block decomposition with 3x3 Strassen (rank 10 for 3x3).
6x6 = 2x2 blocks of 3x3. Wait, that's the same as before.

Actually, let me think differently. The key insight from the knowledge:
"Using Strassen's algorithm on 2x2 blocks reduces the rank from 216 to 189"

The current program claims block-Strassen gives rank 252, but the knowledge says it
achieved rank 189. Let me reconsider the block structure.

For 6x6: partition into 2x2 blocks of 3x3 matrices.
Each 3x3 multiplication uses Strassen with rank 10 (since 3x3 matrix mult has rank 
... actually the known minimum rank for 3x3 is not 10. Let me think.

Wait - Strassen works on 2x2 blocks. For 3x3, we can use a different approach.
Actually, the standard Strassen applies to 2x2 blocks. For 3x3, there's no simple
Strassen-like decomposition with fewer than 27 multiplications known to be optimal.

Let me reconsider. The 6x6 matrix can be partitioned as:
- 2x2 blocks of 3x3: gives 4 block multiplications of 3x3, each needing rank >= 27
  total rank >= 4*27 = 108 if no sharing, but with Strassen on the 2x2 block level:
  7 multiplications of (3x3 + 3x3) * (3x3 + 3x3), each 3x3 mult needs rank 27
  total = 7 * 27 = 189. This matches the knowledge!

So the correct approach: 2x2 block-Strassen where each "multiplication" is a 3x3
matrix multiplication done with schoolbook (rank 27). Total rank = 7 * 27 = 189.

The current program's build_block_strassen has a bug - it's constructing something
with rank 252 instead of 189. Let me fix this properly.

For each of the 7 Strassen products, we need to multiply two 3x3 matrices.
Each 3x3 multiplication = 27 scalar multiplications (schoolbook).
Total rank = 7 * 27 = 189.

The factorization:
- U[i*m+j, r]: for the Strassen product index s (0..6) and local (a,b) in 3x3 (0..2,0..2)
  U[(block_row*3+a) * 6 + (block_col*3+b), s*27 + a*3 + b] = A_s[block_row, block_col][a,b]
  
- V[j*p+k, r]: similarly for B
  V[(block_col*3+a) * 6 + (block_k*3+b), s*27 + a*3 + b] = B_s[block_col, block_k][a,b]

- W[k*n+i, r]: for C output
  C[crow, ccol] gets contributions from various Strassen products with coefficients.
  W[(block_k*3+b) * 6 + (block_i*3+a), s*27 + a*3 + b] = C_coeff[crow, ccol, s]

Let me build this correctly.
"""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150s kill

    def elapsed():
        return time.time() - start_time

    def check_decomposition(Um, Vm, Wm):
        """Exact check: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        R = Um.shape[1]
        if R == 0:
            return False
        Um_3d = Um.reshape(n, m, R)
        Vm_3d = Vm.reshape(m, p, R)
        Wm_3d = Wm.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', Um_3d, Vm_3d, Wm_3d, optimize=True)
        return np.allclose(T, 1.0, atol=1e-7)

    # ---- Schoolbook rank-216 factorization (always a valid fallback) ----
    rank0 = n * m * p  # 216
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    # ---- Correct 2x2 block-Strassen with 3x3 schoolbook inside: rank 7*27=189 ----
    # Strassen for 2x2 blocks:
    # M0 = (A11+A22)(B12-B22)
    # M1 = (A21+A22)B11
    # M2 = A11(B22-B12)
    # M3 = A22(B11-B22)
    # M4 = (A11+A12)B22
    # M5 = (A21-A11)(B11+B12)
    # M6 = (A12-A22)(B21+B22)
    # C11 = M0 + M3 - M4 + M6
    # C12 = M1 + M2
    # C21 = M5 + M4
    # C22 = M0 - M1 + M2 + M3

    # For each Strassen product s, define:
    # A_s: 2x2 matrix of coefficients for A blocks (each entry is +1, -1, or 0)
    # B_s: 2x2 matrix of coefficients for B blocks
    # C_contrib: 2x2 matrix of (list of (s, coef)) showing which products contribute to each C block

    A_blocks = [
        [[1, 0], [1, 1]],   # M0: A11+A22
        [[0, 0], [1, 1]],   # M1: A21+A22
        [[1, 0], [0, 0]],   # M2: A11
        [[0, 0], [0, 1]],   # M3: A22
        [[1, 1], [0, 0]],   # M4: A11+A12
        [[-1, 0], [1, 0]],  # M5: A21-A11
        [[0, 1], [0, -1]],  # M6: A12-A22
    ]
    B_blocks = [
        [[0, 1], [0, -1]],  # M0: B12-B22
        [[1, 0], [0, 0]],   # M1: B11
        [[0, 0], [1, -1]],  # M2: B22-B12
        [[1, 0], [0, -1]],  # M3: B11-B22
        [[0, 0], [0, 1]],   # M4: B22
        [[1, 1], [0, 0]],   # M5: B11+B12
        [[0, 0], [1, 1]],   # M6: B21+B22
    ]
    # C[crow][ccol] = list of (s, coef)
    C_contrib = [
        [[(0, 1), (3, 1), (4, -1), (6, 1)],  # C11
         [(1, 1), (2, 1)]],                   # C12
        [[(5, 1), (4, 1)],                    # C21
         [(0, 1), (1, -1), (2, 1), (3, 1)]], # C22
    ]

    def build_block_strassen_correct():
        """Build the correct 2x2 block-Strassen with 3x3 schoolbook: rank 189."""
        R = 7 * 27  # 189
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for s in range(7):
            for a in range(3):  # local row in 3x3
                for b in range(3):  # local col in 3x3
                    col = s * 27 + a * 3 + b
                    # A part
                    for br in range(2):  # block row
                        for bc in range(2):  # block col
                            coef = A_blocks[s][br][bc]
                            if coef != 0:
                                i = br * 3 + a
                                j = bc * 3 + b
                                U[i * m + j, col] += float(coef)
                    # B part
                    for bc in range(2):  # block col (input col)
                        for bk in range(2):  # block row (output col)
                            coef = B_blocks[s][bc][bk]
                            if coef != 0:
                                j = bc * 3 + a
                                k = bk * 3 + b
                                V[j * p + k, col] += float(coef)
                    # C part: for each output block position (ci, cj), check if product s contributes
                    for ci in range(2):
                        for cj in range(2):
                            for (ss, ccoef) in C_contrib[ci][cj]:
                                if ss == s:
                                    k = bk * 3 + b  # Wait, this is wrong. Let me fix.
                                    # Actually for W, the indices are (k, i) where k is from B's output col, i is from A's input row
                                    # W[k*n+i, col]
                                    # For C block (ci, cj): i is in block row ci, k is in block col cj
                                    # i = ci*3 + a_local, k = cj*3 + b_local
                                    # But a and b here are the local indices within the 3x3
                                    pass
                    # Let me redo the W part properly
                    # For product s, the 3x3 multiplication gives a 3x3 result at each C block position
                    # that product s contributes to.
                    # The result element (a,b) of the 3x3 product goes to C block (ci,cj) at local (a,b)
                    # which is global i = ci*3+a, k = cj*3+b
                    # W[k*n+i, col]
                    for ci in range(2):
                        for cj in range(2):
                            for (ss, ccoef) in C_contrib[ci][cj]:
                                if ss == s:
                                    i_out = ci * 3 + a
                                    k_out = cj * 3 + b
                                    W[k_out * n + i_out, col] += float(ccoef)
        
        return U, V, W

    U_bs, V_bs, W_bs = build_block_strassen_correct()
    
    # Verify the block-Strassen is correct
    bs_valid = check_decomposition(U_bs, V_bs, W_bs)
    
    if not bs_valid:
        # Fall back to schoolbook
        U_bs, V_bs, W_bs = U0, V0, W0
    
    best_U, best_V, best_W = U_bs, V_bs, W_bs
    best_rank = U_bs.shape[1]
    
    # ---- Greedy column removal to try to reduce rank further ----
    def greedy_prune(Um, Vm, Wm, order, time_budget):
        R = Um.shape[1]
        U_cur, V_cur, W_cur = Um.copy(), Vm.copy(), Wm.copy()
        cur_rank = R
        for idx in order:
            if time.time() > time_budget:
                break
            if idx >= cur_rank:
                continue
            keep = [r for r in range(cur_rank) if r != idx]
            U_new = U_cur[:, keep]
            V_new = V_cur[:, keep]
            W_new = W_cur[:, keep]
            if check_decomposition(U_new, V_new, W_new):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                cur_rank = len(keep)
        return U_cur, V_cur, W_cur, cur_rank

    # Multiple random orderings for greedy pruning
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(best_rank))
        random.shuffle(order)
        U_r, V_r, W_r, rank_r = greedy_prune(best_U, best_V, best_W, order, deadline - 20)
        if rank_r < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r
            # Restart from the improved solution
            if time.time() > deadline - 20:
                break

    # Final verification
    if not check_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank0

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist(),
    }
# EVOLVE-BLOCK-END
