import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def verify(U, V, W):
        r = U.shape[1]
        res = np.zeros((nm, mp, pn), dtype=object)
        for t in range(r):
            res += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    if res[u, v, w] != 1:
                        return False
        valid_mask = np.zeros((nm, mp, pn), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    valid_mask[u, v, w] = True
        if np.any(res[~valid_mask] != 0):
            return False
        return True

    # ---------- Strassen construction (rank 46) ----------
    # Strassen for 2x2 block multiplication:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B10)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    #
    # U[i, (j,k,s)] = coefficient of A[i,j] in the s-th Strassen product input
    # V[(j,k), (j',k',s)] = coefficient of B[j',k'] in the s-th Strassen product input
    # W[(k,i), (j,k',s)] = coefficient of C[i,k] output in the s-th Strassen product
    #
    # Here each "block" is 1x5, so A[i,j] is the (i,j) entry, B[j,k] is the (j,k) entry.
    # The Strassen products M_s are 1x5 (since A block is 1x5, B block is 5x5, product is 1x5).
    # Wait, that's not right. Let me reconsider.
    #
    # Actually A is 2x5, B is 5x5. So A[i,j] for i in {0,1}, j in {0..4}.
    # B[j,k] for j in {0..4}, k in {0..4}.
    # C[i,k] for i in {0,1}, k in {0..4}.
    #
    # The "blocks" in the 2x2 structure are:
    #   A00 = A[0, :5] (1x5), A01 = A[0, :5]... 
    #
    # Hmm, this doesn't decompose into 2x2 blocks of matrices in the usual sense.
    # Instead, think of it as: C[i,k] = sum_j A[i,j] B[j,k].
    # We can group the sum over j into Strassen-like combinations.
    #
    # Alternative: Use the known rank-46 decomposition for 2x5x5.
    # Actually, let me use a different approach: the Strassen trick works when
    # we have 2x2 blocks. For 2x5x5, we can write:
    #   C[0,k] = sum_j A[0,j] B[j,k]
    #   C[1,k] = sum_j A[1,j] B[j,k]
    #
    # This is just two independent dot products, which is rank 2*5*5=50 in schoolbook.
    #
    # To use Strassen, we need to find a 2x2 substructure. One way:
    # Pair up the j indices. But 5 is odd, so we can't perfectly pair.
    #
    # Let me try a different known result. The tensor rank of <2,5,5> is known
    # to be at most 46 via Strassen-like constructions. Let me construct it explicitly.
    #
    # Actually, the cleanest way: treat the multiplication as computing C = A*B
    # where A is 2x5 and B is 5x5. We can view this as 5 independent 2x1 times 1x5
    # multiplications summed. Each 2x1x5 has rank 10 (schoolbook), total 50.
    #
    # For Strassen reduction, we need to find linear combinations. Let me try
    # the approach of combining pairs of j-indices.
    #
    # For j pairs (0,1), (2,3), and single j=4:
    # For each pair (j, j+1), we compute:
    #   C[i,k] contribution = A[i,j]*B[j,k] + A[i,j+1]*B[j+1,k]
    # This is a 2x2x2 tensor (i in {0,1}, k in {0..4}, but j is fixed pair).
    # Actually it's 2 x 2 x 5 if we keep k free, or we can think of it as
    # 5 independent 2x2x1 problems (one for each k).
    #
    # For each k, the 2x2x1 multiplication (A_2x2 times B_2x1) has rank 7 via Strassen.
    # So for 5 k-values and 2 j-pairs, we get 2*5*7 = 70? No, that's more than 50.
    #
    # I think I'm overcomplicating. Let me just use the random search but start
    # from a better initialization based on known structure.
    #
    # Actually, the simplest improvement: use the schoolbook as a starting point
    # and try to merge columns via linear dependencies.
    #
    # Let me try: initialize with schoolbook (rank 50), then try to find a rank-49
    # decomposition by finding a linear combination that eliminates one column.

    # ---------- Approach: Start from schoolbook, try to reduce rank ----------
    # Schoolbook: rank 50, each column corresponds to (i,j,k)
    # Try to find a rank-r decomposition by solving a system.

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def solve_for_W(U, V):
        r = U.shape[1]
        W_sol = np.zeros((pn, r), dtype=float)
        for w in range(pn):
            A_w = np.zeros((nm * mp, r), dtype=float)
            b_w = np.zeros(nm * mp, dtype=float)
            idx = 0
            for u in range(nm):
                for v in range(mp):
                    i1, j1 = divmod(u, m)
                    j2, k1 = divmod(v, p)
                    k2, i2 = divmod(w, n)
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        b_w[idx] = 1.0
                    for t in range(r):
                        A_w[idx, t] = U[u, t] * V[v, t]
                    idx += 1
            sol, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_sol[w, :] = sol
        return W_sol

    def round_to_half(X):
        return np.round(X * 2) / 2.0

    def round_to_int(X):
        return np.round(X).astype(float)

    # Try to find a rank-49 or lower decomposition
    # Strategy: random search with Gaussian init, solve for W, round, verify

    best_result = None
    best_rank = n * m * p

    # First, try a structured approach: perturb schoolbook slightly
    U0, V0, W0 = build_schoolbook()

    # Try random search for ranks 49 down to 40
    for r in range(49, 39, -1):
        if time.time() > deadline:
            break
        for trial in range(500):
            if time.time() > deadline:
                break
            np.random.seed(42 + r * 1000 + trial)
            scale = 1.0 / np.sqrt(r)
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale

            # Try both half-integer and integer rounding
            for rounder in [round_to_half, round_to_int]:
                U_r = rounder(U)
                V_r = rounder(V)
                if np.any(np.all(U_r == 0, axis=0)) or np.any(np.all(V_r == 0, axis=0)):
                    continue
                W_f = solve_for_W(U_r, V_r)
                W_r = rounder(W_f)
                if verify(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback: schoolbook
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
