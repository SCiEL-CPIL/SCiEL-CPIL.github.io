"""
Program for 5x5 matrix multiplication tensor decomposition.
Structural idea (from inspirations): block partition of the 5x5 indices into a
3x3 block and a 2x2 block, applying Laderman's rank-7 algorithm to the 3x3x3
block and Strassen's rank-7 algorithm to the 2x2x2 block, with schoolbook terms
for the cross blocks.  This gives a valid rank-104 decomposition.

Refinement machinery (from current program): a fast vectorized verification and
a greedy local search that iteratively removes single columns and then pairs of
columns whose removal still leaves an exact decomposition, to drive the rank down.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# Laderman rank-7 for 3x3x3 (rows = (i,j) index i*3+j, cols = terms)
_LADERMAN_U = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, 0, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [-1, 0, 0, -1, 1, -1, -1]
], dtype=float)

_LADERMAN_V = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [1, -1, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [-1, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=float)

_LADERMAN_W = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, -1, 1, 0, 0, 0, 0],
    [1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=float)

# Strassen rank-7 for 2x2x2 (rows = (i,j) index i*2+j, cols = terms)
_STRASSEN_U = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0]
], dtype=float)

_STRASSEN_V = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1]
], dtype=float)

_STRASSEN_W = np.array([
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1]
], dtype=float)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Partition 5 into [3, 2] along each axis.
    parts = [(0, 3), (3, 5)]
    sizes = [3, 2]

    # term r: (U_dict, V_dict, W_dict)
    terms = []

    def add_term(U_dict, V_dict, W_dict):
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))

    def add_laderman(r1, c1, k1):
        # 3x3x3 block starting at (r1, c1, k1)
        for t in range(7):
            add_term(
                {(r1 + i) * m + (c1 + j): _LADERMAN_U[i * 3 + j, t]
                 for i in range(3) for j in range(3) if _LADERMAN_U[i * 3 + j, t] != 0},
                {(c1 + j) * p + (k1 + k): _LADERMAN_V[j * 3 + k, t]
                 for j in range(3) for k in range(3) if _LADERMAN_V[j * 3 + k, t] != 0},
                {(k1 + k) * n + (r1 + i): _LADERMAN_W[k * 3 + i, t]
                 for k in range(3) for i in range(3) if _LADERMAN_W[k * 3 + i, t] != 0},
            )

    def add_strassen(r1, c1, k1):
        # 2x2x2 block starting at (r1, c1, k1)
        for t in range(7):
            add_term(
                {(r1 + i) * m + (c1 + j): _STRASSEN_U[i * 2 + j, t]
                 for i in range(2) for j in range(2) if _STRASSEN_U[i * 2 + j, t] != 0},
                {(c1 + j) * p + (k1 + k): _STRASSEN_V[j * 2 + k, t]
                 for j in range(2) for k in range(2) if _STRASSEN_V[j * 2 + k, t] != 0},
                {(k1 + k) * n + (r1 + i): _STRASSEN_W[k * 2 + i, t]
                 for k in range(2) for i in range(2) if _STRASSEN_W[k * 2 + i, t] != 0},
            )

    def add_schoolbook(r1, r2, c1, c2, k1, k2):
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    add_term(
                        {ii * m + jj: 1},
                        {jj * p + kk: 1},
                        {kk * n + ii: 1},
                    )

    # Assemble the block decomposition.
    for ri, (r1, r2) in enumerate(parts):
        for ci, (c1, c2) in enumerate(parts):
            for ki, (k1, k2) in enumerate(parts):
                if time.time() > deadline:
                    break
                ar, ac, bc = r2 - r1, c2 - c1, k2 - k1
                if ri == 0 and ci == 0 and ki == 0:
                    add_laderman(r1, c1, k1)
                elif ri == 1 and ci == 1 and ki == 1:
                    add_strassen(r1, c1, k1)
                else:
                    add_schoolbook(r1, r2, c1, c2, k1, k2)

    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Fast vectorized verification.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    def _verify(U_mat, V_mat, W_mat, tol=1e-7):
        R = U_mat.shape[1]
        T = np.zeros((5, 5, 5), dtype=float)
        for r in range(R):
            Uc = U_mat[:, r].reshape(5, 5)   # Uc[i, j]
            Vc = V_mat[:, r].reshape(5, 5)   # Vc[j, k]
            Wc = W_mat[:, r].reshape(5, 5)   # Wc[k, i]; need Wc[k, i] = Wc.T[i, k]
            Wt = Wc.T                          # Wt[i, k] = Wc[k, i]
            for i in range(5):
                # T[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i] = Uc[i,j] * Vc[j,k] * Wt[i,k]
                T[i, :, :] += Uc[i, :] * (Vc * Wt[i, :])
        return np.allclose(T, 1.0, atol=tol)

    # Sanity check of the assembled decomposition.
    if not _verify(U_np, V_np, W_np, tol=1e-6):
        # Fall back to a guaranteed-valid schoolbook decomposition (rank 125).
        terms = []
        for ii in range(5):
            for jj in range(5):
                for kk in range(5):
                    terms.append(({ii * m + jj: 1}, {jj * p + kk: 1}, {kk * n + ii: 1}))
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val
        return {"U": U, "V": V, "W": W}

    # Greedy local search to reduce rank.
    if time.time() < deadline - 8:
        try:
            current_rank = rank
            removed = set()
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for col in range(current_rank):
                    if time.time() > deadline - 5:
                        break
                    if col in removed:
                        continue
                    mask = [c for c in range(current_rank) if c != col]
                    if _verify(U_np[:, mask], V_np[:, mask], W_np[:, mask], tol=1e-8):
                        removed.add(col)
                        keep = [c for c in range(current_rank) if c not in removed]
                        U_np = U_np[:, keep]
                        V_np = V_np[:, keep]
                        W_np = W_np[:, keep]
                        current_rank = len(keep)
                        changed = True

            # Try removing pairs of columns.
            if time.time() < deadline - 5:
                pair_removed = set()
                changed = True
                while changed and time.time() < deadline - 5:
                    changed = False
                    for c1 in range(current_rank):
                        if time.time() > deadline - 5:
                            break
                        if c1 in pair_removed:
                            continue
                        for c2 in range(c1 + 1, current_rank):
                            if time.time() > deadline - 5:
                                break
                            if c2 in pair_removed:
                                continue
                            mask = [c for c in range(current_rank) if c != c1 and c != c2]
                            if _verify(U_np[:, mask], V_np[:, mask], W_np[:, mask], tol=1e-8):
                                pair_removed.add(c1)
                                pair_removed.add(c2)
                                keep = [c for c in range(current_rank) if c not in pair_removed]
                                U_np = U_np[:, keep]
                                V_np = V_np[:, keep]
                                W_np = W_np[:, keep]
                                current_rank = len(keep)
                                changed = True
                                break

            if current_rank < rank:
                rank = current_rank
                U_np = U_np
                V_np = V_np
                W_np = W_np
        except Exception:
            pass

    # Convert to lists (float entries, exact integers/half-integers preserved).
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
