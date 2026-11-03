import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_cols = []
    V_cols = []
    W_cols = []
    covered = set()

    # --- 1. Apply Strassen 2x2x2 blocks on disjoint sub-cubes ---
    # i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}
    # This covers 2*2*2*6 = 48 of the 105 tensor entries.
    # Rank cost: 6 blocks * 7 = 42. (Saving 6 over schoolbook's 48)

    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))

        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1

        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1

        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1

        # M1 = (A00+A11)(B00+B11)
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})

        # M2 = (A10+A11)B00
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})

        # M3 = A00(B01-B11)
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})

        # M4 = A11(B10-B00)
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})

        # M5 = (A00+A01)B11
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})

        # M6 = (A10-A00)(B00+B01)
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})

        # M7 = (A01-A11)(B10+B11)
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})

    # Define blocks
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen_block(i0, i0+1, j0, j0+1, k0, k0+1)

    # --- 2. Handle remaining entries with Schoolbook ---
    # Remaining entries are those NOT in the covered set.
    # Total entries: 105. Covered: 48. Remaining: 57.
    # Rank cost: 57.
    # Total Rank so far: 42 + 57 = 99.

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})

    # --- 3. Rank Reduction via Gaussian Elimination ---
    # We can potentially reduce the rank by finding linear dependencies among the terms.
    # The terms are vectors in R^{nm} x R^{mp} x R^{pn}.
    # Actually, the rank of the tensor is the minimum number of terms.
    # We have a specific decomposition. We can try to see if any term is a linear combination
    # of others in a way that preserves the tensor structure.
    # However, for generic entries, the rank is fixed.
    # But since our decomposition is structured, maybe we can compress?
    # The "rank" of the decomposition is the number of terms.
    # If we can find a basis for the span of the U vectors, V vectors, W vectors
    # such that the sum of outer products is preserved...
    
    # Let's represent the current decomposition as matrices U, V, W of shape (N, R), (M, R), (P, R).
    # T = sum_r U[:,r] V[:,r]^T ... wait, it's a 3-way tensor.
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # We can try to reduce R by checking if the columns of U, V, W are linearly dependent
    # in a correlated way.
    # This is equivalent to checking if the tensor has a lower rank decomposition.
    # For now, let's just use the 99 rank decomposition.
    # Is it possible to do better?
    # Strassen on 2x2x2 saves 1.
    # We used 6 blocks.
    # Can we use a 3x3x3 block?
    # Rank of 3x3x3 is 23. Schoolbook is 27. Saving 4.
    # If we use one 3x3x3 block (i=0,1,2; j=0,1,2; k=0,1,2), we save 4.
    # Remaining:
    # j=3,4,5 (3 cols), k=0..6 (7 cols) for i=0,1,2?
    # No, the block covers j=0,1,2 and k=0,1,2.
    # Remaining j: 3,4. Remaining k: 3,4,5,6.
    # Plus the rest of the j=0,1,2 for k=3,4,5,6.
    # And i=2 for the Strassen blocks?
    
    # Let's stick to the robust 99 rank solution for now, but verify it.
    # If time permits, we can try to implement the 3x3x3 de Groote block.
    # But 99 is already an improvement over 105.
    
    # Let's verify the 99 rank decomposition.
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val

    # Verify correctness
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # T_comp[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Using einsum: 'ar,br,cr->abc'
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)

    if error > 1e-9:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Check time
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to improve using de Groote 3x3x3 block
    # This is complex to implement correctly from scratch without a verified table.
    # Given the risk of introducing errors and the time limit, we will stick to the
    # verified Strassen-based rank-99 decomposition.
    
    # However, we can try to see if we can pack the schoolbook terms more efficiently
    # or if there are other simple optimizations.
    # For 3x5x7, 99 is a solid result.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
