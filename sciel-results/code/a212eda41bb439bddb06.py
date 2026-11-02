"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Structural idea (from inspiration 1):
  Split A (2x3) into A0 (2x1, j=0) and A2 (2x2, j=1,2).
  - A0 * B0 (2x1 * 1x10): naive, 2*1*10 = 20 multiplications.
  - A2 * B2 (2x2 * 2x10): split B2 into 5 blocks of 2x2, apply Strassen (7 mults each) = 35.
  Total rank = 55.

Refinement:
  Verify the solution with vectorized numpy, then search for a lower rank
  using structured perturbations of the Strassen decomposition.
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Build the full tensor efficiently: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Use einsum for the full contraction
    T_reconstructed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    # Check all required entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False

    # Check all other entries are zero (sample if too many)
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    # Check a sample of zero-positions
    total_zeros = (n * m) * (m * p) * (p * n) - len(nonzero_set)
    sample_size = min(500, total_zeros)
    for _ in range(sample_size):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.

    A is 2x3: columns j=0,1,2.
    B is 3x10: rows j=0,1,2.
    C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k].

    Part A: j=0 -> A[i,0]*B[0,k] for i in {0,1}, k in 0..9.  20 terms.
    Part B: j=1,2 -> A2 (2x2) * B2 (2x10). Split B2 into 5 blocks of 2 columns.
             Each 2x2 * 2x2 block uses Strassen: 7 multiplications. 5*7 = 35.
    Total: 55.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10. Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries)
        u_11 = 0 * m + 1  # A[0,1]
        u_12 = 0 * m + 2  # A[0,2]
        u_21 = 1 * m + 1  # A[1,1]
        u_22 = 1 * m + 2  # A[1,2]

        # Strassen indices for V (B2 block entries)
        v_11 = 1 * p + k0  # B[1,k0]
        v_12 = 1 * p + k1  # B[1,k1]
        v_21 = 2 * p + k0  # B[2,k0]
        v_22 = 2 * p + k1  # B[2,k1]

        # W indices for outputs
        w_00 = k0 * n + 0  # C[0,k0]
        w_01 = k1 * n + 0  # C[0,k1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = k1 * n + 1  # C[1,k1]

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _structured_search_rank(n, m, p, target_rank, deadline):
    """
    Structured search: start from the rank-55 Strassen decomposition and try
    to merge/remove columns to reach target_rank. This is much more likely to
    succeed than pure random search because it respects the block structure.
    """
    U55, V55, W55 = _build_strassen55(n, m, p)
    U55_arr = np.array(U55, dtype=np.float64)
    V55_arr = np.array(V55, dtype=np.float64)
    W55_arr = np.array(W55, dtype=np.float64)

    # Strategy: try to eliminate (55 - target_rank) columns by finding
    # linear dependencies or by merging blocks.
    # For target_rank=54, we need to eliminate 1 column.
    # For target_rank=53, we need to eliminate 2 columns.
    
    num_to_remove = 55 - target_rank
    if num_to_remove <= 0:
        return None

    # Try all single-column removals (for rank 54)
    if num_to_remove == 1:
        for col_to_remove in range(55):
            if time.time() > deadline - 15:
                break
            U_trial = np.delete(U55_arr, col_to_remove, axis=1)
            V_trial = np.delete(V55_arr, col_to_remove, axis=1)
            W_trial = np.delete(W55_arr, col_to_remove, axis=1)
            
            # Check if this still works
            T = np.einsum('ar,br,cr->abc', U_trial, V_trial, W_trial)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if abs(T[a, b, c] - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Check zero entries
                nonzero_set = set()
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            nonzero_set.add((i * m + j, j * p + k, k * n + i))
                for _ in range(200):
                    a = random.randint(0, n * m - 1)
                    b = random.randint(0, m * p - 1)
                    c = random.randint(0, p * n - 1)
                    if (a, b, c) not in nonzero_set:
                        if abs(T[a, b, c]) > 1e-9:
                            valid = False
                            break
                if valid:
                    # Convert back to list format with half-integer check
                    U_list = U_trial.tolist()
                    V_list = V_trial.tolist()
                    W_list = W_trial.tolist()
                    if _is_valid_entries(U_list, V_list, W_list):
                        return U_list, V_list, W_list

    # For rank 53, try all pairs of column removals
    if num_to_remove == 2:
        for c1 in range(55):
            if time.time() > deadline - 15:
                break
            for c2 in range(c1 + 1, 55):
                if time.time() > deadline - 15:
                    break
                U_trial = np.delete(U55_arr, [c1, c2], axis=1)
                V_trial = np.delete(V55_arr, [c1, c2], axis=1)
                W_trial = np.delete(W55_arr, [c1, c2], axis=1)
                
                T = np.einsum('ar,br,cr->abc', U_trial, V_trial, W_trial)
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            if abs(T[a, b, c] - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    nonzero_set = set()
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                nonzero_set.add((i * m + j, j * p + k, k * n + i))
                    for _ in range(100):
                        a = random.randint(0, n * m - 1)
                        b = random.randint(0, m * p - 1)
                        c = random.randint(0, p * n - 1)
                        if (a, b, c) not in nonzero_set:
                            if abs(T[a, b, c]) > 1e-9:
                                valid = False
                                break
                    if valid:
                        U_list = U_trial.tolist()
                        V_list = V_trial.tolist()
                        W_list = W_trial.tolist()
                        if _is_valid_entries(U_list, V_list, W_list):
                            return U_list, V_list, W_list

    return None


def _random_search_rank(n, m, p, rank, values, trials, deadline):
    """Random search for a decomposition of the given rank."""
    for _ in range(trials):
        if time.time() > deadline - 15:
            break
        U = [[random.choice(values) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(rank)] for _ in range(p * n)]
        if _verify(U, V, W, n, m, p) and _is_valid_entries(U, V, W):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave margin for the evaluator

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try structured search first (much more efficient)
    # Theoretical lower bound: n*m + m*p + p*n - m = 6 + 30 + 20 - 3 = 53
    for r in range(53, best_rank):
        if time.time() > deadline - 20:
            break
        result = _structured_search_rank(n, m, p, r, deadline)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    # If structured search didn't find anything, try random search with limited trials
    if best_rank > 53:
        values = [0, 0.5, 1, -0.5, -1]
        for r in range(53, best_rank):
            if time.time() > deadline - 10:
                break
            result = _random_search_rank(n, m, p, r, values, trials=200, deadline=deadline)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = r
                break

    return {"U": U_best, "V": V_best, "W": W_best}
