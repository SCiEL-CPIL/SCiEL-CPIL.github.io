"""Tensor decomposition search for 2x4x4 matrix multiplication.

Strategy:
  1. Immediately produce a known rank-28 construction (Strassen on 2x2 blocks).
  2. Spend the remaining wall-clock budget on exact random searches (rank 27, then 26)
     to try to find a lower-rank exact decomposition.
  3. Return the best valid result found before a safety margin before the deadline.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]
N, M, P = 2, 4, 4
NM, MP, PN = N * M, M * P, P * N  # 8, 16, 8

# Deadline guard: evaluator kills at 150s; keep a safety margin.
DEADLINE = 145.0


def _tensor_entries():
    """Yield (i, j, k) triples for the multiplication tensor (one per entry)."""
    for i in range(N):
        for j in range(M):
            for k in range(P):
                yield (i, j, k)


def _check(U, V, W, rank, tol=1e-7):
    """Return True iff sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for every (i,j,k)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for c in range(rank):
                    s += U[i * M + j, c] * V[j * P + k, c] * W[k * N + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _check_fast(U, V, W, rank, tol=1e-7):
    """Vectorized check using matrix products."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U: (NM, rank), V: (MP, rank), W: (PN, rank)
    # For each c: outer(U[:,c], V[:,c]) -> (NM, MP); contract with W[:,c] -> (NM, MP, PN)
    # We only need entries where U index = i*M+j, V index = j*P+k, W index = k*N+i.
    # Build via einsum: result[i,j,k] = sum_c U[iM+j,c] V[jP+k,c] W[kN+i,c]
    # Reshape U to (N,M,rank), V to (M,P,rank), W to (P,N,rank)
    Ur = U.reshape(N, M, rank)
    Vr = V.reshape(M, P, rank)
    Wr = W.reshape(P, N, rank)
    # sum over c: Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    # einsum 'ijc,jkc,kic->ijk'
    res = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(res, 1.0, atol=tol)


def _strassen_rank28():
    """Construct a known rank-28 decomposition via Strassen on 2x2 sub-blocks.

    Decompose A (2x4) and B (4x4) into 2x2 blocks:
      A = [[A11, A12], [A21, A22]]  each Ai is 1x2
      B = [[B11, B12], [B21, B22]]  each Bj is 2x2
    Output C (2x4) = [[C11, C12], [C21, C22]] each Ci is 1x2.

    Using Strassen's 7 products for 2x2 matrix multiplication, but adapted to
    the rectangular case. We treat the problem as multiplying a 2x2 "block matrix"
    of 1x2 by 2x2 blocks.

    Actually, a cleaner approach: split the 4 columns into two groups of 2.
    A (2x4) = [A_left (2x2), A_right (2x2)]
    B (4x4) = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
    C (2x4) = [C_left (2x2), C_right (2x2)]
    C_left = A_left * B11 + A_right * B21
    C_right = A_left * B12 + A_right * B22

    This is 2 independent 2x2x2 multiplications, each of rank 7 (Strassen).
    Total rank = 14. But wait, that's for 2x2 by 2x2. Let me reconsider.

    Actually, the standard Strassen gives rank 7 for 2x2 matrix multiplication.
    For 2x4 by 4x4, we can view it as:
      A is 2x4, B is 4x4.
      Split B into two 4x2 halves: B = [B_L (4x2), B_R (4x2)]
      C = [C_L (2x2), C_R (2x2)]
      C_L = A * B_L, C_R = A * B_R

    Each of A (2x4) times B_half (4x2) is a 2x4 x 4x2 multiplication.
    That's not directly Strassen.

    Let me use a different decomposition. Split both A and B:
      A (2x4) = [A1 (2x2), A2 (2x2)]
      B (4x4) = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
      C (2x4) = [C1 (2x2), C2 (2x2)]
      C1 = A1*B11 + A2*B21
      C2 = A1*B12 + A2*B22

    C1 and C2 each require 2x2 by 2x2 multiplication. Using Strassen for each:
      C1 needs 7 products, C2 needs 7 products. Total = 14.

    But we can share some structure. Actually, let's just do 14 rank-1 terms
    using Strassen for each half.

    Strassen for 2x2: given X (2x2), Y (2x2), compute Z = X*Y with 7 products:
      m1 = (x11+x22)(y11+y22)
      m2 = (x21+x22)*y11
      m3 = x11*(y12-y22)
      m4 = x22*(y21-y11)
      m5 = (x11+x12)*y22
      m6 = (x21-x11)*(y11+y12)
      m7 = (x12-x22)*(y21+y22)
      z11 = m1 + m4 - m5 + m7
      z12 = m3 + m5
      z21 = m2 + m4
      z22 = m1 - m2 + m3 + m6

    For C1 = A1*B11 + A2*B21:
      This is (A1, A2) * [[B11], [B21]] where (A1,A2) is 2x4 and [[B11],[B21]] is 4x2.
      We can write this as a 2x2 "block" multiplication:
        X = [[A1, A2]] (1x2 block row, each 2x2) -- no, this is getting complicated.

    Let me just use the simple approach: 4 independent 2x2 multiplications,
    each with Strassen rank 7, total rank 28.

    A (2x4) = [A1 (2x2), A2 (2x2)]
    B (4x4) = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
    C (2x4) = [C1 (2x2), C2 (2x2)]
    C1 = A1*B11 + A2*B21
    C2 = A1*B12 + A2*B22

    For C1: it's a 2x4 times 4x2. We can decompose:
      C1 = A1*B11 + A2*B21
      Use Strassen for A1*B11 (7 terms) and A2*B21 (7 terms). Total 14 for C1.
    Similarly 14 for C2. Total 28.
    """
    # Strassen factors for 2x2 multiplication.
    # U entries: for product m, U[i,j] where i is row of X (0,1), j is col of X (0,1)
    # V entries: for product m, V[i,j] where i is row of Y (0,1), j is col of Y (0,1)
    # W entries: for product m, W[i,j] where i is row of Z (0,1), j is col of Z (0,1)
    #
    # Strassen products (X is 2x2, Y is 2x2, Z = X*Y):
    # m1: X[[0,0],[1,1]] * Y[[0,0],[1,1]] -> Z[[0,0],[1,1]]
    # m2: X[[1,0],[1,1]] * Y[[0,0],[0,0]] -> Z[[1,0],[1,0]]
    # m3: X[[0,0],[0,0]] * Y[[0,1],[1,1]] -> Z[[0,1],[1,1]]
    # m4: X[[1,1],[1,1]] * Y[[1,0],[0,0]] -> Z[[1,0],[0,0]]
    # m5: X[[0,0],[0,1]] * Y[[1,1],[1,1]] -> Z[[0,1],[0,1]]
    # m6: X[[1,0],[0,0]] * Y[[0,0],[0,1]] -> Z[[1,1],[0,1]]
    # m7: X[[0,1],[1,1]] * Y[[1,0],[1,1]] -> Z[[0,0],[1,1]]

    # Let me just hardcode the 28-term construction directly.
    # For each of the 4 sub-multiplications (A1*B11, A2*B21, A1*B12, A2*B22),
    # use 7 Strassen terms.

    # Sub-multiplication indices:
    # (a_idx, b_idx, c_idx) where:
    #   a_idx: 0 -> A1 (cols 0-1 of A), 1 -> A2 (cols 2-3 of A)
    #   b_idx: 0 -> B11 (rows 0-1, cols 0-1), 1 -> B21 (rows 2-3, cols 0-1)
    #           2 -> B12 (rows 0-1, cols 2-3), 3 -> B22 (rows 2-3, cols 2-3)
    #   c_idx: 0 -> C1 (cols 0-1 of C), 1 -> C2 (cols 2-3 of C)

    # Strassen factor patterns for 2x2:
    # (u_pattern, v_pattern, w_pattern) where each is a dict mapping (row,col) to coeff
    strassen_terms = [
        # m1: (x11+x22)(y11+y22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        # m2: (x21+x22)*y11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # m3: x11*(y12-y22)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # m4: x22*(y21-y11)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):-1}),
        # m5: (x11+x12)*y22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        # m6: (x21-x11)*(y11+y12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1, (0,1):-1}),
        # m7: (x12-x22)*(y21+y22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]

    rank = 28
    U = [[0.0] * rank for _ in range(NM)]
    V = [[0.0] * rank for _ in range(MP)]
    W = [[0.0] * rank for _ in range(PN)]

    col = 0
    # 4 sub-multiplications, each 7 Strassen terms
    sub_mults = [
        (0, 0, 0),  # A1 * B11 -> C1
        (1, 1, 0),  # A2 * B21 -> C1
        (0, 2, 1),  # A1 * B12 -> C2
        (1, 3, 1),  # A2 * B22 -> C2
    ]

    for a_idx, b_idx, c_idx in sub_mults:
        for (u_pat, v_pat, w_pat) in strassen_terms:
            # U: A has shape 2x4. A1 is rows 0-1, cols 0-1. A2 is rows 0-1, cols 2-3.
            # U index: i*M + j where i in [0,1], j in [0,3]
            for (ri, ci), coeff in u_pat.items():
                # ri is row within 2x2 block, ci is col within 2x2 block
                actual_j = ci if a_idx == 0 else ci + 2
                U[ri * M + actual_j][col] += coeff
            # V: B has shape 4x4.
            # B11: rows 0-1, cols 0-1. B21: rows 2-3, cols 0-1.
            # B12: rows 0-1, cols 2-3. B22: rows 2-3, cols 2-3.
            for (ri, ci), coeff in v_pat.items():
                if b_idx == 0:   # B11
                    actual_r, actual_c = ri, ci
                elif b_idx == 1: # B21
                    actual_r, actual_c = ri + 2, ci
                elif b_idx == 2: # B12
                    actual_r, actual_c = ri, ci + 2
                else:            # B22
                    actual_r, actual_c = ri + 2, ci + 2
                V[actual_r * P + actual_c][col] += coeff
            # W: C has shape 2x4. C1 is cols 0-1, C2 is cols 2-3.
            # W index: k*N + i where k in [0,3], i in [0,1]
            # For 2x2 output block: k is row within block, i is col within block
            for (ri, ci), coeff in w_pat.items():
                # ri is row within 2x2 output (maps to k), ci is col within 2x2 output (maps to i)
                actual_k = ri if c_idx == 0 else ri + 2
                actual_i = ci
                W[actual_k * N + actual_i][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}


def _random_search(rank, time_limit, seed):
    """Try to find a rank-`rank` exact decomposition via random search.

    Returns (U, V, W) as lists if found, else None.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    start = time.time()

    # Random initialization with small integers/halves
    def rand_entry():
        return rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2])

    for attempt in range(100000):
        if time.time() - start > time_limit:
            break

        U = [[rand_entry() for _ in range(rank)] for _ in range(NM)]
        V = [[rand_entry() for _ in range(rank)] for _ in range(MP)]
        W = [[rand_entry() for _ in range(rank)] for _ in range(PN)]

        if _check_fast(U, V, W, rank, tol=1e-9):
            return {"U": U, "V": V, "W": W}

    return None


def construct():
    t0 = time.time()

    # 1. Get the known rank-28 solution immediately.
    best = _strassen_rank28()
    best_rank = 28

    # 2. Try random search for rank 27 with a time budget.
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        budget27 = min(remaining - 3, 30)  # spend up to 30s on rank 27
        if budget27 > 1:
            result27 = _random_search(27, budget27, seed=42)
            if result27 is not None:
                best = result27
                best_rank = 27

    # 3. If time remains, try rank 26.
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5 and best_rank > 26:
        budget26 = min(remaining - 3, 20)
        if budget26 > 1:
            result26 = _random_search(26, budget26, seed=123)
            if result26 is not None:
                best = result26
                best_rank = 26

    # Safety: ensure we're before the deadline.
    if time.time() - t0 > DEADLINE - 2:
        pass  # just return what we have

    return best


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
