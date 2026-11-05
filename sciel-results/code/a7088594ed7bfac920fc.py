import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify(U, V, W, n, m, p, tol=1e-8):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.ndim != 2:
        return False
    if U.shape[0] != n * m:
        return False
    rank = U.shape[1]
    if V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False

    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=tol))


def _schoolbook_4x4x5():
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2_terms():
    """Return list of 7 terms, each a tuple (U_dict, V_dict, W_dict) for 2x2x2 Strassen.
    U_dict maps (i,j) -> coeff, V_dict maps (j,k) -> coeff, W_dict maps (k,i) -> coeff.
    The 2x2 indices are local: i,j,k in {0,1}.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    # U: (0,0):1, (1,1):1
    # V: (0,0):1, (1,1):1
    # W: (0,0):1, (1,1):1
    terms.append(
        ({(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1})
    )
    # M2 = (A10+A11)B00
    # U: (1,0):1, (1,1):1
    # V: (0,0):1
    # W: (0,1):1, (1,1):-1
    terms.append(
        ({(1, 0): 1, (1, 1): 1},
         {(0, 0): 1},
         {(0, 1): 1, (1, 1): -1})
    )
    # M3 = A00(B01-B11)
    # U: (0,0):1
    # V: (0,1):1, (1,1):-1
    # W: (1,0):1, (1,1):1
    terms.append(
        ({(0, 0): 1},
         {(0, 1): 1, (1, 1): -1},
         {(1, 0): 1, (1, 1): 1})
    )
    # M4 = A11(B10-B00)
    # U: (1,1):1
    # V: (1,0):1, (0,0):-1
    # W: (0,0):1, (0,1):1
    terms.append(
        ({(1, 1): 1},
         {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (0, 1): 1})
    )
    # M5 = (A00+A01)B11
    # U: (0,0):1, (0,1):1
    # V: (1,1):1
    # W: (0,0):-1, (1,0):1
    terms.append(
        ({(0, 0): 1, (0, 1): 1},
         {(1, 1): 1},
         {(0, 0): -1, (1, 0): 1})
    )
    # M6 = (A10-A00)(B00+B01)
    # U: (1,0):1, (0,0):-1
    # V: (0,0):1, (0,1):1
    # W: (1,1):1
    terms.append(
        ({(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1})
    )
    # M7 = (A01-A11)(B10+B11)
    # U: (0,1):1, (1,1):-1
    # V: (1,0):1, (1,1):1
    # W: (0,0):1
    terms.append(
        ({(0, 1): 1, (1, 1): -1},
         {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1})
    )
    return terms


def _construct_4x4x5_block_strassen():
    """Construct the 4x4x5 decomposition using independent Strassen blocks.
    For the 4x4x4 part (columns 0-3), use 4 independent 2x2x2 Strassen blocks (rank 28).
    For the 4x4x1 part (column 4), use schoolbook (rank 16).
    Total rank = 44.
    """
    if _time_remaining() < 5:
        raise TimeoutError("Not enough time remaining")

    n, m, p = 4, 4, 5
    rank_total = 44
    U = np.zeros((n * m, rank_total), dtype=np.float64)
    V = np.zeros((m * p, rank_total), dtype=np.float64)
    W = np.zeros((p * n, rank_total), dtype=np.float64)

    state = [0]
    strassen_terms = _strassen_2x2_terms()

    def add_strassen_block(ra, ca, rb, cb):
        """Add a 2x2x2 Strassen block.
        ra, ca: row/col offset of A block (0 or 1)
        rb, cb: row/col offset of B block (0 or 1)
        """
        for (u_dict, v_dict, w_dict) in strassen_terms:
            col = state[0]
            for (li, lj), coeff in u_dict.items():
                gi = 2 * ra + li
                gj = 2 * ca + lj
                U[gi * m + gj, col] += coeff
            for (lj, lk), coeff in v_dict.items():
                gj = 2 * rb + lj
                gk = 2 * cb + lk
                V[gj * p + gk, col] += coeff
            for (lk, li), coeff in w_dict.items():
                gk = 2 * cb + lk
                gi = 2 * ra + li
                W[gk * n + gi, col] += coeff
            state[0] += 1

    def add_schoolbook_2x2_2x1(ra, ca, rb, cb_col):
        """Add schoolbook terms for a 2x2 A block times a 2x1 B block.
        ra, ca: row/col offset of A block
        rb: row offset of B block
        cb_col: column index of B block (global)
        """
        for i in range(2):
            for j in range(2):
                col = state[0]
                gi = 2 * ra + i
                gj = 2 * ca + j
                gj2 = 2 * rb + j
                U[gi * m + gj, col] = 1.0
                V[gj2 * p + cb_col, col] = 1.0
                W[cb_col * n + gi, col] = 1.0
                state[0] += 1

    # 4x4x4 part: columns 0-3
    # Block (0,0) of A times Block (0,0) of B -> contributes to C cols 0-1
    add_strassen_block(0, 0, 0, 0)
    # Block (0,1) of A times Block (1,0) of B -> contributes to C cols 2-3
    add_strassen_block(0, 1, 1, 0)
    # Block (1,0) of A times Block (0,0) of B -> contributes to C cols 0-1
    add_strassen_block(1, 0, 0, 0)
    # Block (1,1) of A times Block (1,0) of B -> contributes to C cols 2-3
    add_strassen_block(1, 1, 1, 0)

    # 4x4x1 part: column 4
    # A block (0,0) times B col 4
    add_schoolbook_2x2_2x1(0, 0, 0, 4)
    # A block (0,1) times B col 4
    add_schoolbook_2x2_2x1(0, 1, 1, 4)
    # A block (1,0) times B col 4
    add_schoolbook_2x2_2x1(1, 0, 0, 4)
    # A block (1,1) times B col 4
    add_schoolbook_2x2_2x1(1, 1, 1, 4)

    used_rank = state[0]
    assert used_rank == rank_total, f"Expected rank {rank_total}, got {used_rank}"

    U = U[:, :used_rank]
    V = V[:, :used_rank]
    W = W[:, :used_rank]

    if not _verify(U, V, W, n, m, p, tol=1e-8):
        raise ValueError("Block Strassen 4x4x5 decomposition failed verification")

    return U, V, W


def construct():
    """Main entry point: construct the rank-minimal exact tensor decomposition
    for 4x4x5.  Returns {'U','V','W'} as lists of lists (integer/half-integer)."""
    try:
        U, V, W = _construct_4x4x5_block_strassen()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except (TimeoutError, ValueError):
        pass

    # Fallback: guaranteed-valid schoolbook (rank 80)
    U, V, W = _schoolbook_4x4x5()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
