"""Matrix multiplication 4x4 x 4x6 tensor decomposition.

Strategy: build a valid rank-84 decomposition via block-Strassen (2x2 blocks,
rank-7 Strassen per 2x2*2x2 pair), then attempt to reduce rank with a fast
refinement pass. Fallback is the guaranteed-valid rank-84 (or schoolbook).
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _build_target(n, m, p):
    """T[(i*m+j), (j*p+k), (k*n+i)] = 1  (official trace convention)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _block_strassen_decomp(n, m, p, deadline):
    """
    Build a valid decomposition using 2x2 blocks and Strassen (rank 7) for each
    2x2*2x2 multiplication.

    A is 2x2 blocks of 2x2 (n_blocks x m_blocks).
    B is 2x3 blocks of 2x2 (m_blocks x p_blocks).
    C is 2x3 blocks of 2x2 (n_blocks x p_blocks).

    For each output block C[a,c] = A[a,0]*B[0,c] + A[a,1]*B[1,c],
    we use Strassen on the 2x2*2x2 multiplications, sharing the Strassen
    intermediates where possible.

    Actually, the cleanest correct approach: for each (a, c) output block,
    we need to compute A[a,0]*B[0,c] + A[a,1]*B[1,c]. This is a sum of two
    2x2*2x2 products. We can use Strassen for each product (7 terms each) = 14
    terms per output block. With 2*3=6 output blocks, total = 84.

    Strassen for 2x2*2x2:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
      C00 = M1+M4-M5+M7
      C01 = M3+M5
      C10 = M2+M4
      C11 = M1-M2+M3+M6
    """
    bn = 2
    n_blocks = n // bn  # 2
    m_blocks = m // bn  # 2
    p_blocks = p // bn  # 3

    # Strassen terms: each is (U_pattern, V_pattern, W_pattern)
    # U_pattern: dict from (row,col) in 2x2 A-block to coefficient
    # V_pattern: dict from (row,col) in 2x2 B-block to coefficient
    # W_pattern: dict from (row,col) in 2x2 C-block to coefficient
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00 -> contributes to C10(+1), C11(-1)
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11) -> contributes to C01(+1), C11(+1)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00) -> contributes to C00(+1), C10(+1)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11 -> contributes to C00(-1), C01(+1)
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01) -> contributes to C11(+1)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11) -> contributes to C00(+1)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Total rank: n_blocks * p_blocks * m_blocks * 7
    # For each output block (a,c) and each m-block (b), we use 7 Strassen terms
    total_rank = n_blocks * p_blocks * m_blocks * 7  # 2*3*2*7 = 84

    U_data = np.zeros((n * m, total_rank))
    V_data = np.zeros((m * p, total_rank))
    W_data = np.zeros((p * n, total_rank))

    col = 0
    for a in range(n_blocks):
        for c in range(p_blocks):
            for b in range(m_blocks):
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    # A block (a,b): rows a*2..a*2+1, cols b*2..b*2+1
                    for (ri, ci), coef in u_pat.items():
                        row = a * bn + ri
                        col_a = b * bn + ci
                        U_data[row * m + col_a, col] += coef
                    # B block (b,c): rows b*2..b*2+1, cols c*2..c*2+1
                    for (ri, ci), coef in v_pat.items():
                        row = b * bn + ri
                        col_b = c * bn + ci
                        V_data[row * p + col_b, col] += coef
                    # C block (a,c): rows a*2..a*2+1, cols c*2..c*2+1
                    for (ri, ci), coef in w_pat.items():
                        row = a * bn + ri
                        col_c = c * bn + ci
                        W_data[col_c * n + row, col] += coef
                    col += 1

    # Verify
    T = _build_target(n, m, p)
    recon = np.einsum('ar,br,cr->abc', U_data, V_data, W_data)
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback to schoolbook
        return None

    return U_data, V_data, W_data


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _refine(U, V, W, T, deadline, target_rank):
    """
    Try to reduce rank by zeroing out columns and re-solving.
    Returns (U, V, W, rank) of best found, or None if no improvement.
    """
    n, m, p = T.shape[0] // m, T.shape[1] // p, T.shape[2] // n
    current_rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy(), current_rank)

    # Try to reduce by attempting to zero out columns
    # For each column, check if it can be set to zero while maintaining validity
    # by adjusting other columns (this is complex, so use a simpler heuristic)

    # Alternative: try rank reduction by solving a least-squares problem
    # with a sparsity constraint on the columns
    for attempt_rank in range(target_rank, current_rank - 1, -1):
        if time.time() > deadline - 5:
            break
        # Try to find a rank-attempt_rank decomposition
        # Start from current and try to zero out (current_rank - attempt_rank) columns
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()

        # Randomly select columns to zero out
        cols_to_zero = np.random.choice(current_rank, current_rank - attempt_rank, replace=False)
        for c in cols_to_zero:
            U_try[:, c] = 0
            V_try[:, c] = 0
            W_try[:, c] = 0

        # Coordinate descent to fix the decomposition
        err = np.sum((T - np.einsum('ar,br,cr->abc', U_try, V_try, W_try))**2)
        for it in range(50):
            if time.time() > deadline - 4:
                break
            improved = False
            for c in range(attempt_rank):
                if c in cols_to_zero:
                    continue
                Uo, Vo, Wo = U_try[:,c].copy(), V_try[:,c].copy(), W_try[:,c].copy()
                U_try[:,c] += np.random.randn(n*m)*0.01
                V_try[:,c] += np.random.randn(m*p)*0.01
                W_try[:,c] += np.random.randn(p*n)*0.01
                new_err = np.sum((T - np.einsum('ar,br,cr->abc', U_try, V_try, W_try))**2)
                if new_err < err:
                    err = new_err
                    improved = True
                else:
                    U_try[:,c], V_try[:,c], W_try[:,c] = Uo, Vo, Wo
            if not improved or err < 1e-10:
                break

        if err < 1e-10:
            # Verify and return
            recon = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
            if np.allclose(recon, T, atol=1e-9):
                best = (U_try, V_try, W_try, attempt_rank)
                break

    return best


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit

    T = _build_target(n, m, p)

    # Step 1: Build the rank-84 block-Strassen decomposition
    result = _block_strassen_decomp(n, m, p, deadline)
    if result is not None:
        U, V, W = result
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)

    current_rank = U.shape[1]

    # Step 2: Try to refine to a lower rank
    if time.time() < deadline - 20 and current_rank > 80:
        refined = _refine(U, V, W, T, deadline, target_rank=max(60, current_rank - 12))
        if refined is not None:
            U, V, W, new_rank = refined
            if new_rank < current_rank:
                current_rank = new_rank

    # Final verification
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        # Last resort: schoolbook
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
