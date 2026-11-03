import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 145.0
    n, m, p = 3, 4, 6

    # Rank-66 exact tensor decomposition for 3x4 by 4x6 matrix multiplication.
    # Strategy:
    #   A (3x4): rows split 1+2, cols split 2+2
    #   B (4x6): rows split 2+2, cols split 2+2+2
    #   C (3x6): each block row uses two column blocks of A times two row blocks of B
    #
    # Block ranks:
    #   1x2 * 2x2  -> rank 4 (schoolbook)
    #   2x2 * 2x2  -> rank 7 (Strassen)
    #
    # Total: 6 * (rank 4) + 6 * (rank 7) = 24 + 42 = 66

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)
    col = 0

    # Indexing:
    #   A_{i,k}  -> U row  i * m + k       (i in [0,n), k in [0,m))
    #   B_{k,j}  -> V row  k * p + j       (k in [0,m), j in [0,p))
    #   C_{i,j}  -> W row  j * n + i       (j in [0,p), i in [0,n))  (trace convention)

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook).
        A[ar:ar+1, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr, cc], C[cr, cc+1]
        """
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        # M1 = A0 * B00 -> C0
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        # M2 = A1 * B10 -> C0
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        # M3 = A0 * B01 -> C1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        # M4 = A1 * B11 -> C1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen).
        A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr:cr+2, cc:cc+2]
        Standard Strassen:
          M1 = (A00+A11)(B00+B11)   -> contributes to C00, C11
          M2 = (A10+A11)B00         -> C10 += M2, C11 -= M2
          M3 = A00(B01-B11)         -> C01 += M3, C11 += M3
          M4 = A11(B10-B00)         -> C00 += M4, C10 += M4
          M5 = (A00+A01)B11         -> C00 -= M5, C01 += M5
          M6 = (A10-A00)(B00+B01)   -> C11 += M6
          M7 = (A01-A11)(B10+B11)   -> C00 += M7
        """
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # ---- Build the 66-term decomposition ----
    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # --- Row 0 of C (A row 0, 1x2 blocks) ---
        # A[0, 0:2] * B[0:2, bc:bc+2]  (1x2 * 2x2, rank 4)
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        # A[0, 2:4] * B[2:4, bc:bc+2]  (1x2 * 2x2, rank 4)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # --- Rows 1-2 of C (A rows 1:3, 2x2 blocks) ---
        # A[1:3, 0:2] * B[0:2, bc:bc+2]  (2x2 * 2x2, rank 7, Strassen)
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        # A[1:3, 2:4] * B[2:4, bc:bc+2]  (2x2 * 2x2, rank 7, Strassen)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # Safety check: verify no term exceeded the allocated rank
    for r in range(n * m):
        for c in range(col):
            assert abs(U[r][c]) <= 1, f"U[{r}][{c}] = {U[r][c]}"
    for r in range(m * p):
        for c in range(col):
            assert abs(V[r][c]) <= 1, f"V[{r}][{c}] = {V[r][c]}"
    for r in range(p * n):
        for c in range(col):
            assert abs(W[r][c]) <= 1, f"W[{r}][{c}] = {W[r][c]}"

    # Attempt to prune redundant columns via linear dependency detection.
    # Convert to numpy arrays for efficient linear algebra.
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The tensor decomposition condition is:
    # sum_r U[i,r] * V[k,r] * W[j,r] = delta(i, k) * A[i,k] * B[k,j] ... 
    # Actually, the tensor T_{i,k,j} = delta(i,k) is not quite right.
    # The multiplication tensor T_{i,k,j} is 1 if i is row of A, k is col of A/row of B, j is col of B.
    # Specifically, C[i,j] = sum_k A[i,k] B[k,j].
    # The tensor is M_{i,k,j} = 1 if valid indices, else 0.
    # The decomposition is sum_r U[i,r] V[k,r] W[j,r] = M_{i,k,j}.
    
    # We can check if any column r is linearly dependent on the others by checking
    # if the vector v_r = (U[:,r], V[:,r], W[:,r]) is in the span of the other columns.
    # However, the tensor condition is a product, not a sum of vectors.
    # A better approach: check if removing column r still allows the tensor to be represented.
    # This is equivalent to checking if the column space of the "flattened" tensor matrices
    # still spans the target tensor.
    
    # Let's try a simpler heuristic: check if any column has all zeros in one of U, V, or W.
    # If so, it contributes nothing and can be removed.
    # In our construction, all columns have non-zero entries in all three matrices.
    # So this simple check won't find anything.
    
    # A more sophisticated check: try to solve for coefficients that express one column as a linear combination of others.
    # This is computationally expensive for 66 columns.
    # Given the time constraint, we'll skip this and return the rank-66 solution.
    
    # Check time before returning
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
