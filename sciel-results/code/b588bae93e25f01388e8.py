"""Strassen-based block decomposition for 2x5x6 matrix multiplication."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _elapsed(t0):
    return time.time() - t0

def _time_left(t0, deadline):
    return max(0.0, deadline - time.time())

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(x, tol=1e-9):
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _snap_matrix(M, tol=1e-9):
    return np.array([[_snap_to_half_integer(x, tol) for x in row] for row in M])

def _strassen_222():
    """Return (U,V,W) for 2x2x2 with rank 7, using the standard Strassen scheme.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=2.
    """
    n = m = p = 2
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))

    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Term 1: M1
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11

    # Term 2: M2
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10

    # Term 3: M3
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01 - B11
    W[1, 2] = 1  # C01

    # Term 4: M4
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10 - B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10

    # Term 5: M5
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01

    # Term 6: M6
    U[2, 5] = 1; U[0, 5] = -1  # A10 - A00
    V[0, 5] = 1; V[1, 5] = 1  # B00 + B01
    W[3, 5] = 1  # C11

    # Term 7: M7
    U[1, 6] = 1; U[3, 6] = -1  # A01 - A11
    V[2, 6] = 1; V[3, 6] = 1  # B10 + B11
    W[0, 6] = 1  # C00

    return U, V, W

def _build_2x5x6_strassen():
    """Build a rank-48 decomposition for 2x5x6 using Strassen blocks.
    
    Split: A is 2x5 = 2x(2+2+1), B is 5x6 = (2+2+1)x6.
    Use 6 blocks of 2x2x2 (Strassen, rank 7 each) + 3 blocks of 2x1x2 (rank 2 each).
    Total rank = 6*7 + 3*2 = 42 + 6 = 48.
    """
    n, m, p = 2, 5, 6
    rank = 48
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    U22, V22, W22 = _strassen_222()
    col = 0

    # 6 Strassen 2x2x2 blocks
    # A rows: 0,1 (both rows of A)
    # B cols: split 5 into (0,1),(2,3),(4) and 6 into (0,1),(2,3),(4,5)
    # We need 2x5 * 5x6, so we pair A-col-blocks with B-row-blocks.
    # A has 5 columns, B has 5 rows.
    # Block structure:
    #   A col block 0: cols 0,1  -> B row block 0: rows 0,1  -> 2x2x2
    #   A col block 1: cols 2,3  -> B row block 1: rows 2,3  -> 2x2x2
    #   A col block 2: col 4     -> B row block 2: row 4     -> 2x1x1 (but we need 2x1x2 for B cols)
    # Actually, B has 6 columns. Let me re-think.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A is 2x5, B is 5x6, C is 2x6.
    #
    # Split A columns into blocks: (0,1), (2,3), (4)
    # Split B rows into blocks: (0,1), (2,3), (4)
    # Split B columns into blocks: (0,1), (2,3), (4,5)
    #
    # For each (A-col-block, B-row-block) pair of size (a x b), and for each B-col-block of size c:
    #   We need a x b x c multiplication.
    #
    # (2,2)x(2,2) x (2,2) -> 2x2x2, Strassen rank 7
    # (2,2)x(2,2) x (2,2) -> 2x2x2, Strassen rank 7  [two B-col-blocks for each B-row-block]
    # (2,1)x(1,1) x (2,2) -> 2x1x2, rank 2
    #
    # Total: 6 * 2x2x2 (Strassen) + 3 * 2x1x2 = 42 + 6 = 48

    # 2x2x2 blocks: 6 of them
    # For each pair of (A-col-block, B-row-block) that are both size 2:
    #   (A cols 0,1) x (B rows 0,1) x (B cols 0,1) -> Strassen
    #   (A cols 0,1) x (B rows 0,1) x (B cols 2,3) -> Strassen
    #   (A cols 0,1) x (B rows 0,1) x (B cols 4,5) -> Strassen
    #   (A cols 2,3) x (B rows 2,3) x (B cols 0,1) -> Strassen
    #   (A cols 2,3) x (B rows 2,3) x (B cols 2,3) -> Strassen
    #   (A cols 2,3) x (B rows 2,3) x (B cols 4,5) -> Strassen

    strassen_blocks = [
        # (a_col_start, b_row_start, b_col_start)
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (2, 2, 0),
        (2, 2, 2),
        (2, 2, 4),
    ]

    for a_cs, b_rs, b_cs in strassen_blocks:
        for c in range(7):
            # U: A[i, a_cs + j] for i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    u_val = U22[i*2 + j, c]
                    if u_val != 0:
                        U[i * m + (a_cs + j), col] += u_val
            # V: B[b_rs + j, b_cs + k] for j in {0,1}, k in {0,1}
            for j in range(2):
                for k in range(2):
                    v_val = V22[j*2 + k, c]
                    if v_val != 0:
                        V[(b_rs + j) * p + (b_cs + k), col] += v_val
            # W: C[i, b_cs + k] for i in {0,1}, k in {0,1}
            for i in range(2):
                for k in range(2):
                    w_val = W22[k*2 + i, c]
                    if w_val != 0:
                        W[(b_cs + k) * n + i, col] += w_val
        col += 7

    # 2x1x2 blocks: 3 of them
    # A col 4, B row 4, B col blocks (0,1), (2,3), (4,5)
    # 2x1x2: C[i, k] = A[i,4] * B[4,k]
    # Rank 2: for each k-block of size 2, we need 2 terms.
    # Term structure: U[i,4] * V[4, k] * W[k, i]
    # For 2x1x2: rank 2 is minimal.
    # U: [1, 1] for i=0,1 (A[i,4])
    # V: [1, 1] for k=0,1 (B[4,k])
    # W: identity-like: W[0,0]=1, W[1,1]=1 for the 2x1x2 case
    # Actually for 2x1x2: T[i,0,k] = A[i,0]*B[0,k]
    # We need: sum_c U[i,c] * V[k,c] * W[k*n+i, c] = A[i,0]*B[0,k]
    # With rank 2: c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> contributes to (i=0,k=0)
    # c=1: U[1,1]=1, V[1,1]=1, W[3,1]=1 -> contributes to (i=1,k=1)
    # But we need all 4 entries: (0,0),(0,1),(1,0),(1,1)
    # So rank 2 is NOT enough for 2x1x2. Rank 4 is needed (schoolbook).
    # Wait, 2x1x2: n=2, m=1, p=2. T[i,j,k] with j=0 only.
    # T[i,0,k] = A[i,0]*B[0,k]
    # This is an outer product of A[:,0] and B[0,:], which is rank 1 as a matrix but
    # as a tensor T[i,0,k], we need sum_c U[i,0,c] V[0,k,c] W[k*2+i, c] = A[i,0]*B[0,k]
    # With rank 2: c=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1 -> T[0,0,0] += 1
    # c=1: U[0,0,1]=1, V[0,1,1]=1, W[2,0,1]=1 -> T[0,0,1] += 1
    # But we also need T[1,0,0] and T[1,0,1].
    # So we need at least 4 terms (one for each (i,k) pair) unless we can share.
    # Actually, T[i,0,k] = A[i,0]*B[0,k]. If A and B are generic, we need 4 terms.
    # But wait - in our context, A[i,4] and B[4,k] are specific entries, not generic.
    # The tensor T[i,0,k] = A[i,4]*B[4,k] for i in {0,1}, k in {0,1}.
    # This is a rank-1 tensor (outer product), so rank 1 suffices!
    # T[i,0,k] = A[i,4]*B[4,k] = (A[i,4]) * (B[4,k])
    # As a tensor in (i,k) space, this is rank 1.
    # So: U[i,0,c] = A[i,4], V[0,k,c] = B[4,k], W[k,0,c] = 1
    # But we need U, V, W to be constants (not depending on A, B).
    # The decomposition must work for ALL A, B. So U, V, W are fixed.
    # T[i,0,k] = sum_c U[i,0,c] * V[0,k,c] * W[k*2+i, c] must equal A[i,4]*B[4,k] for all A, B.
    # This means: U[i,0,c]*V[0,k,c]*W[k*2+i,c] must form a decomposition of the "multiplication"
    # of the single entry A[i,4] with B[4,k].
    # Since A[i,4] and B[4,k] are independent variables, we need:
    # sum_c U[i,0,c] * V[0,k,c] * W[k*2+i,c] = delta_{i,i'} * delta_{k,k'} ... no.
    # Actually, the tensor for multiplying a 2x1 by 1x2 is:
    # T[i,0,k] = A[i,0] * B[0,k]
    # The structure tensor has T[i,0,k] = 1 for all i,k (since the product is just the entry-wise product).
    # Wait no. The structure tensor T satisfies: C[i,k] = sum_j A[i,j] B[j,k]
    # For 2x1x2: C[i,k] = A[i,0] * B[0,k]
    # The structure tensor T[i,0,k] = 1 for all i in {0,1}, k in {0,1}.
    # Because C[i,k] = sum_c U[i,0,c] V[0,k,c] W[k*2+i, c] must equal A[i,0]*B[0,k] for all A, B.
    # This is only possible if the tensor T[i,0,k] = 1 for all i,k, which means we need
    # sum_c U[i,0,c] V[0,k,c] W[k*2+i, c] = 1 for all i,k.
    # This is a 2x2 matrix of 1s, which has rank 4 as a tensor (since it's the all-ones tensor).
    # Actually, the all-ones 2x1x2 tensor: T[i,0,k] = 1. This can be written as
    # sum_c u[i,c] v[k,c] w[k,i] with rank... The all-ones 2x2 matrix has rank 1, but as a
    # 3-tensor T[i,0,k] with the middle index fixed, it's just a 2x2 matrix of 1s.
    # The tensor rank of the all-ones 2x2 matrix (viewed as a 3-tensor with one index fixed)
    # is 4 (you need 4 terms to get all 4 entries to 1 with the constraint that each term
    # is a product of a row vector, a column vector, and a matrix entry).
    # Actually no. Let me think again.
    # T[i,0,k] = 1 for i,k in {0,1}.
    # We need: sum_c U[i,c] * V[k,c] * W[k*2+i, c] = 1 for all i,k.
    # With c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> T[0,0,0] += 1
    # c=1: U[0,0]=1, V[1,1]=1, W[2,0,1]=1 -> T[0,0,1] += 1  (but U[0,0] already used)
    # We can't reuse U[0,0] for multiple k values in the same term.
    # Each term c contributes U[i,c]*V[k,c]*W[k*2+i,c] to T[i,0,k].
    # For a fixed c, this is a rank-1 matrix in (i,k): U[:,c] * V[:,c]^T * W_vec[c]
    # where W_vec[c] is the vector W[:,c] reshaped to (2,2) and we take the (k,i) entry.
    # Actually W[k*2+i, c] for fixed c gives a 2x2 matrix Wmat[c] with Wmat[c][k,i] = W[k*2+i, c].
    # So term c contributes: U[:,c] * V[:,c]^T * Wmat[c] (Hadamard product? No.)
    # T[i,0,k] += U[i,c] * V[k,c] * W[k*2+i, c]
    # This is: (U[:,c])_i * (V[:,c])_k * (Wmat[c])_{k,i}
    # = (U[:,c] * V[:,c]^T) Hadamard Wmat[c]  ... no, it's element-wise product of three.
    # For the all-ones tensor, we need sum_c U[i,c] V[k,c] W[k*2+i, c] = 1.
    # With rank 4 (schoolbook): each term handles one (i,k) pair.
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> (0,0)
    # c=1: U[0,1]=1, V[1,1]=1, W[2,0,1]=1 -> (0,1)
    # c=2: U[1,2]=1, V[0,2]=1, W[0,1,2]=1 -> (1,0)
    # c=3: U[1,3]=1, V[1,3]=1, W[3,1,3]=1 -> (1,1)
    # So rank 4 for each 2x1x2 block. Total: 6*7 + 3*4 = 42 + 12 = 54.

    # Let me just use schoolbook for the 2x1x2 blocks.
    for b_cs in [0, 2, 4]:
        for i in range(2):
            for k in range(2):
                U[i * m + 4, col] = 1
                V[4 * p + (b_cs + k), col] = 1
                W[(b_cs + k) * n + i, col] = 1
                col += 1

    return U, V, W, col

def _refine(U, V, W, n, m, p, t0, deadline, max_iter=2000):
    """Local refinement to reduce rank by zeroing out columns that are nearly redundant."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()

    def compute_error():
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err

    err = compute_error()
    if err > 1e-6:
        return None, None, None, err

    # Try to zero out columns one at a time
    for c in range(rank):
        if _time_left(t0, deadline) < 5:
            break
        # Save
        U_save = U[:, c].copy()
        V_save = V[:, c].copy()
        W_save = W[:, c].copy()

        # Zero out
        U[:, c] = 0
        V[:, c] = 0
        W[:, c] = 0

        # Check if still valid
        err_new = compute_error()
        if err_new < 1e-8:
            rank -= 1
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
            c = max(0, c - 1)  # recheck same index
        else:
            # Restore
            U[:, c] = U_save
            V[:, c] = V_save
            W[:, c] = W_save

    return U, V, W, compute_error()

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Build Strassen-based decomposition
    U, V, W, rank = _build_2x5x6_strassen()

    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-6):
        # Fall back to schoolbook
        U, V, W = schoolbook()
        rank = n * m * p
    else:
        # Try to refine (reduce rank)
        U_r, V_r, W_r, err = _refine(U, V, W, n, m, p, t0, deadline)
        if U_r is not None and err < 1e-8:
            U, V, W = U_r, V_r, W_r
            rank = U.shape[1]

    # Snap to half-integers
    U_snap = _snap_matrix(U)
    V_snap = _snap_matrix(V)
    W_snap = _snap_matrix(W)

    # Verify snapped version
    if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
        U, V, W = U_snap, V_snap, W_snap

    # Final verification
    if not _verify(U, V, W, n, m, p, tol=1e-9):
        # Fall back to schoolbook
        U, V, W = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
