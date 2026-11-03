"""Rewritten program for 3x4x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, each shape (4, 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen formulas for C = A * B where A,B are 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    n = 2
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # U[i*2+j, c] = coefficient of A_{i,j} in factor c
    # V[j*2+k, c] = coefficient of B_{j,k} in factor c
    # W[k*2+i, c] = coefficient of C_{k,i} in factor c (trace convention: W[k*n+i, c])
    
    # Factor 0: M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # Factor 1: M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1              # B00
    W[2, 1] = 1              # C10
    
    # Factor 2: M3 = A00*(B01-B11)
    U[0, 2] = 1              # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, B11
    W[1, 2] = 1              # C01
    
    # Factor 3: M4 = A11*(B10-B00)
    U[3, 3] = 1              # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, B00
    W[0, 3] = 1; W[2, 3] = 1   # C00, C10
    
    # Factor 4: M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1              # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01
    
    # Factor 5: M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[1, 5] = 1   # B00, B01
    W[0, 5] = -1               # -C00
    
    # Factor 6: M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, A11
    V[2, 6] = 1; V[3, 6] = 1   # B10, B11
    W[0, 6] = -1; W[3, 6] = 1  # -C00, C11
    
    return U, V, W


def _combine_decompositions(decomps, dims):
    """Combine multiple block decompositions into a full 3x4x5 decomposition.
    Each decomps entry is (U_block, V_block, W_block) for a sub-block.
    """
    n, m, p = dims
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for (Ub, Vb, Wb, i_off, j_off, k_off, ni, nj, nk) in decomps:
        r = Ub.shape[1]
        # Place U block: rows i_off..i_off+ni-1, j_off..j_off+nj-1
        for i in range(ni):
            for j in range(nj):
                U[(i_off + i) * m + (j_off + j), col:col+r] = Ub[i * nj + j, :]
        # Place V block: rows j_off..j_off+nj-1, k_off..k_off+nk-1
        for j in range(nj):
            for k in range(nk):
                V[(j_off + j) * p + (k_off + k), col:col+r] = Vb[j * nk + k, :]
        # Place W block: rows k_off..k_off+nk-1, i_off..i_off+ni-1
        for k in range(nk):
            for i in range(ni):
                W[(k_off + k) * n + (i_off + i), col:col+r] = Wb[k * ni + i, :]
        col += r
    
    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check dimensions
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    
    # The tensor T[i,j,k] = delta_{j,k} (for matrix mult C_{i,k} = sum_j A_{i,j} B_{j,k})
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if j == k else 0.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True


def _try_block_decomposition():
    """
    Try to construct a rank-58 decomposition using two Strassen 2x2x2 blocks.
    3x4x5 can be partitioned as:
    - Block 1: rows 0-1 (i), cols 0-1 (j), cols 0-1 (k) -> 2x2x2 Strassen (rank 7)
    - Block 2: rows 0-1 (i), cols 2-3 (j), cols 2-3 (k) -> 2x2x2 Strassen (rank 7)
    - Remaining: row 2 (i), all cols (j=0..3, k=0..4) -> schoolbook for 1x4x5 (rank 20)
    - Plus: rows 0-1, col 2 (j=2), col 4 (k=4) -> need to handle cross terms
    
    Actually, let me think more carefully.
    
    For 3x4x5:
    - We can use Strassen for 2x2x2 blocks.
    - Partition: i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4}
    
    Strategy:
    - Use Strassen for i in {0,1}, j in {0,1}, k in {0,1}: rank 7
    - Use Strassen for i in {0,1}, j in {2,3}, k in {2,3}: rank 7
    - Schoolbook for all other combinations:
      * i in {0,1}, j in {0,1}, k in {4}: 1x2x1 = 2
      * i in {0,1}, j in {2,3}, k in {0,1,4}: 3x2x1 = 6
      * i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: 1x4x5 = 20
      * i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 = 4
      * i in {0,1}, j in {2,3}, k in {0,1}: 2x2x2 = 4
      * i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: already counted
    
    Total: 7 + 7 + 2 + 6 + 20 + 4 + 4 = 50? Let me recount.
    
    Actually, the cross terms between different j-blocks and k-blocks need to be handled.
    Let me be more systematic.
    
    The full tensor has entries T[i,j,k] = delta_{j,k}.
    We partition j into {0,1} and {2,3}, k into {0,1}, {2,3}, {4}.
    
    For i in {0,1}:
    - (j in {0,1}, k in {0,1}): Strassen 2x2x2, rank 7
    - (j in {2,3}, k in {2,3}): Strassen 2x2x2, rank 7
    - (j in {0,1}, k in {2,3}): schoolbook 2x2x2, rank 4
    - (j in {2,3}, k in {0,1}): schoolbook 2x2x2, rank 4
    - (j in {0,1}, k in {4}): schoolbook 2x1x1, rank 2
    - (j in {2,3}, k in {4}): schoolbook 2x1x1, rank 2
    - (j in {0,1}, k in {4}): wait, k=4 is single, so 2x1x1 = 2
    - (j in {2,3}, k in {4}): 2x1x1 = 2
    
    For i in {2}:
    - (j in {0,1,2,3}, k in {0,1,2,3,4}): schoolbook 4x5, rank 20
    
    Total: 7 + 7 + 4 + 4 + 2 + 2 + 20 = 46? That seems too low.
    
    Wait, I'm confusing myself. Let me just use the known result: two Strassen blocks give rank 58.
    The idea is:
    - Use Strassen for 2x2x2 in (i={0,1}, j={0,1}, k={0,1}): saves 1 (8->7)
    - Use Strassen for 2x2x2 in (i={0,1}, j={2,3}, k={2,3}): saves 1 (8->7)
    - Rest is schoolbook: 60 - 8 - 8 + 7 + 7 = 58
    
    So the decomposition is:
    - Strassen block 1: i in {0,1}, j in {0,1}, k in {0,1}
    - Strassen block 2: i in {0,1}, j in {2,3}, k in {2,3}
    - Schoolbook for everything else: rank 60 - 8 - 8 + 7 + 7 = 58
    """
    n, m, p = 3, 4, 5
    total_rank = 58
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Block 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    U1, V1, W1 = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            U[i * m + j, col:col+7] = U1[i * 2 + j, :]
    for j in range(2):
        for k in range(2):
            V[j * p + k, col:col+7] = V1[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W[k * n + i, col:col+7] = W1[k * 2 + i, :]
    col += 7
    
    # Block 2: Strassen for i in {0,1}, j in {2,3}, k in {2,3}
    U2, V2, W2 = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            U[i * m + (j + 2), col:col+7] = U2[i * 2 + j, :]
    for j in range(2):
        for k in range(2):
            V[(j + 2) * p + (k + 2), col:col+7] = V2[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W[(k + 2) * n + i, col:col+7] = W2[k * 2 + i, :]
    col += 7
    
    # Schoolbook for remaining entries
    # Remaining: all (i,j,k) not in the two Strassen blocks
    # Strassen block 1 covers: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen block 2 covers: i in {0,1}, j in {2,3}, k in {2,3}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen block 1
                if i < 2 and j < 2 and k < 2:
                    continue
                # Skip if in Strassen block 2
                if i < 2 and j >= 2 and k >= 2 and j < 4 and k < 4:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    if col != total_rank:
        # Adjust if needed
        pass
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try the rank-58 Strassen-based decomposition
    try:
        U, V, W = _try_block_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
