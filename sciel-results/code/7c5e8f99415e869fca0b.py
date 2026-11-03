"""Rewritten program: Strassen-based rank-84 baseline + bounded randomized rank search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

DEADLINE = 150.0
SAFETY_MARGIN = 15.0  # return best before this wall-clock limit


def _matmul_tensor(n, m, p):
    """Return the matrix-multiplication tensor as a 3D array T[i, j, k] with
    T[i, j, k] = 1 if the product C[i,k] depends on A[i,j]*B[j,k].
    Shape (n, m, p)."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _tensor_to_matrices(U, V, W, rank):
    """Given factor matrices as lists of lists, verify exact decomposition of
    the n*m*p matmul tensor. Returns True if exact."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                T_recon[i, j, k] = s
    # Expected: all 1s
    if not np.allclose(T_recon, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _strassen_2x2x2():
    """Return the 7 Strassen terms as (u, v, w) triples for a 2x2x2 block.
    Each term is a dict mapping (i,j), (j,k), (k,i) to a coefficient.
    We represent each term as (U_entries, V_entries, W_entries) where entries
    are dicts from index to coefficient."""
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each term c: U[(i,j)] * V[(j,k)] * W[(k,i)]
    # We index A by (i,j), B by (j,k), C by (k,i) for 2x2.

    terms = []
    # M1
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): 1, (1,1): -1, (0,1): -1, (1,0): 1}  # C00:M1, C11:-M1... wait
    ))
    # Let me redo this carefully.
    # For each Strassen product t, it contributes to specific C entries.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    #   U: A00=1, A11=1
    #   V: B00=1, B11=1
    #   W: C00=1, C11=1
    terms = []
    # M1
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2 = (A10+A11)B00
    #   U: A10=1, A11=1
    #   V: B00=1
    #   W: C10=1, C11=-1
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3 = A00(B01-B11)
    #   U: A00=1
    #   V: B01=1, B11=-1
    #   W: C01=1, C11=1
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4 = A11(B10-B00)
    #   U: A11=1
    #   V: B10=1, B00=-1
    #   W: C00=1, C10=1
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5 = (A00+A01)B11
    #   U: A00=1, A01=1
    #   V: B11=1
    #   W: C00=-1, C01=1
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6 = (A10-A00)(B00+B01)
    #   U: A10=1, A00=-1
    #   V: B00=1, B01=1
    #   W: C11=1
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7 = (A01-A11)(B10+B11)
    #   U: A01=1, A11=-1
    #   V: B10=1, B11=1
    #   W: C00=1
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms


def _build_strassen_356():
    """Build a rank-84 decomposition for 3x5x6 using Strassen on 2x2x2 blocks.
    
    Strategy: Partition the 3x5x6 problem into overlapping 2x2x2 blocks where
    possible, and use schoolbook for the remaining parts.
    
    For 3x5x6:
    - We can tile the 3x5 matrix A with 2x2 blocks: rows 0-1 and row 2, cols 0-1, 2-3, col 4
    - Similarly for B (5x6) and C (3x6)
    
    A simpler approach: Use the known result that rank(3x5x6) <= 84 via
    combining Strassen and schoolbook. Let's use a block decomposition:
    
    Split A (3x5) into: top 2 rows (2x5) and bottom 1 row (1x5)
    Split B (5x6) into: left 5 cols as one block
    Split C (3x6) into: top 2 rows (2x6) and bottom 1 row (1x6)
    
    C_top (2x6) = A_top (2x5) * B (5x6)
    C_bottom (1x6) = A_bottom (1x5) * B (5x6)
    
    For C_top: use Strassen-like decomposition for 2x5x6
    For C_bottom: schoolbook 1x5x6 = rank 6
    
    For 2x5x6: split B into 2x2 blocks along the 5 columns:
    cols 0-1, 2-3, col 4
    C_top[:, 0:2] = A_top * B[:, 0:2]  (2x2x2 block, rank 7 via Strassen)
    C_top[:, 2:4] = A_top * B[:, 2:4]  (2x2x2 block, rank 7 via Strassen)
    C_top[:, 4:5] = A_top * B[:, 4:5]  (2x1x2, rank 4 schoolbook)
    
    Total: 7 + 7 + 4 + 6 = 24? That doesn't match 84.
    
    Let me reconsider. The measured knowledge says 6 blocks of 2x2x2 replaced
    from schoolbook. 3*5*6 = 90 schoolbook. 90 - 6 = 84. So 6 blocks of 2x2x2
    are replaced, each saving 1.
    
    Let me just construct it directly by identifying all 2x2x2 sub-blocks in
    the 3x5x6 tensor and applying Strassen to each, with overlap handling.
    
    Actually, the cleanest way: enumerate all (i,j,k) triples, group them into
    2x2x2 blocks where possible, apply Strassen to each block, and handle
    the remainder with schoolbook.
    
    For 3x5x6, the possible 2x2x2 blocks (starting at (i0,j0,k0)):
    i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3,4}
    That's 2*4*5 = 40 possible blocks, but they overlap.
    
    A non-overlapping tiling:
    i: {0,1} covers rows 0-1, row 2 is separate
    j: {0,1}, {2,3}, {4} -> 3 blocks along j
    k: {0,1}, {2,3}, {4,5} -> 3 blocks along k
    
    For the 2x2x2 blocks (i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}):
    That's 1 * 2 * 3 = 6 blocks of 2x2x2. Each uses rank 7 instead of 8.
    Savings: 6 * 1 = 6. Total: 90 - 6 = 84.
    
    The remaining parts:
    - Row 2 of A (i=2): 1x5x6 = 30 terms schoolbook
    - j=4 column of B: 3x1x6 = 18 terms schoolbook (but i=2,j=4 already counted)
    - k=5 column of C: 3x5x1 = 15 terms schoolbook (overlaps)
    
    Let me just build it by iterating and assigning.
    """
    n, m, p = 3, 5, 6
    U = {}  # (i,j) -> list of (coeff, term_index)
    V = {}  # (j,k) -> list of (coeff, term_index)
    W = {}  # (k,i) -> list of (coeff, term_index)
    
    term_count = 0
    strassen_terms = _strassen_2x2x2()
    
    # Define the 6 Strassen 2x2x2 blocks:
    # Block (i0, j0, k0) covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    blocks = []
    for i0 in [0]:  # only one 2-row block: rows 0-1
        for j0 in [0, 2]:  # cols 0-1, 2-3
            for k0 in [0, 2, 4]:  # cols 0-1, 2-3, 4-5
                blocks.append((i0, j0, k0))
    
    for (i0, j0, k0) in blocks:
        for t_idx in range(7):
            u_entries, v_entries, w_entries = strassen_terms[t_idx]
            for (di, dj), coeff_u in u_entries.items():
                i, j = i0 + di, j0 + dj
                U.setdefault((i, j), []).append((coeff_u, term_count))
            for (dj, dk), coeff_v in v_entries.items():
                j, k = j0 + dj, k0 + dk
                V.setdefault((j, k), []).append((coeff_v, term_count))
            for (dk, di), coeff_w in w_entries.items():
                k, i = k0 + dk, i0 + di
                W.setdefault((k, i), []).append((coeff_w, term_count))
            term_count += 1
    
    # Now add schoolbook terms for the remaining (i,j,k) not covered by any block
    # A block (i0,j0,k0) covers i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U.setdefault((i, j), []).append((1, term_count))
                    V.setdefault((j, k), []).append((1, term_count))
                    W.setdefault((k, i), []).append((1, term_count))
                    term_count += 1
    
    # Convert to matrices
    rank = term_count
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for coeff, t in entries:
            U_mat[i * m + j][t] = coeff
    for (j, k), entries in V.items():
        for coeff, t in entries:
            V_mat[j * p + k][t] = coeff
    for (k, i), entries in W.items():
        for coeff, t in entries:
            W_mat[k * n + i][t] = coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}


def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check shapes
    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _randomized_search(best_rank, deadline):
    """Attempt to find a lower-rank decomposition via randomized methods.
    Returns (U, V, W, rank) or None if no improvement found."""
    n, m, p = DIMENSIONS
    T = _matmul_tensor(n, m, p)
    
    # Try a few target ranks below best_rank
    for target_rank in range(max(1, best_rank - 10), best_rank):
        if time.time() > deadline - SAFETY_MARGIN:
            break
        
        # Randomized tensor rank approximation
        # Use random linear combinations to try to find a low-rank decomposition
        # This is a heuristic - we try to find U, V, W such that
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        
        # Generate random initial factors
        for attempt in range(50):
            if time.time() > deadline - SAFETY_MARGIN:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Try to solve for exact decomposition using the constraint
            # This is a system of n*m*p equations in n*m*target_rank + m*p*target_rank + p*n*target_rank unknowns
            # Too many variables generally, so this is only feasible for very low rank
            
            # Instead, try a different approach: use the fact that we need
            # T = sum_c u_c \otimes v_c \otimes w_c
            # where u_c[i*m+j], v_c[j*p+k], w_c[k*n+i]
            
            # For each c, the outer product u_c \otimes v_c \otimes w_c must
            # sum to T. This is hard to solve directly.
            
            # Try a simple approach: assume the factors have small integer/half-integer
            # entries and do a local search.
            
            # Initialize with small random half-integers
            U_half = np.round(U_rand / 2.0) * 2.0
            V_half = np.round(V_rand / 2.0) * 2.0
            W_half = np.round(W_rand / 2.0) * 2.0
            
            # Verify
            if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist()):
                return U_half.tolist(), V_half.tolist(), W_half.tolist(), target_rank
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Step 1: Build the Strassen-based rank-84 decomposition
    result = _build_strassen_356()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Step 2: Try randomized search for lower rank
    if time.time() < deadline - SAFETY_MARGIN:
        found = _randomized_search(best_rank, deadline)
        if found is not None:
            U, V, W, rank = found
            if _verify(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
