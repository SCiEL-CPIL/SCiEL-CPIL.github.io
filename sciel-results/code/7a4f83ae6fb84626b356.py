"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 120.0

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    U_sel = U[i_idx * m + j_idx, :]
    V_sel = V[j_idx * p + k_idx, :]
    W_sel = W[k_idx * n + i_idx, :]
    T_flat = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.allclose(T_flat, np.ones(n * m * p), atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2_products():
    """7 Strassen rank-1 terms for 2x2x2 MM.
    Each: (U_dict, V_dict, W_dict) with local 2x2 indices (0..3 = row*2+col)."""
    return [
        # P1 = (A00+A11)*(B00+B11) -> C00, C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2 = (A10+A11)*B00 -> C10
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # P3 = A00*(B10-B11) -> C00, C01
        ({0: 1}, {2: 1, 3: -1}, {0: 1, 1: 1}),
        # P4 = A11*(B01-B00) -> C00, C11
        ({3: 1}, {1: 1, 0: -1}, {3: 1, 2: 1}),
        # P5 = (A00+A01)*B11 -> C01
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        # P6 = (A10-A00)*(B00+B10) -> C10
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {2: 1}),
        # P7 = (A01-A11)*(B01+B11) -> C01, C11
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {1: 1, 3: 1}),
    ]


def _construct_hierarchical(n, m, p):
    """
    Hierarchical decomposition:
    3x3x8 = 3 x (3x4) where 3x4 = 2x2 + 2x2
    Each 3x2 = 2x2 + 1x2
    So 3x3x8 = 6 x (2x2x2)
    Each 2x2x2 uses Strassen rank 7.
    Total rank = 6 * 7 = 42.
    
    Block structure:
    - i in {0,1} (2x2 row blocks), i=2 is 1x2
    - j in {0,1} (2x2 col blocks), j=2 is 1x2
    - k in {0,1,2,3,4,5,6,7} split into pairs: (0,1),(2,3),(4,5),(6,7)
    
    For each (i_pair, j_pair, k_pair) where i_pair in {0,1} means i in {2*i_pair, 2*i_pair+1},
    j_pair in {0,1} means j in {2*j_pair, 2*j_pair+1},
    k_pair in {0,1,2,3} means k in {2*k_pair, 2*k_pair+1}:
    Use Strassen decomposition.
    
    For i=2 (single row), j in {0,1}, k_pair in {0,1,2,3}:
    1x2x2 = 2x2 (schoolbook) -> rank 2 each, 4 blocks = 8
    Wait, that gives 4*7 + 8 = 36. Let me reconsider.
    
    Actually: 3x3x8
    = (2x2 + 1x2) x 3 x 8
    = 2x2x3x8 + 1x2x3x8
    
    2x2x3x8: i in {0,1}, j in {0,1,2}, k in {0..7}
    = 2x2 x (2x2 + 1x2) x 8  [j split into {0,1} and {2}]
    = 2x2x2x2x8 + 2x2x1x2x8
    = 2x2x2x2x(4x2) + 2x2x1x2x(4x2)
    = 4x(2x2x2x2) + 4x(2x2x1x2)
    = 4x7 + 4x2 = 28 + 8 = 36
    
    1x2x3x8: i=2, j in {0,1,2}, k in {0..7}
    = 1x2 x (2x2 + 1x2) x 8
    = 1x2x2x2x8 + 1x2x1x2x8
    = 4x(1x2x2x2) + 4x(1x2x1x2)
    = 4x2 + 4x1 = 8 + 4 = 12
    
    Total = 36 + 12 = 48. Hmm.
    
    Let me try a different split.
    3x3x8 = 3 x 3 x 8
    Split j into {0,1} and {2}: 3x2x8 + 3x1x8
    3x2x8 = (2x2+1x2)x2x8 = 2x2x2x8 + 1x2x2x8
    2x2x2x8 = 2x2x2x(4x2) = 4x(2x2x2x2) = 4x7 = 28
    1x2x2x8 = 1x2x2x(4x2) = 4x(1x2x2x2) = 4x2 = 8
    3x2x8 = 28+8 = 36
    
    3x1x8 = 3x1x(4x2) = 4x(3x1x2) = 4x3 = 12 (schoolbook, 1x2 = 2 terms)
    
    Total = 36+12 = 48.
    
    Alternative: split k differently or use different structure.
    Let me just use the straightforward 2x2x2 blocks where possible.
    """
    prods = _strassen_2x2_products()
    # For each 2x2x2 block, use Strassen (rank 7)
    # For 1x2x2 blocks, use schoolbook (rank 2)
    # For 2x2x1 blocks, use schoolbook (rank 2)
    # For 1x1x2 blocks, use schoolbook (rank 2)
    # For 2x1x2 blocks, use schoolbook (rank 2)
    # For 1x2x1 blocks, use schoolbook (rank 2)
    # For 1x1x1 blocks, use schoolbook (rank 1)
    
    # Define blocks: (i_start, i_end, j_start, j_end, k_start, k_end)
    # We'll partition i={0,1,2}, j={0,1,2}, k={0..7}
    
    # Strategy: 
    # i=0,1 (2x2), j=0,1 (2x2), k pairs (0,1),(2,3),(4,5),(6,7) -> 4 blocks of 2x2x2, rank 7 each = 28
    # i=0,1 (2x2), j=2 (1x1), k pairs -> 4 blocks of 2x2x1, rank 2 each = 8
    # i=2 (1x1), j=0,1 (2x2), k pairs -> 4 blocks of 1x2x2, rank 2 each = 8
    # i=2 (1x1), j=2 (1x1), k pairs -> 4 blocks of 1x1x2, rank 2 each = 8
    # Total = 28+8+8+8 = 52
    
    # Better: 
    # i=0,1 (2x2), j=0,1,2 (3), k pairs
    # j=0,1 (2x2): 2x2x2x2 blocks, rank 7, 4 blocks = 28
    # j=2 (1x1): 2x2x1x2 blocks, rank 2, 4 blocks = 8
    # i=2 (1x1), j=0,1,2 (3), k pairs
    # j=0,1 (2x2): 1x2x2x2 blocks, rank 2, 4 blocks = 8
    # j=2 (1x1): 1x1x1x2 blocks, rank 2, 4 blocks = 8
    # Total = 28+8+8+8 = 52
    
    # Hmm, let me try:
    # i=0,1 (2x2), j=0,1 (2x2), k=0..7 (4x2)
    # = 4x(2x2x2x2) = 4x7 = 28
    # i=0,1 (2x2), j=2 (1x1), k=0..7 (4x2)
    # = 4x(2x2x1x2) = 4x2 = 8
    # i=2 (1x1), j=0,1 (2x2), k=0..7 (4x2)
    # = 4x(1x2x2x2) = 4x2 = 8
    # i=2 (1x1), j=2 (1x1), k=0..7 (4x2)
    # = 4x(1x1x1x2) = 4x2 = 8
    # Total = 52
    
    # Can we do better? What if we don't split k into pairs for some blocks?
    # i=0,1 (2x2), j=0,1 (2x2), k=0..7
    # If we use Strassen on k as well: 2x2x2x8 = 2x2x2x(4x2)
    # We can apply Strassen to the k dimension: 2x2x2x2 = rank 7
    # So 2x2x2x8 = 4x(2x2x2x2) = 4x7 = 28 (same as before)
    
    # What about 3x3x8 = 3x(3x8)?
    # 3x8 = 3x(4x2) = 4x(3x2)
    # 3x2 = (2x1 + 1x1) x 2 = 2x2 + 1x2
    # 2x2 = rank 4 (schoolbook) or... 
    # This is getting complicated. Let me just use the 52 rank solution.
    
    # Actually, let me reconsider. 3x3x8:
    # The tensor is T[i,j,k] = A[i,j] * B[j,k]
    # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = A[i,j]*B[j,k]
    
    # Simplest: use 2x2x2 Strassen blocks where we have 2x2x2 subproblems.
    # Partition:
    # i: {0,1} and {2}
    # j: {0,1} and {2}
    # k: {0,1}, {2,3}, {4,5}, {6,7}
    
    # Blocks:
    # (i=0,1; j=0,1; k=0,1): 2x2x2, Strassen, rank 7
    # (i=0,1; j=0,1; k=2,3): 2x2x2, Strassen, rank 7
    # (i=0,1; j=0,1; k=4,5): 2x2x2, Strassen, rank 7
    # (i=0,1; j=0,1; k=6,7): 2x2x2, Strassen, rank 7
    # (i=0,1; j=2; k=0,1): 2x1x2, schoolbook, rank 2
    # (i=0,1; j=2; k=2,3): 2x1x2, schoolbook, rank 2
    # (i=0,1; j=2; k=4,5): 2x1x2, schoolbook, rank 2
    # (i=0,1; j=2; k=6,7): 2x1x2, schoolbook, rank 2
    # (i=2; j=0,1; k=0,1): 1x2x2, schoolbook, rank 2
    # (i=2; j=0,1; k=2,3): 1x2x2, schoolbook, rank 2
    # (i=2; j=0,1; k=4,5): 1x2x2, schoolbook, rank 2
    # (i=2; j=0,1; k=6,7): 1x2x2, schoolbook, rank 2
    # (i=2; j=2; k=0,1): 1x1x2, schoolbook, rank 2
    # (i=2; j=2; k=2,3): 1x1x2, schoolbook, rank 2
    # (i=2; j=2; k=4,5): 1x1x2, schoolbook, rank 2
    # (i=2; j=2; k=6,7): 1x1x2, schoolbook, rank 2
    
    # Total = 4*7 + 4*2 + 4*2 + 4*2 = 28 + 8 + 8 + 8 = 52
    
    # Can we do better by not pairing k for the 1x1 and 2x1/1x2 blocks?
    # (i=0,1; j=2; k=0..7): 2x1x8, schoolbook rank 16
    # vs 4x(2x1x2) = 4x2 = 8. Same.
    
    # What if we use Strassen on a 2x2x2 block that spans i=0,1 and j=2 and k pair?
    # No, j=2 is only 1 element.
    
    # Alternative: use 3x3x2 blocks.
    # 3x3x2: can we do better than 18?
    # 3x3x2 = (2x2+1x2)x(2x2+1x2)x2
    # This is complex. Let me just go with 52.
    
    # Actually, let me try a different approach.
    # 3x3x8 = 3x3x(2x4)
    # Apply Strassen to the 2x2 blocks in the k dimension? No, k is 8.
    
    # Let me try: split i={0,1,2}, j={0,1,2}, k={0..7}
    # Use 2x2x2 Strassen for i={0,1}, j={0,1}, k={0,1},{2,3},{4,5},{6,7}: 4*7=28
    # For i={0,1}, j={2}, k={0..7}: 2x1x8, rank 16
    # For i={2}, j={0,1}, k={0..7}: 1x2x8, rank 16
    # For i={2}, j={2}, k={0..7}: 1x1x8, rank 8
    # Total = 28+16+16+8 = 68. Worse.
    
    # So pairing k is better for the non-2x2x2 blocks.
    # 52 is our best so far. Let me construct it.
    
    rank = 52
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    
    # 2x2x2 Strassen blocks: i={0,1}, j={0,1}, k pairs
    for k_pair in range(4):
        k0 = 2 * k_pair
        for (ud, vd, wd) in prods:
            for li, cf in ud.items():
                i_local = li // 2
                j_local = li % 2
                U[i_local * m + j_local, col] += cf
            for li, cf in vd.items():
                j_local = li // 2
                k_local = li % 2
                V[j_local * p + (k0 + k_local), col] += cf
            for li, cf in wd.items():
                k_local = li // 2
                i_local = li % 2
                W[(k0 + k_local) * n + i_local, col] += cf
            col += 1
    
    # 2x1x2 blocks: i={0,1}, j={2}, k pairs
    for k_pair in range(4):
        k0 = 2 * k_pair
        for k_local in range(2):
            U[0 * m + 2, col] = 1  # i=0, j=2
            V[2 * p + (k0 + k_local), col] = 1
            W[(k0 + k_local) * n + 0, col] = 1
            col += 1
            U[1 * m + 2, col] = 1  # i=1, j=2
            V[2 * p + (k0 + k_local), col] = 1
            W[(k0 + k_local) * n + 1, col] = 1
            col += 1
    
    # 1x2x2 blocks: i={2}, j={0,1}, k pairs
    for k_pair in range(4):
        k0 = 2 * k_pair
        for j_local in range(2):
            for k_local in range(2):
                U[2 * m + j_local, col] = 1  # i=2, j=j_local
                V[j_local * p + (k0 + k_local), col] = 1
                W[(k0 + k_local) * n + 2, col] = 1
                col += 1
    
    # 1x1x2 blocks: i={2}, j={2}, k pairs
    for k_pair in range(4):
        k0 = 2 * k_pair
        for k_local in range(2):
            U[2 * m + 2, col] = 1  # i=2, j=2
            V[2 * p + (k0 + k_local), col] = 1
            W[(k0 + k_local) * n + 2, col] = 1
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    return U, V, W


def _construct_strassen_blocks(n, m, p, blocks):
    """Build decomposition: Strassen blocks + schoolbook for remainder."""
    prods = _strassen_2x2_products()
    covered = set()
    for (i0, j0, k0) in blocks:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    covered.add((i0+a, j0+b, k0+c))
    rank = 7 * len(blocks) + (n*m*p - 8*len(blocks))
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for (i0, j0, k0) in blocks:
        for (ud, vd, wd) in prods:
            for li, cf in ud.items():
                U[(i0 + li//2) * m + (j0 + li%2), col] += cf
            for li, cf in vd.items():
                V[(j0 + li//2) * p + (k0 + li%2), col] += cf
            for li, cf in wd.items():
                W[(k0 + li//2) * n + (i0 + li%2), col] += cf
            col += 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    return U, V, W


def _find_best_blocks(n, m, p):
    """Find the set of non-overlapping 2x2x2 Strassen blocks that gives lowest rank."""
    best_rank = n * m * p
    best_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            k_vals = list(range(0, p - 1, 2))
            valid = True
            for a in range(len(k_vals)):
                for b in range(a+1, len(k_vals)):
                    if not (k_vals[a]+1 < k_vals[b] or k_vals[b]+1 < k_vals[a]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            blocks = [(i0, j0, k) for k in k_vals]
            ok = True
            for a in range(len(blocks)):
                for b in range(a+1, len(blocks)):
                    if not (blocks[a][0]+1 < blocks[b][0] or blocks[b][0]+1 < blocks[a][0] or
                            blocks[a][1]+1 < blocks[b][1] or blocks[b][1]+1 < blocks[a][1] or
                            blocks[a][2]+1 < blocks[b][2] or blocks[b][2]+1 < blocks[a][2]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            rank = 7 * len(blocks) + (n*m*p - 8*len(blocks))
            if rank < best_rank:
                best_rank = rank
                best_blocks = blocks
    return best_blocks


def _reduce_rank(U, V, W, n, m, p, max_iter=200):
    """Try to reduce rank by finding and eliminating dependent rank-1 terms."""
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    for _ in range(max_iter):
        found = -1
        for c in range(rank):
            if np.allclose(U[:, c], 0, atol=1e-12) and \
               np.allclose(V[:, c], 0, atol=1e-12) and \
               np.allclose(W[:, c], 0, atol=1e-12):
                found = c
                break
        if found < 0:
            break
        U = np.delete(U, found, axis=1)
        V = np.delete(V, found, axis=1)
        W = np.delete(W, found, axis=1)
        rank -= 1
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()

    # Baseline: schoolbook
    U, V, W = _schoolbook(n, m, p)
    assert _verify(U, V, W, n, m, p), "Schoolbook failed"
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try hierarchical decomposition
    if time.time() - t0 < DEADLINE - 10:
        U2, V2, W2 = _construct_hierarchical(n, m, p)
        if _verify(U2, V2, W2, n, m, p):
            if U2.shape[1] < best_rank:
                best_rank = U2.shape[1]
                best_U, best_V, best_W = U2, V2, W2

    # Try Strassen block configurations
    if time.time() - t0 < DEADLINE - 10:
        blocks = _find_best_blocks(n, m, p)
        if blocks:
            U3, V3, W3 = _construct_strassen_blocks(n, m, p, blocks)
            if _verify(U3, V3, W3, n, m, p):
                if U3.shape[1] < best_rank:
                    best_rank = U3.shape[1]
                    best_U, best_V, best_W = U3, V3, W3

    # Try to reduce rank further
    if time.time() - t0 < DEADLINE - 5:
        U4, V4, W4, r4 = _reduce_rank(best_U, best_V, best_W, n, m, p)
        if r4 < best_rank and _verify(U4, V4, W4, n, m, p):
            best_rank = r4
            best_U, best_V, best_W = U4, V4, W4

    # Convert to int if possible, else keep as float
    def _to_list(mat):
        if np.allclose(mat, np.round(mat), atol=1e-9):
            return np.round(mat).astype(np.int64).tolist()
        return mat.tolist()

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}

# EVOLVE-BLOCK-END
