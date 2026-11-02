import time
import itertools

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def _build_strassen_block_decomposition():
    """Fallback: 6 independent Strassen blocks, rank 42 (known valid)."""
    n, m, p = 2, 2, 12
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},
    ]
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b][col] = val
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] = val
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=2, m=2, p=12, tol=1e-9):
    """Check sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] exactly (within tol).
    T[a,b,c] = 1 iff row(A)=row(C) and col(B)=col(C), else 0.
    a in [0,4): a = 2*rowA + colA
    b in [0,24): b = 2*rowB + colB   (B is m x p = 2 x 12, row-major)
    c in [0,24): c = colC*n + rowC   (W trace convention W[k*n+i], k=colC, i=rowC)
    """
    rank = len(U[0])
    # Precompute for speed: for each r, U[:,r] (4), V[:,r] (24), W[:,r] (24)
    # T is sparse: nonzero only when rowA==rowC and colB==colC.
    # For each (rowA, colA) in A and (rowB, colB) in B and (rowC, colC) in C:
    #   T = 1 if rowA==rowC and colB==colC
    # We check all 4*24*24 = 2304 entries.
    for a in range(4):
        rowA, colA = divmod(a, 1)  # a = colA since n*m=4, A is 2x2: a = rowA*2+colA
        # Actually a indexes A entries: a = rowA * 2 + colA
        rowA, colA = a // 2, a % 2
        for b in range(24):
            rowB, colB = b // 12, b % 12  # B is 2 x 12, row-major: b = rowB*12+colB
            for c in range(24):
                colC, rowC = c // 2, c % 2  # W[k*n+i]: c = colC*2 + rowC
                expected = 1.0 if (rowA == rowC and colB == colC) else 0.0
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - expected) > tol:
                    return False
    return True


def _build_global_strassen(rank_target):
    """Attempt a global rank-r decomposition.
    Note: This approach assumes shared A-forms across all blocks, which is generally
    invalid for p > 2 unless specific symmetries are exploited. This function is kept
    for structural consistency but will likely fail verification for p=12, triggering
    the fallback to the valid block decomposition.
    """
    n, m, p = 2, 2, 12
    # 7 Strassen A-forms: A_s[a] for a in 0..3
    strassen_A = [
        [1, 0, 0, 1],   # M1 = a + d
        [0, 0, 1, 1],   # M2 = c + d
        [1, 0, 0, 0],   # M3 = a
        [0, 0, 0, 1],   # M4 = d
        [1, 1, 0, 0],   # M5 = a + b
        [-1, 0, 1, 0],  # M6 = c - a
        [0, 1, 0, -1],  # M7 = b - d
    ]
    r = rank_target
    # Cap the number of Strassen terms to 7 to avoid index errors.
    # If rank_target > 7, we cannot simply tile more Strassen terms without changing the structure.
    # We will just use the first min(r, 7) terms and leave the rest as zeros, which will fail verification.
    num_terms = min(r, 7)
    
    # Block-local Strassen B-forms (for 2x2x2): indexed by s (0..6) and local_b (0..3)
    strassen_B_block = [
        [1, 0, 0, 1],   # M1: e + h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f - h
        [-1, 0, 1, 0],  # M4: g - e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e + f
        [0, 0, 1, 1],   # M7: g + h
    ]
    # Block-local Strassen W-forms (for 2x2x2): indexed by s and local_c (0..3)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W_block = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    num_blocks = p // 2  # 6
    
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(24)]
    W = [[0] * r for _ in range(24)]
    
    for s in range(num_terms):
        # U: A-form s
        for a in range(4):
            U[a][s] = strassen_A[s][a]
        # V: B-form s, tiled over blocks
        for t in range(num_blocks):
            for local_b in range(4):
                val = strassen_B_block[s][local_b]
                if val == 0:
                    continue
                if local_b == 0:
                    gidx = 0 * 12 + 2 * t
                elif local_b == 1:
                    gidx = 0 * 12 + 2 * t + 1
                elif local_b == 2:
                    gidx = 1 * 12 + 2 * t
                else:
                    gidx = 1 * 12 + 2 * t + 1
                V[gidx][s] += val
        # W: W-form s, tiled over blocks
        for t in range(num_blocks):
            for local_c in range(4):
                val = strassen_W_block[s][local_c]
                if val == 0:
                    continue
                if local_c == 0:
                    gidx = (2 * t) * 2 + 0
                elif local_c == 1:
                    gidx = (2 * t + 1) * 2 + 0
                elif local_c == 2:
                    gidx = (2 * t) * 2 + 1
                else:
                    gidx = (2 * t + 1) * 2 + 1
                W[gidx][s] += val
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # STRUCTURAL CHANGE: global Strassen decomposition with shared A-forms.
    # Rank 7 (down from 42). Verify exactly; fall back to rank 42 if invalid.
    best = None
    best_rank = None

    # Try rank 7 global
    for r in [7, 8]:
        if time.time() > deadline:
            break
        cand = _build_global_strassen(r)
        if _verify(cand["U"], cand["V"], cand["W"]):
            if best is None or r < best_rank:
                best = cand
                best_rank = r
            break

    if best is None:
        # Fallback to known-valid rank 42 block decomposition
        best = _build_strassen_block_decomposition()
        best_rank = 42

    return best
# EVOLVE-BLOCK-END
