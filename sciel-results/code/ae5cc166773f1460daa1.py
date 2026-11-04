"""Combined strategy: Use Strassen decomposition for 2x2 blocks and schoolbook for the rest."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Target tensor T[i, a, k] = 1 for all i, a, k
        # Entries are integers in this construction, so plain int arithmetic is exact.
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_c = u_row[c]
                        if u_c == 0:
                            continue
                        v_c = v_row[c]
                        if v_c == 0:
                            continue
                        w_c = w_row[c]
                        if w_c == 0:
                            continue
                        acc += u_c * v_c * w_c
                    if acc != 1:
                        return False
        return True

    def _reduce_rank(candidate):
        """Scan rank slots; if two slots are scalar multiples of each other
        (same support pattern up to a rational scale), merge them.
        Only applies when all entries are integers and the merge is exact.
        Returns (possibly new) candidate with lower or equal rank."""
        U = [row[:] for row in candidate["U"]]
        V = [row[:] for row in candidate["V"]]
        W = [row[:] for row in candidate["W"]]
        rank = len(U[0])

        # Build signature per column: frozenset of (row_idx, value) for nonzero entries
        # Two columns c1, c2 are mergeable if there exists a rational s != 0 such that
        # for every row: U[r][c1] == s * U[r][c2] (and same for V, W).
        # Equivalently, the vectors (U_col, V_col, W_col) are linearly dependent with ratio.
        # Simpler: check if the concatenated vectors are proportional.

        from math import gcd
        def _proportional(a, b):
            """Return (num, den) such that a = (num/den) * b, or None if not proportional.
            a, b are lists of ints. Must have same nonzero pattern."""
            # Find first nonzero in a
            idx = None
            for j in range(len(a)):
                if a[j] != 0:
                    idx = j
                    break
            if idx is None:
                # a is all zeros; proportional only if b is all zeros
                if all(x == 0 for x in b):
                    return (0, 1)
                return None
            if b[idx] == 0:
                return None
            num = a[idx]
            den = b[idx]
            g = gcd(abs(num), abs(den))
            num //= g
            den //= g
            # Verify for all j
            for j in range(len(a)):
                if a[j] * den != b[j] * num:
                    return None
            return (num, den)

        # Concatenate U, V, W columns for each rank slot
        def _col_vec(c):
            return [U[r][c] for r in range(len(U))] + \
                   [V[r][c] for r in range(len(V))] + \
                   [W[r][c] for r in range(len(W))]

        cols = [_col_vec(c) for c in range(rank)]

        # Try to merge columns: iterate, if col c is proportional to an earlier kept col, skip it
        kept = []  # list of (col_index, col_vector)
        mapping = []  # mapping[c] = index in kept, or -1 if merged
        for c in range(rank):
            merged = False
            for ki, (kidx, kvec) in enumerate(kept):
                ratio = _proportional(cols[c], kvec)
                if ratio is not None:
                    # cols[c] = (num/den) * kvec
                    # To merge: we can just drop cols[c] if it's a multiple,
                    # but we need the sum to still give 1.
                    # Actually, if we have two terms t1 and t2 = s*t1, their sum is (1+s)*t1.
                    # We can't just drop one; we need to scale.
                    # This is more complex. For now, only merge if one is zero (already handled)
                    # or if they are identical (s=1), in which case we'd need to keep one and
                    # adjust... Actually for matrix mult tensor, duplicate identical terms
                    # would give 2 instead of 1, so they shouldn't exist in a valid decomp.
                    # So proportional pairs with s != 1 are the interesting case, but merging
                    # requires rescaling which changes the decomposition.
                    # Skip for now - only exact duplicates (which shouldn't occur) or
                    # one being zero.
                    if ratio == (0, 1):
                        # cols[c] is all zeros, safe to drop
                        merged = True
                        break
                    # For now, don't merge proportional pairs (too risky)
                    pass
            if not merged:
                mapping.append(len(kept))
                kept.append((c, cols[c]))
            else:
                mapping.append(-1)

        if len(kept) < rank:
            new_rank = len(kept)
            new_U = [[0] * new_rank for _ in range(len(U))]
            new_V = [[0] * new_rank for _ in range(len(V))]
            new_W = [[0] * new_rank for _ in range(len(W))]
            for c in range(rank):
                ki = mapping[c]
                if ki == -1:
                    continue
                for r in range(len(U)):
                    new_U[r][ki] = U[r][c]
                for r in range(len(V)):
                    new_V[r][ki] = V[r][c]
                for r in range(len(W)):
                    new_W[r][ki] = W[r][c]
            return {"U": new_U, "V": new_V, "W": new_W}
        return candidate

    # Strassen 2x2x2 decomposition
    # Vectors for U (i,a), V (a,k), W (k,i)
    # Indices: U: 00,01,10,11 -> 0,1,2,3
    #          V: 00,01,10,11 -> 0,1,2,3
    #          W: 00,01,10,11 -> 0,1,2,3 (k*n+i, n=2)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Verify Strassen decomposition internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0,1. Cols 0..7.
    # Split B1 cols into 4 blocks of 2: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            # U indices: i in 0,1; a in 0,1 (global a = local a)
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0: continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            
            # V indices: a in 0,1 (global a = local a); k in k0, k1 (global k = k0 + lk)
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0: continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            
            # W indices: k in k0, k1 (global k = k0 + lk); i in 0,1 (global i = li)
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0: continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 uses col 2 of A (global a=2).
    # B2 uses row 2 of B (global a=2).
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    
    if not _validate(candidate):
        return _schoolbook()

    # Attempt rank reduction
    if _elapsed_ok():
        reduced = _reduce_rank(candidate)
        if _validate(reduced):
            return reduced

    return candidate

# EVOLVE-BLOCK-END
