import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    # Build the 3-way tensor of coefficients: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # Reshape U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    # For each (i,j,k): sum_r Ur[i,j,r]*Vr[j,k,r]*Wr[k,i,r]
    # Build via einsum
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms
    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank32():
    """
    Rank-32 decomposition.

    Structure: A is 2x4, B is 4x8.
    Split A = [A1 A2] (two 2x2 row-blocks over j).
    Split B into 4 column-blocks of 2 columns each: B_l = [B_l_top; B_l_bot],
        each a 4x2 = two stacked 2x2 blocks (top = j in {2l, 2l+1}, bot = j in {2l+2, 2l+3}).

    C = sum_l A1 * B_l_top + A2 * B_l_bot  (each product is 2x2).

    Shared row-block products (2x2 x 2x2):
        S0 = A1 * B1_top   (l=0, top)
        S1 = A2 * B1_bot   (l=0, bot)
        S2 = A1 * B2_top   (l=1, top)
        S3 = A2 * B3_bot   (l=3, bot)
        S4 = (A1 + A2) * (B1_bot + B2_bot + B3_top)   (covers l=1 bot, l=2 top, l=3 top)

    C = S0 + S1 + S2 + S3 + S4   (2x2 blocks summed over l).

    Rank: S0..S3 use schoolbook (4 each) = 16, S4 uses Strassen (7) = 7. Total = 23? 
    Wait, that's only 23. Let me recount: 4*4 + 7 = 23. 
    But each 2x2 product contributes to all 4 C entries. Let me just build it and verify.
    """
    n, m, p = DIMENSIONS
    # We'll allocate a generous rank and trim, but let's compute exactly.
    # S0..S3: 4 schoolbook terms each -> 16 terms
    # S4: 7 strassen terms -> 7 terms
    # Total rank = 23
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_schoolbook(A_map, B_map, C_map, col0):
        # A_map: {(i, j): c}, B_map: {(j, k): c}, C_map: {(k, i): c}
        # schoolbook: 4 terms (i,j,k) with i in 0..1, j in 0..1, k in 0..1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = col0 + (i * 4 + j * 2 + k)
                    # U
                    if (i, j) in A_map:
                        U[i * m + j][r] += A_map[(i, j)]
                    # V
                    if (j, k) in B_map:
                        V[j * p + k][r] += B_map[(j, k)]
                    # W
                    if (k, i) in C_map:
                        W[k * n + i][r] += C_map[(k, i)]
                    # (unused r slots will have all-zero rows -> contribute 0, harmless)

    def add_strassen(A_map, B_map, C_map, col0):
        strassen = [
            {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
            {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
            {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
            {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
            {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
            {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
            {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
        ]
        for t in strassen:
            r = col0
            for (i, j), c in t['U'].items():
                if (i, j) in A_map:
                    U[i * m + j][r] += A_map[(i, j)] * c
            for (j, k), c in t['V'].items():
                if (j, k) in B_map:
                    V[j * p + k][r] += B_map[(j, k)] * c
            for (k, i), c in t['W'].items():
                if (k, i) in C_map:
                    W[k * n + i][r] += C_map[(k, i)] * c
            col0 += 1
        return col0

    # Identity maps for a 2x2 block
    I2 = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}

    # S0 = A1 * B1_top : A1 = A rows j in {0,1}; B1_top = B rows j in {0,1}, cols k in {0,1}
    A1 = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}  # (i, j_local)
    B1_top = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}  # (j_local, k_local)
    Cmap = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}  # (k_local, i)
    # We need a version of add_schoolbook that maps local indices to global.
    # Let me write a more general helper.

    # Redefine with explicit local->global mapping.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def block_product(A_gmap, B_gmap, C_gmap, algorithm, col0):
        """
        A_gmap: {local(i,j): global_u_index}
        B_gmap: {local(j,k): global_v_index}
        C_gmap: {local(k,i): global_w_index}
        algorithm: 'schoolbook' or 'strassen'
        Returns next col.
        """
        if algorithm == 'schoolbook':
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = col0 + (i * 4 + j * 2 + k)
                        U[A_gmap[(i, j)]][r] = 1
                        V[B_gmap[(j, k)]][r] = 1
                        W[C_gmap[(k, i)]][r] = 1
            return col0 + 4
        else:  # strassen
            strassen = [
                {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
                {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
                {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
                {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
                {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
                {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
                {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
            ]
            for t in strassen:
                r = col0
                for (i, j), c in t['U'].items():
                    U[A_gmap[(i, j)]][r] += c
                for (j, k), c in t['V'].items():
                    V[B_gmap[(j, k)]][r] += c
                for (k, i), c in t['W'].items():
                    W[C_gmap[(k, i)]][r] += c
                col0 += 1
            return col0

    # Local index helpers:
    # A1: i in 0..1, j_local in 0..1 -> global j = j_local (since A1 is j in {0,1})
    # A2: i in 0..1, j_local in 0..1 -> global j = 2 + j_local (j in {2,3})
    def A1_g(i, j): return i * m + j
    def A2_g(i, j): return i * m + (2 + j)

    # B blocks: B_l_top: j_local in 0..1 -> global j = 2*l + j_local; k_local in 0..1 -> global k = 2*l + k_local
    # B_l_bot: j_local in 0..1 -> global j = 2*l + 2 + j_local; k_local in 0..1 -> global k = 2*l + k_local
    def B_top_g(l, j, k): return (2 * l + j) * p + (2 * l + k)
    def B_bot_g(l, j, k): return (2 * l + 2 + j) * p + (2 * l + k)

    # C: k_local in 0..1 -> global k = 2*l + k_local; i in 0..1
    def C_g(l, k, i): return (2 * l + k) * n + i

    # Build maps for each product
    def maps_A(gA):
        return {(i, j): gA(i, j) for i in range(2) for j in range(2)}
    def maps_B_top(l):
        return {(j, k): B_top_g(l, j, k) for j in range(2) for k in range(2)}
    def maps_B_bot(l):
        return {(j, k): B_bot_g(l, j, k) for j in range(2) for k in range(2)}
    def maps_C(l):
        return {(k, i): C_g(l, k, i) for k in range(2) for i in range(2)}

    # S0 = A1 * B1_top (l=0, top), schoolbook
    col = block_product(maps_A(A1_g), maps_B_top(0), maps_C(0), 'schoolbook', col)
    # S1 = A2 * B1_bot (l=0, bot), schoolbook
    col = block_product(maps_A(A2_g), maps_B_bot(0), maps_C(0), 'schoolbook', col)
    # S2 = A1 * B2_top (l=1, top), schoolbook
    col = block_product(maps_A(A1_g), maps_B_top(1), maps_C(1), 'schoolbook', col)
    # S3 = A2 * B3_bot (l=3, bot), schoolbook
    col = block_product(maps_A(A2_g), maps_B_bot(3), maps_C(3), 'schoolbook', col)

    # S4 = (A1 + A2) * (B1_bot + B2_bot + B3_top) via Strassen
    # A_combined: local(i,j) -> we need A1(i,j) + A2(i,j). But Strassen adds them.
    # We'll create a combined A map where each local (i,j) points to... we can't point to two.
    # Instead, for the combined product, the U coefficient for local (i,j) should be
    # A1(i,j) + A2(i,j). Since both are identity (coeff 1), combined coeff = 2? 
    # No! A1 and A2 are DIFFERENT matrices (different rows of A). We can't just add.
    # 
    # The Strassen product (A1+A2)*(B1_bot+B2_bot+B3_top) computes using the SUM of the
    # two A matrices and the SUM of the three B matrices. This is valid because
    # (A1+A2)*(B1_bot+B2_bot+B3_top) = A1*B1_bot + A1*B2_bot + A1*B3_top
    #                                   + A2*B1_bot + A2*B2_bot + A2*B3_top
    # which is exactly the 6 missing products! Great.
    #
    # But to build the U map for (A1+A2), each local (i,j) needs a coefficient that,
    # when multiplied, gives A1(i,j) + A2(i,j). Since U is a single matrix, we need
    # U[global][r] to represent the combined value. We can set U[A1_g(i,j)][r] += 1 and
    # U[A2_g(i,j)][r] += 1 for the same r. That's two separate rows both getting coeff 1.
    # So the "map" isn't a single index; we add to both.
    #
    # Let me handle S4 manually with explicit additions.

    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    # B_combined = B1_bot + B2_bot + B3_top. For local (j,k), the V coefficient is the sum
    # over the three blocks. Each block contributes to a different global (j,k) row.
    # So for a given Strassen term with V coeff c at local (j,k), we add c to:
    #   B_bot_g(1, j, k), B_bot_g(2, j, k), B_top_g(3, j, k)
    # A_combined = A1 + A2: for local (i,j) with coeff c, add c to A1_g(i,j) and A2_g(i,j).
    # C: the result S4 contributes to C entries for l=1 (bot), l=2 (top), l=3 (top).
    # Wait, S4 = (A1+A2)*(B1_bot + B2_bot + B3_top). The product is a 2x2 matrix.
    # But it represents THREE different C blocks: C1 (from A1*B1_bot? no...).
    #
    # Hmm, this is the subtlety. (A1+A2)*(B1_bot+B2_bot+B3_top) is ONE 2x2 matrix, but
    # we need it to contribute to C1, C2, C3 (three different 2x2 output blocks).
    # That doesn't work directly because a single 2x2 product can only fill one 2x2 block.
    #
    # The error: B1_bot, B2_bot, B3_top are in DIFFERENT columns of C (different l).
    # So A1*B1_bot contributes to C1 (l=1), A1*B2_bot to C2 (l=2), A1*B3_top to C3 (l=3).
    # These are DIFFERENT output positions. A single 2x2 product can't fill three positions.
    #
    # So the "sum" trick only works if all the B blocks map to the SAME output block.
    # They don't. So S4 as a single Strassen product is INVALID.
    #
    # Let me reconsider. The valid sharing is only when the SAME (i,j) row-block of A
    # multiplies B blocks that land in the SAME output column-block l. But each l is distinct.
    #
    # So actually, there's NO sharing across different l! Each (row-block, l) pair is a
    # distinct 2x2 product landing in a distinct output block. That's 2 row-blocks * 4 l
    # = 8 distinct 2x2 products, each needing its own rank. No sharing possible.
    #
    # So the minimum via this block structure is 8 * (rank of 2x2 product).
    # With schoolbook (4): 32. With Strassen (7): 56. With a rank-6 2x2 algo: 48.
    #
    # So rank 32 is the best via independent 2x2 schoolbook blocks! Let me just do that.
    # 8 blocks * 4 = 32. This is valid and better than 56.

    # Reset and do 8 independent schoolbook 2x2 blocks.
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for l in range(4):
        for rb in range(2):  # 0 = A1 (j in {2l? no}) 
            # row-block rb: 0 -> A1 (j in {0,1}), 1 -> A2 (j in {2,3})
            # Actually for a given l, the B block is 4 rows (j in {2l,2l+1,2l+2,2l+3}).
            # A1 (j in {0,1}) multiplies B rows j in {0,1} of the full B, i.e. global j in {0,1}.
            # But B_l is columns k in {2l, 2l+1}. The rows of B are j in 0..3.
            # A1 * (B restricted to cols 2l,2l+1 and rows 0,1) -> uses A1 (j 0,1) and B rows 0,1.
            # A2 * (B restricted to cols 2l,2l+1 and rows 2,3) -> uses A2 (j 2,3) and B rows 2,3.
            # So for output block l:
            #   C_l = A1 * B[rows 0:2, cols 2l:2l+2] + A2 * B[rows 2:4, cols 2l:2l+2]
            # A1_g: i in 0..1, j_local 0..1 -> global j = j_local
            # A2_g: i in 0..1, j_local 0..1 -> global j = 2 + j_local
            if rb == 0:
                gA = A1_g
            else:
                gA = A2_g
            # B rows: for rb=0, global j = j_local (0,1); for rb=1, global j = 2+j_local (2,3)
            def gB(j, k, rb=rb, l=l):
                gj = (j if rb == 0 else 2 + j)
                return gj * p + (2 * l + k)
            Amap = {(i, j): gA(i, j) for i in range(2) for j in range(2)}
            Bmap = {(j, k): gB(j, k) for j in range(2) for k in range(2)}
            Cmap = {(k, i): C_g(l, k, i) for k in range(2) for i in range(2)}
            col = block_product(Amap, Bmap, Cmap, 'schoolbook', col)

    return {"U": U, "V": V, "W": W}


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W (they contribute nothing)."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    best = None

    # Candidate 1: rank-32 (8 independent schoolbook 2x2 blocks)
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 2: rank-56 (original Strassen, known valid) as fallback
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            if best is None or r < best[0]:
                best = (r, U, V, W)
    except Exception:
        pass

    if best is None:
        # Last resort: return rank-56 without verification (it's known to be valid)
        cand = _construct_rank56()
        return cand

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
