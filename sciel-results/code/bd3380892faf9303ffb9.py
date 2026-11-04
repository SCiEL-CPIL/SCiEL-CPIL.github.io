"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Primary design (valid, rank 32):
  Split the 8 output columns into four 2-column blocks.  Each 2x3 x 3x2
  all-ones sub-tensor is the direct sum of
    * a 2x2x2 all-ones tensor  -> rank 7 (Strassen)
    * a 2x1x2 all-ones tensor  -> rank 1 (outer product)
  giving rank 8 per block, total rank 32 (vs schoolbook 48).

Refinement (rank-reducing search, only ever accepted when validated):
  * GF(2) set-cover heuristic: model each rank-1 term's support as a
    48-bit vector; greedily select minimal set whose XOR covers all-ones.
  * Bounded by explicit deadline guard.
"""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition (T[i,j,k] == 1)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

# Local 2x2 Strassen rank-7 terms in (i, j), (j, k'), (k', i) coordinates.
_STRASSEN_U = [
    {(0, 0): 1, (1, 1): 1},
    {(1, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (0, 0): -1},
    {(0, 1): 1, (1, 1): -1},
]
_STRASSEN_V = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(0, 1): 1, (1, 1): -1},
    {(1, 0): 1, (0, 0): -1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (1, 1): 1},
]
_STRASSEN_W = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 1): 1, (1, 1): -1},
    {(0, 1): 1, (1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(0, 0): -1, (0, 1): 1},
    {(1, 1): 1},
    {(0, 0): 1},
]

def _block_decomposition(n, m, p):
    """Valid rank-32 decomposition: 4 blocks x (7 Strassen + 1 outer product)."""
    n, m, p = 2, 3, 8
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b  # first column of this block
        col = b * 8

        # 7 Strassen terms (j in {0,1}, k in {k0, k0+1})
        for r in range(7):
            for (i, j), val in _STRASSEN_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val

        # 1 outer-product term (j = 2)
        oc = col + 7
        for i in range(n):
            U[i * m + 2][oc] = 1
        for kp in range(2):
            V[2 * p + (k0 + kp)][oc] = 1
        for kp in range(2):
            for i in range(n):
                W[(k0 + kp) * n + i][oc] = 1

    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Refinement machinery: rank-2 searches that only ever improve the rank.
# ---------------------------------------------------------------------------

def _gf2_set_cover(n, m, p, deadline, rng):
    """GF(2) greedy set-cover heuristic for rank minimization.
    
    Each rank-1 term has a binary support pattern (48-bit vector indicating
    which (i,j,k) entries it covers). We greedily select terms whose XOR
    reconstructs the all-ones tensor, then solve for scalar coefficients.
    """
    # Generate candidate rank-1 terms: all combinations of sparse U, V, W
    # patterns. For efficiency, use structured candidates.
    
    # A rank-1 term is defined by:
    # U: sparse vector in R^(n*m) with support S_U
    # V: sparse vector in R^(m*p) with support S_V  
    # W: sparse vector in R^(p*n) with support S_W
    # The term contributes to (i,j,k) iff j in proj_U(i,j), k in proj_V(j,k), i in proj_W(k,i)
    
    # For GF(2) set cover, we just need the support pattern (binary)
    # Let's enumerate all possible support patterns from simple terms
    
    # Simple term types:
    # Type A: U has support {(i0,j0)}, V has support {(j0,k0)}, W has support {(k0,i0)}
    #   -> covers only (i0,j0,k0)
    # Type B: U has support {(i0,j0),(i1,j0)}, V has support {(j0,k0)}, W has support {(k0,i0),(k0,i1)}
    #   -> covers (i0,j0,k0) and (i1,j0,k0)
    # etc.
    
    # For a more systematic approach, let's generate terms where each factor
    # has support of size 1 or 2, and compute the resulting 48-bit mask.
    
    all_masks = []  # list of (mask, U_dict, V_dict, W_dict)
    
    # Generate all rank-1 terms with small supports
    # U support: subsets of {(i,j) for i in range(n), j in range(m)} with size 1 or 2
    # V support: subsets of {(j,k) for j in range(m), k in range(p)} with size 1 or 2
    # W support: subsets of {(k,i) for k in range(p), i in range(n)} with size 1 or 2
    
    U_supports = []
    for i in range(n):
        for j in range(m):
            U_supports.append([(i, j)])
    for i1 in range(n):
        for j1 in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    if (i1, j1) < (i2, j2):
                        U_supports.append([(i1, j1), (i2, j2)])
    
    V_supports = []
    for j in range(m):
        for k in range(p):
            V_supports.append([(j, k)])
    for j1 in range(m):
        for k1 in range(p):
            for j2 in range(m):
                for k2 in range(p):
                    if (j1, k1) < (j2, k2):
                        V_supports.append([(j1, k1), (j2, k2)])
    
    W_supports = []
    for k in range(p):
        for i in range(n):
            W_supports.append([(k, i)])
    for k1 in range(p):
        for i1 in range(n):
            for k2 in range(p):
                for i2 in range(n):
                    if (k1, i1) < (k2, i2):
                        W_supports.append([(k1, i1), (k2, i2)])
    
    # This is too many combinations. Let's be smarter:
    # Only consider terms where the supports are "aligned" to produce non-empty intersection.
    # A term (U, V, W) covers (i,j,k) iff:
    #   (i,j) in U_support AND (j,k) in V_support AND (k,i) in W_support
    
    # Let's just generate a manageable set of candidate terms
    # and use greedy set cover on their masks.
    
    # For now, let's use the known Strassen terms and outer product terms
    # as our candidate pool, and see if a smaller subset works via GF(2).
    
    # Actually, the directive asks for a set-cover approach. Let me implement
    # a proper greedy set cover on the space of all simple rank-1 terms.
    
    # To keep it tractable, limit to terms where each factor has support size <= 2
    # and the resulting mask is non-empty.
    
    candidates = []  # (mask_int, U_dict, V_dict, W_dict)
    
    # Limit the search space for tractability
    max_U_size = 2
    max_V_size = 2
    max_W_size = 2
    
    # Generate U supports of size 1 and 2
    u_list = []
    for i in range(n):
        for j in range(m):
            u_list.append([(i, j)])
    for i1 in range(n):
        for j1 in range(m):
            for i2 in range(i1, n):
                for j2 in range(m):
                    if i1 == i2:
                        if j1 < j2:
                            u_list.append([(i1, j1), (i2, j2)])
                    else:
                        u_list.append([(i1, j1), (i2, j2)])
    
    # Generate V supports of size 1 and 2
    v_list = []
    for j in range(m):
        for k in range(p):
            v_list.append([(j, k)])
    for j1 in range(m):
        for k1 in range(p):
            for j2 in range(j1, m):
                for k2 in range(p):
                    if j1 == j2:
                        if k1 < k2:
                            v_list.append([(j1, k1), (j2, k2)])
                    else:
                        v_list.append([(j1, k1), (j2, k2)])
    
    # Generate W supports of size 1 and 2
    w_list = []
    for k in range(p):
        for i in range(n):
            w_list.append([(k, i)])
    for k1 in range(p):
        for i1 in range(n):
            for k2 in range(k1, p):
                for i2 in range(n):
                    if k1 == k2:
                        if i1 < i2:
                            w_list.append([(k1, i1), (k2, i2)])
                    else:
                        w_list.append([(k1, i1), (k2, i2)])
    
    # Now combine: for each (u, v, w), compute the mask
    # This is O(|u_list| * |v_list| * |w_list|) which could be large.
    # |u_list| ~ 6 + 18 = 24
    # |v_list| ~ 24 + 288 = 312  (too many)
    # Let's limit V and W supports to size 1 only for the first pass.
    
    # Simplified: only size-1 supports for all three factors
    # This gives us the 48 basic terms, each covering exactly one (i,j,k).
    
    # That's not useful for set cover since we'd need all 48.
    
    # Let's try: U size 1, V size 1, W size 2
    # U: (i,j), V: (j,k), W: {(k,i1),(k,i2)} where i1 != i2
    # Covers: (i,j,k) if i1==i or i2==i, i.e., covers (i,j,k) for both i values? No...
    # W support {(k,i1),(k,i2)} means the term has W[k*n+i1]=1 and W[k*n+i2]=1
    # So it contributes to (i1,j,k) and (i2,j,k)
    
    # Let's just do a focused search: try to find a set of <= 30 terms
    # whose XOR gives the all-ones mask, using greedy set cover.
    
    # For tractability, let's precompute a moderate set of candidate terms.
    
    candidate_terms = []
    
    # Type 1: U={(i,j)}, V={(j,k)}, W={(k,i)} - covers single point
    for i in range(n):
        for j in range(m):
            for k in range(p):
                mask = 0
                mask |= (1 << (k * n + i * m + j))  # bit for (i,j,k)
                U_dict = {(i, j): 1}
                V_dict = {(j, k): 1}
                W_dict = {(k, i): 1}
                candidate_terms.append((mask, U_dict, V_dict, W_dict))
    
    # Type 2: U={(i,j)}, V={(j,k1),(j,k2)}, W={(k1,i),(k2,i)} - covers two points
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1 + 1, p):
                    mask = 0
                    mask |= (1 << (k1 * n + i * m + j))
                    mask |= (1 << (k2 * n + i * m + j))
                    U_dict = {(i, j): 1}
                    V_dict = {(j, k1): 1, (j, k2): 1}
                    W_dict = {(k1, i): 1, (k2, i): 1}
                    candidate_terms.append((mask, U_dict, V_dict, W_dict))
    
    # Type 3: U={(i1,j),(i2,j)}, V={(j,k)}, W={(k,i1),(k,i2)} - covers two points
    for j in range(m):
        for k in range(p):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    mask = 0
                    mask |= (1 << (k * n + i1 * m + j))
                    mask |= (1 << (k * n + i2 * m + j))
                    U_dict = {(i1, j): 1, (i2, j): 1}
                    V_dict = {(j, k): 1}
                    W_dict = {(k, i1): 1, (k, i2): 1}
                    candidate_terms.append((mask, U_dict, V_dict, W_dict))
    
    # Type 4: U={(i,j1),(i,j2)}, V={(j1,k),(j2,k)}, W={(k,i)} - covers two points
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    mask = 0
                    mask |= (1 << (k * n + i * m + j1))
                    mask |= (1 << (k * n + i * m + j2))
                    U_dict = {(i, j1): 1, (i, j2): 1}
                    V_dict = {(j1, k): 1, (j2, k): 1}
                    W_dict = {(k, i): 1}
                    candidate_terms.append((mask, U_dict, V_dict, W_dict))
    
    # Type 5: Strassen-like 2x2x2 patterns (7 terms per block)
    # These have more complex supports. Let's add them explicitly.
    # For each 2x2x2 sub-block at (i_base, j_base, k_base), add the 7 Strassen terms.
    for i_base in range(0, n, 2):
        for j_base in range(0, m, 2):
            for k_base in range(0, p, 2):
                if i_base + 1 >= n or j_base + 1 >= m or k_base + 1 >= p:
                    continue
                # Add the 7 Strassen terms for this 2x2x2 block
                for r in range(7):
                    mask = 0
                    U_dict = {}
                    V_dict = {}
                    W_dict = {}
                    for (di, dj), val in _STRASSEN_U[r].items():
                        i, j = i_base + di, j_base + dj
                        U_dict[(i, j)] = val
                    for (dj, dk), val in _STRASSEN_V[r].items():
                        j, k = j_base + dj, k_base + dk
                        V_dict[(j, k)] = val
                    for (dk, di), val in _STRASSEN_W[r].items():
                        k, i = k_base + dk, i_base + di
                        W_dict[(k, i)] = val
                    # Compute mask: (i,j,k) covered if U[i,j]*V[j,k]*W[k,i] != 0
                    for (i, j) in U_dict:
                        for (j2, k) in V_dict:
                            if j == j2:
                                for (k2, i2) in W_dict:
                                    if k == k2 and i2 == i:
                                        mask |= (1 << (k * n + i * m + j))
                    candidate_terms.append((mask, U_dict, V_dict, W_dict))
    
    if not candidate_terms:
        return None
    
    # Target: all-ones mask (48 bits set)
    target_mask = (1 << (n * m * p)) - 1
    
    # Greedy set cover: repeatedly pick the term that covers the most uncovered bits
    remaining = target_mask
    selected = []
    
    while remaining != 0:
        if time.time() > deadline:
            return None
        
        best_gain = 0
        best_idx = -1
        for idx, (mask, _, _, _) in enumerate(candidate_terms):
            gain = mask & remaining
            popcount = bin(gain).count('1')
            if popcount > best_gain:
                best_gain = popcount
                best_idx = idx
        
        if best_idx == -1 or best_gain == 0:
            return None  # Can't cover remaining bits
        
        selected.append(candidate_terms[best_idx])
        remaining &= ~candidate_terms[best_idx][0]
    
    # Now we have a set of terms whose XOR covers all bits.
    # But we need to find scalar coefficients such that the sum equals 1 everywhere.
    # This is a linear system: for each (i,j,k), sum of coeff[r] * prod = 1.
    # Over reals, this is a system of 48 equations in len(selected) unknowns.
    # If len(selected) <= 48, we can solve it.
    
    num_terms = len(selected)
    if num_terms > 48:
        # Too many terms; the linear system is underdetermined.
        # We could try to find a solution, but it's complex.
        # For now, if we found a small set, proceed; otherwise return None.
        if num_terms > 40:
            return None
    
    # Build the linear system: A[x] = b where A[eq, var] = product of values
    # eq: (i,j,k) index, var: term index
    # A[eq, var] = U_var[i,j] * V_var[j,k] * W_var[k,i]
    # b[eq] = 1
    
    import numpy as np
    
    A = np.zeros((n * m * p, num_terms))
    for eq_idx in range(n * m * p):
        k = eq_idx // (n * m)
        rem = eq_idx % (n * m)
        i = rem // m
        j = rem % m
        for var_idx in range(num_terms):
            _, U_dict, V_dict, W_dict = selected[var_idx]
            u_val = U_dict.get((i, j), 0)
            v_val = V_dict.get((j, k), 0)
            w_val = W_dict.get((k, i), 0)
            A[eq_idx, var_idx] = u_val * v_val * w_val
    
    b = np.ones(n * m * p)
    
    # Solve the linear system
    try:
        if num_terms == n * m * p:
            coeffs = np.linalg.solve(A, b)
        elif num_terms < n * m * p:
            # Underdetermined: use least squares
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if the solution is exact
            if not np.allclose(A @ coeffs, b, atol=1e-9):
                return None
        else:
            # Overdetermined: use least squares
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            if not np.allclose(A @ coeffs, b, atol=1e-9):
                return None
    except np.linalg.LinAlgError:
        return None
    
    # Round coefficients to nearest integer or half-integer
    coeffs_rounded = np.round(coeffs * 2) / 2
    
    # Verify the rounded solution
    if not np.allclose(A @ coeffs_rounded, b, atol=1e-9):
        # Try rounding to integers
        coeffs_rounded = np.round(coeffs)
        if not np.allclose(A @ coeffs_rounded, b, atol=1e-9):
            return None
    
    # Build the final U, V, W matrices
    U_mat = [[0] * num_terms for _ in range(n * m)]
    V_mat = [[0] * num_terms for _ in range(m * p)]
    W_mat = [[0] * num_terms for _ in range(p * n)]
    
    for var_idx in range(num_terms):
        _, U_dict, V_dict, W_dict = selected[var_idx]
        coeff = coeffs_rounded[var_idx]
        for (i, j), val in U_dict.items():
            U_mat[i * m + j][var_idx] = val * coeff
        for (j, k), val in V_dict.items():
            V_mat[j * p + k][var_idx] = val
        for (k, i), val in W_dict.items():
            W_mat[k * n + i][var_idx] = val
    
    # Validate
    result = {"U": U_mat, "V": V_mat, "W": W_mat}
    if _validate(U_mat, V_mat, W_mat, n, m, p):
        return result
    
    return None


def _search_rank2_block(n, m, p, deadline, rng, max_iter=200000):
    """Search for a rank-2 decomposition of one 2x1x2 all-ones block.

    T[i, j=2, k] = 1 for i in {0,1}, k in {0,1}.  A rank-1 solution exists
    (the outer product).  We probe whether a rank-2 (or confirm rank-1)
    identity is needed; this is bounded and only used if it beats rank 1,
    which it cannot, so it is a fast no-op guard kept for structural parity.
    """
    if time.time() > deadline:
        return None
    return None


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115  # 150s kill, keep 35s safety margin

    rng = random.Random(12345)

    # Primary valid decomposition (rank 32).  Verify once.
    best = _block_decomposition(n, m, p)
    if not _validate(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    # Refinement: GF(2) set-cover heuristic (only ever accepted if valid & lower).
    if time.time() < deadline:
        gf2_result = _gf2_set_cover(n, m, p, deadline, rng)
        if gf2_result is not None:
            gf2_rank = len(gf2_result["U"][0])
            current_rank = len(best["U"][0])
            if gf2_rank < current_rank and _validate(gf2_result["U"], gf2_result["V"], gf2_result["W"], n, m, p):
                best = gf2_result

    if time.time() < deadline:
        rb = _search_rank2_block(n, m, p, deadline, rng)
        if rb is not None and _validate(rb["U"], rb["V"], rb["W"], n, m, p):
            best = rb

    return best

# EVOLVE-BLOCK-END
