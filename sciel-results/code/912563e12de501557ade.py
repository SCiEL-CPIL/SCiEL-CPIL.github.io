"""
Rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.

Strategy:
  1. Build a known-good construction using 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5
     plus schoolbook for j=4. Rank = 84 + 24 = 108.
  2. Attempt to reduce the rank by finding linear dependencies among the tensor terms
     using rational Gaussian elimination.
  3. Verify the result exactly with random integer matrices.
"""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)


def _schoolbook():
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, rank):
    """Verify exact decomposition using random integer matrices."""
    n, m, p = DIMENSIONS
    for _ in range(8):
        if remaining() <= 2:
            return False
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u_val = U[i * m + j][c]
                        if u_val == 0:
                            continue
                        acc += u_val * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True


def _winograd_2x2x2_terms():
    """
    Returns 7 rank-1 terms for 2x2x2 matmul.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C00 = p1 - p4 + p5 - p7
    C01 = p3 + p4 + p5 + p7
    C10 = p2 + p4 - p6 + p7
    C11 = p1 - p2 + p6 - p7
    """
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 1]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [-1, 1, 1, 0]),
    ]


def _embed_block(terms, i_base, j_base, k_base, i_size, j_size, k_size, n, m, p):
    """Embed a block's terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)

        for a in range(i_size):
            for b in range(j_size):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * j_size + b]

        for b in range(j_size):
            for c in range(k_size):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * k_size + c]

        for c in range(k_size):
            for a in range(i_size):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * i_size + a]

        full_terms.append((ufull, vfull, wfull))
    return full_terms


def _build_from_terms(terms_U, terms_V, terms_W, n, m, p):
    """Build U, V, W matrices from lists of term vectors."""
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]

    return {"U": U, "V": V, "W": W, "rank": rank}


def _strategy_winograd_schoolbook():
    """
    Known-good construction: 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5
    plus schoolbook for j=4.
    Rank = 12 blocks * 7 + 24 schoolbook = 84 + 24 = 108.
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2_terms()

    terms_U = []
    terms_V = []
    terms_W = []

    # 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5
    for ib in range(2):
        for jb in range(2):
            for kb in range(3):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb

                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])

    # Schoolbook for j=4 (all i, k)
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)

    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)


def _term_to_tensor_vector(u, v, w, n, m, p):
    """
    Convert a rank-1 term (u, v, w) into a vector in the tensor product space.
    The tensor T has entries T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i].
    We represent this as a flat vector of length n*m*p.
    """
    vec = [0] * (n * m * p)
    for i in range(n):
        for j in range(m):
            u_val = u[i * m + j]
            if u_val == 0:
                continue
            for k in range(p):
                v_val = v[j * p + k]
                if v_val == 0:
                    continue
                w_val = w[k * n + i]
                if w_val == 0:
                    continue
                vec[i * m * p + j * p + k] = u_val * v_val * w_val
    return vec


def _target_in_span(indices, term_vectors, target, n, m, p):
    """
    Check if the target tensor vector is in the span of the given term vectors.
    Uses Gaussian elimination over rationals.
    """
    if remaining() <= 2:
        return False

    dim = n * m * p
    # Build the matrix with term vectors as rows
    rows = []
    for idx in indices:
        rows.append([Fraction(x) for x in term_vectors[idx]])

    # Add target as the last row
    rows.append([Fraction(x) for x in target])

    # Gaussian elimination
    n_rows = len(rows)
    n_cols = dim
    rank = 0
    pivot_cols = []

    for col in range(n_cols):
        if rank >= n_rows - 1:
            break
        if remaining() <= 2:
            return False

        # Find pivot
        pivot = -1
        for row in range(rank, n_rows):
            if rows[row][col] != 0:
                pivot = row
                break

        if pivot == -1:
            continue

        # Swap
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_cols.append(col)

        # Eliminate
        pivot_val = rows[rank][col]
        for row in range(n_rows):
            if row == rank:
                continue
            factor = rows[row][col] / pivot_val
            if factor != 0:
                for c in range(col, n_cols):
                    rows[row][c] -= factor * rows[rank][c]

        rank += 1

    # Check if the last row (target) is all zeros after elimination
    # (i.e., it was in the span of the other rows)
    for c in range(n_cols):
        if rows[n_rows - 1][c] != 0:
            return False
    return True


def _find_minimal_spanning_subset(pool, n, m, p, max_rank=None):
    """
    Find a minimal subset of terms from the pool that spans the matrix-multiplication tensor.
    Uses greedy Gaussian elimination over rationals.

    Returns a list of indices into pool.
    """
    # First, compute the tensor vector for each term
    term_vectors = []
    for idx, (u, v, w) in enumerate(pool):
        if remaining() <= 5:
            break
        term_vectors.append(_term_to_tensor_vector(u, v, w, n, m, p))

    # The target tensor is the matrix-multiplication tensor:
    # T[i,j,k] = 1 if the term (i,j,k) is part of the multiplication, else 0.
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # Wait, no. The tensor for matrix multiplication C = A*B has:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]

    # So the target vector is all 1s of length n*m*p.
    target = [1] * (n * m * p)

    # Greedy selection: pick terms that reduce the residual the most.
    # Start with the full set, then try to remove terms.
    # Actually, let's do forward selection: start with empty set, add terms that
    # are linearly independent and help span the target.

    # Better approach: use the known Winograd+schoolbook construction as a starting
    # point (rank 108), then try to reduce it further by checking if any term can
    # be expressed as a linear combination of the others.

    # Let's use the Winograd+schoolbook terms as our initial set.
    # Identify which pool indices correspond to the Winograd+schoolbook construction.
    # Winograd blocks: indices 120 to 120+84-1 = 120 to 203
    # Schoolbook j=4: indices 204 to 204+24-1 = 204 to 227

    initial_indices = list(range(120, 204)) + list(range(204, 228))

    # Now try to reduce: for each term in the initial set, check if it can be
    # expressed as a linear combination of the remaining terms.
    # This is expensive, so we'll do a simpler version:
    # Check for duplicate or linearly dependent pairs first.

    selected = list(initial_indices)

    # Try to remove terms one by one
    changed = True
    while changed and remaining() > 10:
        changed = False
        for idx_to_remove in range(len(selected) - 1, -1, -1):
            if remaining() <= 10:
                break

            # Try removing selected[idx_to_remove]
            test_selected = selected[:idx_to_remove] + selected[idx_to_remove + 1:]

            # Check if the target is still in the span
            if _target_in_span(test_selected, term_vectors, target, n, m, p):
                selected = test_selected
                changed = True
                if max_rank and len(selected) <= max_rank:
                    break

    return selected


def _strategy_linear_reduction():
    """
    Strategy: generate a pool of terms, then use linear algebra to find
    a minimal spanning subset.
    """
    n, m, p = DIMENSIONS
    pool = _generate_term_pool(n, m, p)

    if remaining() <= 20:
        return None

    try:
        selected = _find_minimal_spanning_subset(pool, n, m, p)
        if selected is None or len(selected) == 0:
            return None

        # Build the result from selected terms
        terms_U = [pool[i][0] for i in selected]
        terms_V = [pool[i][1] for i in selected]
        terms_W = [pool[i][2] for i in selected]

        result = _build_from_terms(terms_U, terms_V, terms_W, n, m, p)
        return result
    except Exception:
        return None


def _generate_term_pool(n, m, p):
    """
    Generate a pool of candidate rank-1 terms:
      - 120 schoolbook terms
      - 84 Winograd 2x2x2 block terms (i:0-3, j:0-3, k:0-5)
      - 24 schoolbook terms for j=4
    Total pool size: 120 + 84 + 24 = 228 terms.
    Each term is a triple (u_vec, v_vec, w_vec).
    """
    pool = []

    # Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                pool.append((ufull, vfull, wfull))

    # Winograd 2x2x2 block terms
    winograd_222 = _winograd_2x2x2_terms()
    for ib in range(2):
        for jb in range(2):
            for kb in range(3):
                embedded = _embed_block(winograd_222, 2*ib, 2*jb, 2*kb, 2, 2, 2, n, m, p)
                pool.extend(embedded)

    # Schoolbook for j=4
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            pool.append((ufull, vfull, wfull))

    return pool


def construct():
    n, m, p = DIMENSIONS

    candidates = []

    # Strategy 1: Linear reduction (attempt to find rank < 108)
    if remaining() > 30:
        try:
            cand = _strategy_linear_reduction()
            if cand is not None:
                candidates.append(cand)
        except Exception:
            pass

    # Strategy 2: Known-good Winograd + schoolbook (rank 108)
    if remaining() > 10:
        try:
            candidates.append(_strategy_winograd_schoolbook())
        except Exception:
            pass

    # Strategy 3: Schoolbook (rank 120)
    candidates.append(_schoolbook())

    # Verify and pick best
    best = None
    for cand in candidates:
        if remaining() <= 2:
            break
        rank = cand["rank"]
        U, V, W = cand["U"], cand["V"], cand["W"]
        if _verify(U, V, W, rank):
            if best is None or rank < best["rank"]:
                best = cand

    if best is None:
        best = candidates[-1]  # schoolbook

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
