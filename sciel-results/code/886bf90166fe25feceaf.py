import numpy as np
import time

DIMENSIONS = [2, 2, 14]

def _verify(U, V, W, n=2, m=2, p=14, tol=1e-9):
    """Check that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] equals the
    multiplication tensor T[i,j,k] = 1 if (i==j) else 0  ... actually the
    matrix multiplication tensor: (A B)[i,k] = sum_j A[i,j] B[j,k].
    The tensor T[i,j,k] has T[i,j,k]=1 iff the product contributes, i.e.
    T[i,j,k] = 1 for all i,j,k? No.

    The multiplication tensor: T[i,j,k] = delta_{i,?}... The standard
    matmul tensor M_{n,m,p} has entries M[i,j,k] = 1 if the (i,j) of A
    multiplies (j,k) of B, i.e. M[i,j,k] = 1 for all valid i,j,k? No:
    M[i,j,k] = 1 when the index j of A's column equals j of B's row.
    Actually M[i,j,k] = delta_{j,j} which is always 1 for the shared index.
    The tensor is M[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    Wait no — the shared index is the middle one. M[i,j,k] = 1 iff we're
    multiplying A[i,j] * B[j,k], so the entry is 1 for every (i,j,k) triple
    where j is the shared index. Since j ranges over all m values and there's
    no constraint, M[i,j,k] = 1 for ALL i,j,k. That can't be right because
    then rank would be small.

    Correct: the matmul tensor M_{n,m,p} has M[i,j,k] = 1 if and only if the
    column index of A equals the row index of B, i.e. the middle index j is
    shared. In tensor form, M[i,j,k] = delta_{j (A col), j (B row)}. Since
    both use the same j, M[i,j,k] = 1 for all i,j,k. Hmm, that gives a tensor
    of all ones, rank 1. That's wrong.

    The right definition: M[i,j,k] = 1 when the (i,j) entry of A and (j,k)
    entry of B are multiplied. The index j appears in BOTH A (as column) and
    B (as row). So for a fixed triple (i,j,k), the product A[i,j]*B[j,k]
    contributes to output C[i,k]. The tensor entry M[i,j,k] = 1 (the
    coefficient of A[i,j]*B[j,k] in the product) is 1 for every (i,j,k).
    So M is the all-ones tensor of shape (n,m,p)?? No — the tensor is indexed
    so that the contraction over the shared index is implicit. The tensor
    M[i,j,k] = 1 for all i,j,k is indeed the matmul tensor in the "un-
    contracted" form where the summation index is explicit. Yes! The matmul
    tensor with explicit summation index is M[i,j,k] = 1 for all i in [n],
    j in [m], k in [p]. Its rank is the bilinear complexity.

    So T[i,j,k] = 1 for all i,j,k.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    # Build T via einsum: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # U indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    # Expected: all ones
    expected = np.ones((n, m, p))
    return bool(np.allclose(T, expected, atol=tol))

def _strassen_49():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    num_blocks = p // 2
    rank = num_blocks * 7
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    for t in range(num_blocks):
        base = t * 7
        ve = 2 * t
        vf = 2 * t + 1
        vg = 14 + 2 * t
        vh = 14 + 2 * t + 1
        w_i = 4 * t
        w_j = 4 * t + 2
        w_k = 4 * t + 1
        w_l = 4 * t + 3
        U[0, base] = 1; U[3, base] = 1
        V[ve, base] = 1; V[vh, base] = 1
        W[w_i, base] = 1; W[w_l, base] = 1
        U[2, base+1] = 1; U[3, base+1] = 1
        V[ve, base+1] = 1
        W[w_k, base+1] = 1; W[w_l, base+1] = -1
        U[0, base+2] = 1
        V[vf, base+2] = 1; V[vh, base+2] = -1
        W[w_j, base+2] = 1; W[w_l, base+2] = 1
        U[3, base+3] = 1
        V[vg, base+3] = 1; V[ve, base+3] = -1
        W[w_i, base+3] = 1; W[w_k, base+3] = 1
        U[0, base+4] = 1; U[1, base+4] = 1
        V[vh, base+4] = 1
        W[w_i, base+4] = -1; W[w_j, base+4] = 1
        U[2, base+5] = 1; U[0, base+5] = -1
        V[ve, base+5] = 1; V[vf, base+5] = 1
        W[w_l, base+5] = 1
        U[1, base+6] = 1; U[3, base+6] = -1
        V[vg, base+6] = 1; V[vh, base+6] = 1
        W[w_i, base+6] = 1
    return U, V, W

def _to_list(arr):
    out = []
    for row in arr:
        r = []
        for x in row:
            xi = int(round(x))
            if abs(x - xi) < 1e-9:
                r.append(xi)
            else:
                r.append(float(x))
        out.append(r)
    return out

def construct():
    t_start = time.time()
    deadline = 150.0 - 15.0  # safety margin: return by 135s
    n, m, p = DIMENSIONS

    # Primary: verified rank-49 Strassen
    U49, V49, W49 = _strassen_49()
    assert _verify(U49, V49, W49, n, m, p), "rank-49 baseline failed verification"
    best = (49, U49, V49, W49)

    # Attempt a bounded search for rank 48 within remaining budget.
    # Strategy: start from the rank-49 factors, zero out one rank slot, and
    # try to repair via a local exact solve on the affected block. We only
    # accept a result if it verifies exactly.
    remaining_budget = lambda: deadline - (time.time() - t_start)

    if remaining_budget() > 5.0:
        # Try to merge two Strassen blocks' worth of terms: attempt rank 48
        # by solving a small linear system that reuses structure.
        # We attempt a direct rank-48 construction: take the 49-term solution
        # and see if one term is redundant given the others (linear
        # dependence in the tensor space). If a term lies in the span of the
        # others, we can drop it.
        rank49 = 49
        # Build the 49 rank-1 tensors as vectors in R^{n*m*p}
        nm, mp, pn = n*m, m*p, p*n
        vecs = []
        for r in range(rank49):
            v = np.einsum('ir,jr,kr->ijk',
                          U49[:, r].reshape(n, m),
                          V49[:, r].reshape(m, p),
                          W49[:, r].reshape(p, n)).reshape(-1)
            vecs.append(v)
        M = np.stack(vecs, axis=1)  # shape (56, 49)
        # Check rank of M; if < 49, some term is redundant.
        try:
            rr = np.linalg.matrix_rank(M, tol=1e-8)
        except Exception:
            rr = 49
        # If the 49 vectors span a space of dimension < 49, we can drop a
        # dependent one. (They all equal the all-ones tensor, so rank(M)=1!
        # This means the "vectors" are all identical, which is expected since
        # each rank-1 term is a different tensor but they all sum to the same
        # target; individually they are NOT all equal. Re-check: each term is
        # a different rank-1 tensor, so they are different vectors. The sum
        # is the all-ones tensor. So M's columns are 49 different vectors whose
        # sum is the all-ones vector. matrix_rank(M) could be up to 49.)
        # If rr < 49, drop a column in the null space.
        if rr < 49:
            # Find a null-space direction to identify a redundant column.
            # Simpler: greedily remove columns while keeping span of remaining
            # columns able to represent the target = sum of all columns.
            target = M.sum(axis=1)
            kept = list(range(49))
            for c in list(kept):
                if time.time() - t_start > deadline:
                    break
                cand = [x for x in kept if x != c]
                if len(cand) == 0:
                    continue
                Mc = M[:, cand]
                # Can target be expressed as a combination of Mc? Solve least
                # squares; if residual ~0, column c is redundant.
                try:
                    coeffs, res, *_ = np.linalg.lstsq(Mc, target, rcond=None)
                    recon = Mc @ coeffs
                    if np.allclose(recon, target, atol=1e-8):
                        kept = cand
                except Exception:
                    pass
            if len(kept) < 49:
                r48 = len(kept)
                U_new = U49[:, kept]
                V_new = V49[:, kept]
                W_new = W49[:, kept]
                if _verify(U_new, V_new, W_new, n, m, p):
                    best = (r48, U_new, V_new, W_new)

    r, U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
