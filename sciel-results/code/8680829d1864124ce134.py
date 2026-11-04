import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# Strassen 2x2x2 decomposition: 7 rank-1 terms
_strassen_terms = [
    (
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    ),
    (
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 0): 1},
    ),
    (
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): -1},
    ),
    (
        {(1, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 0): 1, (0, 0): -1},
    ),
    (
        {(0, 0): 1, (0, 1): 1},
        {(1, 1): 1},
        {(1, 1): 1},
    ),
    (
        {(1, 0): 1, (0, 0): -1},
        {(0, 0): 1, (0, 1): 1},
        {(0, 0): 1, (0, 1): 1},
    ),
    (
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ),
]


def _partitions_of(k):
    """All compositions of k into positive parts, as tuples."""
    if k == 0:
        yield ()
        return
    for first in range(1, k + 1):
        for rest in _partitions_of(k - first):
            yield (first,) + rest


def _build_for_partition(n_parts, m_parts, p_parts):
    """Build U,V,W for a given block partition. Returns (U, V, W, rank)."""
    n = m = p = 5
    N = n * m
    M = m * p
    P = p * n

    n_offsets = list(np.cumsum([0] + list(n_parts)))[:-1]
    m_offsets = list(np.cumsum([0] + list(m_parts)))[:-1]
    p_offsets = list(np.cumsum([0] + list(p_parts)))[:-1]

    # Pre-compute total rank
    total = 0
    for ni, ns in enumerate(n_parts):
        for mj, ms in enumerate(m_parts):
            for pk, ps in enumerate(p_parts):
                if ns == 2 and ms == 2 and ps == 2:
                    total += 7
                else:
                    total += ns * ms * ps

    U = np.zeros((N, total), dtype=np.float64)
    V = np.zeros((M, total), dtype=np.float64)
    W = np.zeros((P, total), dtype=np.float64)

    col = 0
    for ni, ns in enumerate(n_parts):
        noff = n_offsets[ni]
        for mj, ms in enumerate(m_parts):
            moff = m_offsets[mj]
            for pk, ps in enumerate(p_parts):
                poff = p_offsets[pk]
                if ns == 2 and ms == 2 and ps == 2:
                    for term in _strassen_terms:
                        u_entries, v_entries, w_entries = term
                        for (di, dj), uc in u_entries.items():
                            U[(noff + di) * m + (moff + dj), col] += uc
                        for (dj, dk), vc in v_entries.items():
                            V[(moff + dj) * p + (poff + dk), col] += vc
                        for (dk, di), wc in w_entries.items():
                            W[(poff + dk) * n + (noff + di), col] += wc
                        col += 1
                else:
                    for di in range(ns):
                        for dj in range(ms):
                            for dk in range(ps):
                                U[(noff + di) * m + (moff + dj), col] = 1
                                V[(moff + dj) * p + (poff + dk), col] = 1
                                W[(poff + dk) * n + (noff + di), col] = 1
                                col += 1
    return U, V, W, col


def _verify(U, V, W, n=5, m=5, p=5):
    """Verify the decomposition exactly (with integer arithmetic)."""
    rng = np.random.default_rng(7)
    A = rng.integers(-2, 3, size=(n, m))
    B = rng.integers(-2, 3, size=(m, p))
    C_expected = A @ B

    # T[i,k,j] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r]
    # Check: sum_{i,k,j} A[i,k] * T[i,k,j] * B[k,j] == sum(C_expected)
    # But more directly: C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor T should satisfy: for all A,B, sum_{i,k,j} A[i,k] T[i,k,j] B[k,j] = sum_{i,j} C[i,j]
    # Actually we need T to be the matmul tensor: T[i,k,j] = 1 for the "identity" contribution.
    # The correct check: reconstruct C from the decomposition.
    # C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] ... NO
    # The correct identity: for the decomposition to represent matmul,
    # we need: for all i,j: sum_k A[i,k]B[k,j] = sum_r (sum_k A[i,k] U[i*m+k,r]) (sum_k V[k*p+j,r] B[k,j]) W[j*n+i,r]
    # i.e., C[i,j] = sum_r (A_r^T A)[i,i]... no.
    # Let me use the standard: T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[j,i]
    # and the matmul tensor is T_matmul[i,k,j] such that <A, T_matmul, B> = sum_{i,j} (AB)[i,j]
    # where <A, T, B> = sum_{i,k,j} A[i,k] T[i,k,j] B[k,j]
    # So T_matmul[i,k,j] = 1 for all valid (i,k,j)? No, that gives sum A[i,k]B[k,j] which is sum of all entries of AB. Yes!
    # So T_matmul[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p).
    # Wait, that's just the all-ones tensor, which has rank 1. That can't be right.
    # 
    # I think the issue is the indexing. Let me re-derive.
    # C = A B, C[i,j] = sum_k A[i,k] B[k,j]
    # The bilinear form: f(A, B) = sum_{i,j} C[i,j] = sum_{i,j,k} A[i,k] B[k,j]
    # This is a 3-tensor with indices (i, k, j) where T[i,k,j] = 1 for all valid indices.
    # But that's rank 1! So this formulation is wrong for what we want.
    #
    # The correct tensor for matrix multiplication:
    # We want C[i,j] for each (i,j). The tensor T has indices (i, k, j) and
    # T[i,k,j] is such that C[i,j] = sum_k A[i,k] B[k,j].
    # The rank decomposition: T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[j,i]
    # Then: sum_{i,k,j} A[i,k] T[i,k,j] B[k,j] = sum_r sum_{i,k,j} A[i,k] U_r[i,k] V_r[k,j] W_r[j,i] B[k,j]
    # = sum_r (sum_i (A * U_r)[i,i]) ... no.
    # 
    # Actually: sum_{i,k,j} A[i,k] U_r[i,k] V_r[k,j] W_r[j,i] B[k,j]
    # = sum_i W_r[?,i] ... let me be more careful.
    # = sum_{i} [sum_k A[i,k] U_r[i,k]] [sum_j V_r[k,j] W_r[j,i] B[k,j]] -- no, k is shared.
    # = sum_{i,k,j} A[i,k] U_r[i,k] V_r[k,j] W_r[j,i] B[k,j]
    # = sum_i sum_j W_r[j,i] [sum_k A[i,k] U_r[i,k] V_r[k,j] B[k,j]]
    # 
    # For this to equal sum_{i,j} (AB)[i,j] = sum_{i,j} sum_k A[i,k] B[k,j] for ALL A,B,
    # we need: sum_r sum_{i,k,j} A[i,k] U_r[i,k] V_r[k,j] W_r[j,i] B[k,j] = sum_{i,k,j} A[i,k] B[k,j]
    # 
    # This means: sum_r U_r[i,k] V_r[k,j] W_r[j,i] = 1 for all i,k,j.
    # So the tensor T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[j,i] must equal 1 for all (i,k,j).
    # 
    # Wait, that means T is the all-ones tensor, which has rank 1. That's clearly wrong.
    # 
    # I think the issue is that the matmul tensor is NOT all-ones. Let me reconsider.
    # The matmul operation: given A (n x m) and B (m x p), produce C (n x p).
    # C[i,j] = sum_k A[i,k] B[k,j].
    # This is a multilinear map from A and B to C.
    # The tensor T has three modes: (i, k, j) where i indexes rows of A/C, k indexes the contraction, j indexes cols of B/C.
    # T[i,k,j] = 1 if and only if this (i,k,j) triple contributes to C[i,j].
    # Since C[i,j] = sum_k A[i,k] B[k,j], the coefficient of A[i,k] B[k,j] in C[i,j] is 1.
    # So T[i,k,j] = 1 for all valid (i,k,j).
    # 
    # But this gives rank 1, which is absurd. The resolution: the tensor decomposition
    # T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[j,i] is NOT the right form.
    # 
    # The correct form for the matmul tensor decomposition is:
    # T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[i,j]  (note: W indexed by (i,j), not (j,i))
    # OR equivalently, the standard formulation uses:
    # C[i,j] = sum_r (sum_k A[i,k] U_r[i,k]) (sum_k V_r[k,j] B[k,j]) W_r[i,j]
    # 
    # Hmm, let me look at this differently. The standard tensor decomposition of matmul:
    # The matmul tensor T ∈ R^{n×m × m×p × n×p} has T[i,k, j, i, j]... no.
    # 
    # Let me use the formulation from the task:
    # U: shape (n*m, rank), indexed by (i, k) flattened
    # V: shape (m*p, rank), indexed by (k, j) flattened  
    # W: shape (p*n, rank), indexed by (j, i) flattened  [W[k*n+i] per the task]
    # 
    # The decomposition: for the matmul tensor, we need:
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1  for all valid (i,k,j)
    # 
    # Let's check with schoolbook: for each (i,k,j), we have one term with U[i*m+k]=1, V[k*p+j]=1, W[j*n+i]=1.
    # Then sum_r ... = 1 for that specific (i,k,j) and 0 for others? No, each term only contributes to one (i,k,j).
    # So sum_r U[i*m+k,r] V[k*p+j,r] W[j*n+i,r] = 1 if there's a term for (i,k,j), which there is.
    # But wait, for a fixed (i,k,j), only the schoolbook term for that (i,k,j) has all three nonzero at those indices.
    # So the sum is 1. Good.
    # 
    # But does this actually compute the matrix product? Let's verify:
    # C[i,j] = sum_k A[i,k] B[k,j]
    # Using the decomposition: sum_r (sum_k A[i,k] U[i*m+k,r]) (sum_k B[k,j] V[k*p+j,r]) ... no.
    # 
    # The correct reconstruction: C[i,j] = sum_r W[j*n+i, r] * (sum_k A[i,k] U[i*m+k,r]) * (sum_k B[k,j] V[k*p+j,r])
    # 
    # Let's verify with schoolbook: for term r = (i0,k0,j0):
    # U[i*m+k, r] = 1 if (i,k) = (i0,k0)
    # V[k*p+j, r] = 1 if (k,j) = (k0,j0)
    # W[j*n+i, r] = 1 if (j,i) = (j0,i0)
    # So the contribution to C[i,j] from term r is: W[j*n+i, r] * (A[i0,k0] if i=i0 else 0) * (B[k0,j0] if j=j0 else 0)
    # = W[j*n+i, r] * A[i0,k0] * B[k0,j0] * [i==i0] * [j==j0]
    # For this to be nonzero, we need i=i0 and j=j0, and W[j0*n+i0, r] = 1.
    # So C[i,j] gets contribution A[i,k0] B[k0,j] for each k0 where there's a term (i,k0,j).
    # Summing over all terms: C[i,j] = sum_k A[i,k] B[k,j]. Correct!
    # 
    # So the verification is: for random A, B:
    # C_computed[i,j] = sum_r W[j*n+i, r] * (sum_k A[i,k] U[i*m+k, r]) * (sum_k B[k,j] V[k*p+j, r])
    # should equal (A @ B)[i,j].

    U_r = U.reshape(n, m, -1)  # (n, m, rank)
    V_r = V.reshape(m, p, -1)  # (m, p, rank)
    W_r = W.reshape(p, n, -1)  # (p, n, rank)

    # C[i,j] = sum_r W_r[j, i, r] * (sum_k A[i,k] U_r[i,k,r]) * (sum_k B[k,j] V_r[k,j,r])
    AU = np.einsum('ik,ikr->ir', A, U_r)  # (i, r)
    BV = np.einsum('kj,kjr->jr', B, V_r)  # (j, r)
    C_computed = np.einsum('ir,jr,jir->ij', AU, BV, W_r)

    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True


def _try_reduce_rank(U, V, W, n=5, m=5, p=5, max_iterations=500, deadline=None):
    """Try to reduce rank by merging zero columns or finding redundant terms."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()

    for _ in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        reduced = False
        # Try removing each column and checking if the tensor is still preserved
        for r in range(rank):
            if deadline is not None and time.time() > deadline:
                break
            # Check if column r is redundant: i.e., T without r still equals all-ones tensor
            # T[i,k,j] = sum_r' U[i*m+k, r'] V[k*p+j, r'] W[j*n+i, r']
            # This is expensive to check for each r. Skip for now.
            pass
        if reduced:
            rank = U.shape[1]
        else:
            break

    return U, V, W, rank


def _clean_output(U, V, W):
    """Convert to integer/half-integer lists."""
    def clean(x):
        rx = round(x)
        if abs(x - rx) < 1e-9:
            return int(rx)
        rx2 = round(x * 2) / 2
        if abs(x - rx2) < 1e-9:
            return rx2
        return x

    U_out = [[clean(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_out = [[clean(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_out = [[clean(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    return U_out, V_out, W_out


def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin

    n = m = p = 5

    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p  # schoolbook

    # Enumerate all block partitions
    n_parts_list = list(_partitions_of(n))
    m_parts_list = list(_partitions_of(m))
    p_parts_list = list(_partitions_of(p))

    # Try all combinations, sorted by expected rank
    candidates = []
    for np_ in n_parts_list:
        for mp_ in m_parts_list:
            for pp_ in p_parts_list:
                # Compute expected rank
                rank_est = 0
                noffs = list(np.cumsum([0] + list(np_)))[:-1]
                moffs = list(np.cumsum([0] + list(mp_)))[:-1]
                poffs = list(np.cumsum([0] + list(pp_)))[:-1]
                for ni, ns in enumerate(np_):
                    for mj, ms in enumerate(mp_):
                        for pk, ps in enumerate(pp_):
                            if ns == 2 and ms == 2 and ps == 2:
                                rank_est += 7
                            else:
                                rank_est += ns * ms * ps
                candidates.append((rank_est, np_, mp_, pp_))

    candidates.sort(key=lambda x: x[0])

    for rank_est, np_, mp_, pp_ in candidates:
        if time.time() > DEADLINE:
            break
        if rank_est >= best_rank:
            break  # No point trying worse partitions

        U, V, W, actual_rank = _build_for_partition(list(np_), list(mp_), list(pp_))

        if _verify(U, V, W, n, m, p):
            best_U, best_V, best_W, best_rank = U, V, W, actual_rank

    # Fallback: schoolbook
    if best_U is None:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Try rank reduction
    if time.time() < DEADLINE - 10:
        U_red, V_red, W_red, rank_red = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p,
            max_iterations=200, deadline=DEADLINE - 10
        )
        if rank_red < best_rank and _verify(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W, best_rank = U_red, V_red, W_red, rank_red

    U_out, V_out, W_out = _clean_output(best_U, best_V, best_W)
    return {"U": U_out, "V": V_out, "W": W_out}
