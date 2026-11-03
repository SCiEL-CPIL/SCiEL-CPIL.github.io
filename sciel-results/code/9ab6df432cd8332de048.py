DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2_terms():
    """Return 7 rank-1 terms for Strassen's M_{2,2,2} decomposition."""
    return [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

def _build_terms(n, m, p, k_pairs, single_k, use_strassen=True):
    """Build rank-1 terms for 2x3 x 3x9 with given column partition."""
    n, m, p = 2, 3, 9
    terms = []
    strassen = _strassen_2x2x2_terms()
    
    # Strassen blocks on A[0:2, 0:2] x B[0:2, (k1,k2)]
    for (k1, k2) in k_pairs:
        for (u_d, v_d, w_d) in strassen:
            U_t = {i: c for i, c in u_d.items()}
            V_t = {}
            for (lj, lk), c in v_d.items():
                ak = k1 if lk == 0 else k2
                V_t[lj * p + ak] = c
            W_t = {}
            for (li, lk), c in w_d.items():
                ak = k1 if lk == 0 else k2
                W_t[ak * n + li] = c
            terms.append((U_t, V_t, W_t))
    
    # Single column schoolbook 2x2 x 2x1
    for i in range(2):
        for j in range(2):
            terms.append(({i * m + j: 1}, {j * p + single_k: 1}, {single_k * n + i: 1}))
    
    # A[0:2, 2] x B[2, 0:9] schoolbook 2x1 x 1x9
    for i in range(2):
        for k in range(p):
            terms.append(({i * m + 2: 1}, {2 * p + k: 1}, {k * n + i: 1}))
    
    return terms

def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U[idx][c] = coeff
        for idx, coeff in v.items():
            V[idx][c] = coeff
        for idx, coeff in w.items():
            W[idx][c] = coeff
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    T = {}
    for c in range(rank):
        u = {i: U[i][c] for i in range(n * m) if U[i][c] != 0}
        v = {j: V[j][c] for j in range(m * p) if V[j][c] != 0}
        w = {k: W[k][c] for k in range(p * n) if W[k][c] != 0}
        for i, ui in u.items():
            ii, jj = divmod(i, m)
            for j, vj in v.items():
                jj2, kk = divmod(j, p)
                if jj != jj2:
                    continue
                for k, wk in w.items():
                    kk2, ii2 = divmod(k, n)
                    if kk != kk2 or ii2 != ii:
                        continue
                    key = (i, j, k)
                    T[key] = T.get(key, 0.0) + ui * vj * wk
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T.get((i*m+j, j*p+k, k*n+i), 0.0) - 1.0) > tol:
                    return False
    for key, val in T.items():
        i, j, k = key
        ii, jj = divmod(i, m)
        jj2, kk = divmod(j, p)
        kk2, ii2 = divmod(k, n)
        if (jj != jj2 or kk != kk2 or ii2 != ii) and abs(val) > tol:
            return False
    return True

def _rank1_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _fast_reduce(terms, n, m, p, deadline, max_rounds=30):
    """
    Fast greedy reduction: for each term, check if it's a linear combo of others
    using a precomputed tensor matrix. Uses batched lstsq.
    """
    current = list(terms)
    nU, nV, nW = n * m, m * p, p * n
    total_keys = n * m * p  # max possible nonzero positions
    
    def term_to_vec(term):
        u, v, w = term
        vec = {}
        for i, ui in u.items():
            ii, jj = divmod(i, m)
            for j, vj in v.items():
                jj2, kk = divmod(j, p)
                if jj != jj2:
                    continue
                for k, wk in w.items():
                    kk2, ii2 = divmod(k, n)
                    if kk != kk2 or ii2 != ii:
                        continue
                    vec[(i, j, k)] = ui * vj * wk
        return vec
    
    # Precompute all term tensors as dicts
    tensors = [term_to_vec(t) for t in current]
    
    # Get sorted list of all keys that appear
    all_keys_set = set()
    for t in tensors:
        all_keys_set.update(t.keys())
    all_keys = sorted(all_keys_set)
    key_to_idx = {k: idx for idx, k in enumerate(all_keys)}
    
    # Build dense matrix: rows = keys, cols = terms
    dense = np.zeros((len(all_keys), len(current)))
    for c, t in enumerate(tensors):
        for key, val in t.items():
            dense[key_to_idx[key], c] = val
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 10.0:
            break
        removed = False
        for idx in range(len(current)):
            if time.time() > deadline - 10.0:
                break
            # Check if column idx is in span of other columns
            others = [j for j in range(len(current)) if j != idx]
            if not others:
                continue
            A = dense[:, others]
            b = dense[:, idx]
            if np.linalg.norm(b) < 1e-12:
                current.pop(idx)
                tensors.pop(idx)
                dense = dense[:, [j for j in range(len(dense.shape[1])) if j != idx]]
                removed = True
                break
            try:
                coeff, res, rnk, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                resid = res[0] if len(res) > 0 else np.linalg.norm(A @ coeff - b)
                bnorm = np.linalg.norm(b)
                if resid < 1e-8 * max(bnorm, 1e-12):
                    current.pop(idx)
                    tensors.pop(idx)
                    dense = dense[:, [j for j in range(dense.shape[1]) if j != idx]]
                    removed = True
                    break
            except Exception:
                continue
        if not removed:
            break
    
    return current

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    # Try multiple block partitionings
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook

    # Partitioning 1: 4 pairs + 1 single (current approach)
    partitions = [
        ([(0,1),(2,3),(4,5),(6,7)], 8),
        ([(0,1),(2,3),(4,5),(6,7)], 8),
        ([(0,2),(1,3),(4,6),(5,7)], 8),
        ([(0,1),(2,4),(3,5),(6,7)], 8),
        ([(0,3),(1,4),(2,5),(6,7)], 8),
    ]
    
    for k_pairs, single_k in partitions:
        if time.time() > deadline - 20.0:
            break
        try:
            terms = _build_terms(n, m, p, k_pairs, single_k)
            U, V, W = _terms_to_matrices(terms, n, m, p)
            if not _verify(U, V, W, n, m, p):
                continue
            rank = len(terms)
            if rank < best_rank:
                # Try to reduce further
                reduced = _fast_reduce(terms, n, m, p, deadline)
                if len(reduced) < rank:
                    rU, rV, rW = _terms_to_matrices(reduced, n, m, p)
                    if _verify(rU, rV, rW, n, m, p):
                        rank = len(reduced)
                        U, V, W = rU, rV, rW
                best_U, best_V, best_W = U, V, W
                best_rank = rank
        except Exception:
            continue
    
    # Fallback to schoolbook if nothing worked
    if best_U is None:
        best_U, best_V, best_W = _rank1_decomposition(n, m, p)
        best_rank = n * m * p

    # Final safety check
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _rank1_decomposition(n, m, p)

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
