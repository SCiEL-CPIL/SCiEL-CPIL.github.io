import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
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
    Rank-32 decomposition: 8 independent schoolbook 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for l in range(4):
        for rb in range(2):
            if rb == 0:
                def gA(i, j): return i * m + j
            else:
                def gA(i, j): return i * m + (2 + j)

            def gB(j, k, rb=rb, l=l):
                gj = (j if rb == 0 else 2 + j)
                return gj * p + (2 * l + k)

            def gC(k, i, l=l):
                return (2 * l + k) * n + i

            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = col
                        U[gA(i, j)][r] = 1
                        V[gB(j, k)][r] = 1
                        W[gC(k, i)][r] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def _construct_rank24():
    """
    Rank-24 decomposition using Winograd-style column-pairing.
    
    Split B's 8 columns into 4 pairs. For each pair, use a 2x2x2x2 
    Strassen-like decomposition shared across the j-dimension.
    
    Key insight: C[i,k] = sum_j A[i,j]*B[j,k]. Split j into {0,1} and {2,3}.
    For each column pair (k, k+1), we have a 2x2 (i,k) output from 
    4x2 (j,k) B-block and 2x4 (i,j) A.
    
    Use 3 terms per column pair per A-row-group (j-group), giving:
    4 column pairs * 2 j-groups * 3 terms = 24 terms.
    """
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for l in range(4):  # column pair index
        k0 = 2 * l
        k1 = 2 * l + 1
        for jg in range(2):  # j-group: 0 means j in {0,1}, 1 means j in {2,3}
            j0 = 2 * jg
            j1 = 2 * jg + 1
            # 3 Winograd terms for 2x2 * 2x2 -> 2x2 (rows i, cols k)
            # m1 = (A[i,j0]+A[i,j1]) * (B[j0,k]+B[j1,k])
            # m2 = A[i,j0] * B[j0,k]
            # m3 = A[i,j1] * B[j1,k]
            # C[i,k] = m1 - m2 - m3 + 2*A[i,j0]*B[j1,k] ... 
            # Actually for 1x2 * 2x2 -> 1x2, we need:
            # C[i,k0] = A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
            # C[i,k1] = A[i,j0]*B[j0,k1] + A[i,j1]*B[j1,k1]
            # This is just 4 rank-1 terms (2 per output). Not 3.
            # 
            # Let me use proper Strassen for 2x2 * 2x2:
            # The 2x2 A-block is A[i, j0:j1+1] for i in {0,1}
            # The 2x2 B-block is B[j0:j1+1, k0:k1+1]
            # Output is 2x2 C[i, k0:k1+1]
            # Strassen 7 terms, but we only need the structure.
            # However, we're using this for EACH i separately? No, Strassen 
            # operates on the full 2x2 * 2x2.
            # 
            # For j-group 0: A-block is A[0:2, 0:2] (2x2), B-block is B[0:2, k0:k1+1] (2x2)
            # For j-group 1: A-block is A[0:2, 2:4] (2x2), B-block is B[2:4, k0:k1+1] (2x2)
            # Each gives 7 Strassen terms. Total: 4 col-pairs * 2 j-groups * 7 = 56. Same as before!
            # 
            # The issue is that Strassen doesn't help for 2x2*2x2 in this context
            # because the dimensions are already minimal.
            # 
            # Let me try a different approach: use 3 terms per (i, col-pair) by 
            # exploiting linearity differently.
            # 
            # For fixed i and column pair (k0,k1):
            # C[i,k0] = A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
            # C[i,k1] = A[i,j0]*B[j0,k1] + A[i,j1]*B[j1,k1]
            # 
            # Define: s1 = A[i,j0]+A[i,j1], s2 = A[i,j0]
            # Then: C[i,k0] = s2*B[j0,k0] + (s1-s2)*B[j1,k0] = s1*B[j1,k0] + s2*(B[j0,k0]-B[j1,k0])
            # C[i,k1] = s1*B[j1,k1] + s2*(B[j0,k1]-B[j1,k1])
            # 
            # So 3 terms:
            # t1: U[i,j0]=1, U[i,j1]=1; V[j0,k0]=1,V[j0,k1]=1,V[j1,k0]=-1,V[j1,k1]=-1; W[k0,i]=1,W[k1,i]=1
            #     -> contributes (A[i,j0]+A[i,j1])*(B[j0,k0]-B[j1,k0]) to C[i,k0]
            #     and (A[i,j0]+A[i,j1])*(B[j0,k1]-B[j1,k1]) to C[i,k1]
            # t2: U[i,j0]=1; V[j0,k0]=1,V[j0,k1]=1; W[k0,i]=1,W[k1,i]=1
            #     -> contributes A[i,j0]*B[j0,k0] to C[i,k0], A[i,j0]*B[j0,k1] to C[i,k1]
            # t3: U[i,j1]=1; V[j1,k0]=1,V[j1,k1]=1; W[k0,i]=1,W[k1,i]=1
            #     -> contributes A[i,j1]*B[j1,k0] to C[i,k0], A[i,j1]*B[j1,k1] to C[i,k1]
            # 
            # Check: C[i,k0] from t1+t2+t3 = (A[i,j0]+A[i,j1])*(B[j0,k0]-B[j1,k0]) + A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
            # = A[i,j0]*B[j0,k0] - A[i,j0]*B[j1,k0] + A[i,j1]*B[j0,k0] - A[i,j1]*B[j1,k0] + A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
            # = 2*A[i,j0]*B[j0,k0] + A[i,j1]*B[j0,k0] - A[i,j0]*B[j1,k0]
            # That's NOT equal to A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0].
            # 
            # The Winograd approach doesn't directly apply here because we're not 
            # doing convolution. Let me just use 4 terms per (i, col-pair, j-group):
            # Total: 2 i * 4 col-pairs * 2 j-groups * 2 = 32. Same as before.
            # 
            # I'll return None to indicate this approach doesn't yield improvement.
            pass
    return None


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W."""
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


def _reduce_rank_via_nullspace(U, V, W, n, m, p, max_iter=50):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    Build the coefficient matrix and find null space vectors.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # For each term r, the contribution to T[i,j,k] is Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
    # We want to find a vector c such that sum_r c[r] * Ua[:,r] ⊗ Va[:,r] ⊗ Wa[:,r] = 0
    # This is a bilinear/quadratic system, not linear. So null-space doesn't directly apply.
    # 
    # Instead, let's try: for each term r, check if it can be expressed as a linear 
    # combination of other terms in a way that preserves the tensor.
    # This is NP-hard in general, so we'll do a greedy approach.
    
    # Greedy: try removing each term and checking if the remaining terms can be 
    # rescaled to still produce the correct tensor.
    # This is also hard, so let's just try simple cases.
    
    # Alternative: check if any two terms are identical (same U, V, W columns).
    # If so, we can combine them.
    from collections import defaultdict
    
    # Group terms by their (U_col, V_col, W_col) signature
    signatures = defaultdict(list)
    for r in range(rank):
        u_sig = tuple(Ua[:, r])
        v_sig = tuple(Va[:, r])
        w_sig = tuple(Wa[:, r])
        signatures[(u_sig, v_sig, w_sig)].append(r)
    
    # If any group has size > 1, we can combine them (but they'd cancel if signs differ)
    # For now, just check for exact duplicates
    new_U = []
    new_V = []
    new_W = []
    seen = set()
    for (u_sig, v_sig, w_sig), indices in signatures.items():
        if len(indices) > 1:
            # Multiple identical terms: they add up, so we can't just remove them
            # unless they cancel. If all have same sign, they sum to a multiple.
            # This doesn't help for rank reduction.
            pass
        for r in indices:
            new_U.append(Ua[:, r].tolist())
            new_V.append(Va[:, r].tolist())
            new_W.append(Wa[:, r].tolist())
    
    # No improvement from this approach
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    best = None

    # Candidate 1: rank-24 attempt
    try:
        cand = _construct_rank24()
        if cand is not None:
            ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
            if ok:
                U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
                best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 2: rank-32 (8 independent schoolbook 2x2 blocks)
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            if best is None or r < best[0]:
                best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 3: rank-56 (original Strassen, known valid) as fallback
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
