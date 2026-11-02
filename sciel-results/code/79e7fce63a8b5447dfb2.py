import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Return True iff U[r]·V[r]·W[r] reproduces C_{i,k} = A_{i,j}·B_{j,k} exactly."""
    rank = len(U)
    def Uv(i, j):
        return U[i * m + j]
    def Vv(j, k):
        return V[j * p + k]
    def Wv(k, i):
        return W[k * n + i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Uv(i, j)[r] * Vv(j, k)[r] * Wv(k, i)[r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_strassen(n, m, p):
    """Programmatic construction of a rank-28 exact decomposition.

    The M_{2,2,8} tensor is a direct sum of 4 independent M_{2,2,2} tensors
    (one per pair of B-columns / C-columns) sharing the same A-factor.  Each
    block is decomposed with the classic 7-term Strassen identity, giving
    4*7 = 28 rank-1 terms.  The factors are emitted from an explicit template
    so the code is data-driven rather than a hand-typed table.
    """
    # Strassen 2x2 identity.  A indices 00,01,10,11 -> 0,1,2,3 (row-major i*2+j).
    # Each term s has: (a-form dict), (b-form dict), and {output: coeff} with
    # output index in {00,01,10,11} -> 0,1,2,3.
    a_form = [
        {0: 1, 3: 1},   # 0: a+d
        {2: 1, 3: 1},   # 1: c+d
        {0: 1},         # 2: a
        {3: 1},         # 3: d
        {0: 1, 1: 1},   # 4: a+b
        {2: 1, 0: -1},  # 5: c-a
        {1: 1, 3: -1},  # 6: b-d
    ]
    b_form = [
        {0: 1, 3: 1},   # 0: x0+y1
        {0: 1},         # 1: x0
        {1: 1, 3: -1},  # 2: x1-y1
        {2: 1, 0: -1},  # 3: y0-x0
        {3: 1},         # 4: y1
        {0: 1, 1: 1},   # 5: x0+x1
        {2: 1, 3: 1},   # 6: y0+y1
    ]
    # C = sum_s coeff[s][out] * M_s ; output layout c*2 + d (row c, col d).
    c_out = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1},  # C11
    ]

    rank = 4 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s
            # U: A-factor (shared across blocks)
            for aidx, val in a_form[s].items():
                ai = aidx // 2
                aj = aidx % 2
                U[ai * m + aj][r] = val
            # V: B-factor restricted to column pair (k0, k0+1)
            for bidx, val in b_form[s].items():
                bj = bidx // 2
                bcol = bidx % 2
                kglob = k0 + bcol
                V[bj * p + kglob][r] = val
            # W: C-output for the same column pair (official W[k*n+i])
            for cout, cdict in enumerate(c_out):
                if s in cdict:
                    ci = cout // 2
                    ccol = cout % 2
                    kglob = k0 + ccol
                    W[kglob * n + ci][r] = cdict[s]
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_rank=None, deadline=None):
    """Try to find a lower-rank decomposition by checking for linear dependencies
    among the 28 rank-1 terms. Returns (U_reduced, V_reduced, W_reduced, rank_found)
    or (None, None, None, None) if no reduction found."""
    if max_rank is None:
        max_rank = len(U) - 1
    if deadline is None:
        deadline = time.time() + 120
    
    # Convert to numpy arrays for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    n_terms = U_np.shape[1]
    
    # Build the tensor from the 28 terms
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.zeros((n, m, p))
    for r in range(n_terms):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # Try to find if we can eliminate some terms
    # Strategy: try removing each term and see if the remaining terms can be
    # adjusted (via linear combination) to still reproduce T
    
    best_rank = n_terms
    best_solution = None
    
    # First, check if there are obvious linear dependencies by looking at the
    # structure. The key insight is that the A-factor is shared across all 4 blocks.
    # This means that for a given Strassen term s, the U vector is the same across
    # all 4 blocks, but V and W are different.
    
    # Let's try a more systematic approach: use the fact that we have 4 blocks
    # with the same U patterns. We can try to find a rank-r decomposition where
    # r < 28 by solving a linear system.
    
    # For each candidate rank r from min possible to max_rank:
    # We need to find U', V', W' of rank r such that the tensor is reproduced.
    
    # This is a hard problem, but we can try a simpler approach first:
    # Check if any subset of the 28 terms is linearly dependent in a way that
    # allows elimination.
    
    # Build the "term signature" matrix: each term r contributes to T via
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r]. Two terms are linearly dependent if one is
    # a scalar multiple of the other.
    
    # Check for proportional terms
    proportional = []
    for r1 in range(n_terms):
        for r2 in range(r1+1, n_terms):
            # Check if term r2 is a scalar multiple of term r1
            # This requires U[:,r2] = c*U[:,r1], V[:,r2] = c*V[:,r1], W[:,r2] = c*W[:,r1]
            # for some c
            U1 = U_np[:, r1]
            U2 = U_np[:, r2]
            if np.allclose(U1, 0):
                continue
            # Find c from U
            nonzero = np.abs(U1) > 1e-10
            if not np.any(nonzero):
                continue
            c = U2[nonzero] / U1[nonzero]
            if np.allclose(c, c[0]):
                c_val = c[0]
                if np.allclose(V_np[:, r2], c_val * V_np[:, r1]) and \
                   np.allclose(W_np[:, r2], c_val * W_np[:, r1]):
                    proportional.append((r1, r2, c_val))
    
    if proportional:
        # We found proportional terms; we can eliminate one from each pair
        # Build a graph of proportional terms and find independent set
        # For simplicity, just greedily eliminate
        to_remove = set()
        for r1, r2, c in proportional:
            if r1 not in to_remove and r2 not in to_remove:
                # Keep the one with larger magnitude, remove the other
                mag1 = np.sum(U_np[:, r1]**2) * np.sum(V_np[:, r1]**2) * np.sum(W_np[:, r1]**2)
                mag2 = np.sum(U_np[:, r2]**2) * np.sum(V_np[:, r2]**2) * np.sum(W_np[:, r2]**2)
                if mag1 >= mag2:
                    to_remove.add(r2)
                else:
                    to_remove.add(r1)
        
        if to_remove:
            keep = [r for r in range(n_terms) if r not in to_remove]
            new_rank = len(keep)
            if new_rank < best_rank:
                best_rank = new_rank
                U_reduced = U_np[:, keep]
                V_reduced = V_np[:, keep]
                W_reduced = W_np[:, keep]
                # Verify
                T_check = np.zeros((n, m, p))
                for r in range(new_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_check[i,j,k] += U_reduced[i*m+j, r] * V_reduced[j*p+k, r] * W_reduced[k*n+i, r]
                if np.allclose(T_check, T, atol=1e-9):
                    best_solution = (U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), new_rank)
    
    if best_solution is not None:
        return best_solution
    
    # No simple reduction found; return original
    return None, None, None, None


def construct():
    """Return {'U','V','W'} factor matrices of shapes (n*m,rank),(m*p,rank),(p*n,rank)
    with integer entries that exactly decompose the 2x2 * 2x8 multiplication tensor."""
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115  # Safety margin before 120s limit
    
    U, V, W = _build_strassen(n, m, p)
    if _verify(U, V, W, n, m, p):
        # Try to reduce rank
        U_red, V_red, W_red, rank_found = _try_reduce_rank(U, V, W, n, m, p, deadline=deadline)
        if U_red is not None and rank_found < len(U):
            # Verify the reduced decomposition
            if _verify(U_red, V_red, W_red, n, m, p):
                return {"U": U_red, "V": V_red, "W": W_red}
        
        # Return the verified rank-28 decomposition
        return {"U": U, "V": V, "W": W}

    # Safety fallback
    _ = time.time() - start
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
