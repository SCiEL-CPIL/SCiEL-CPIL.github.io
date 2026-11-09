"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition.

Re-structured with a different internal organisation:
  * explicit deadline guard
  * rank-44 Strassen construction written in a compact, index-driven loop
  * fast verification via matmul
  * fallback to schoolbook (rank 48)
"""

import numpy as np
import time

_T0 = time.perf_counter()
_DEADLINE = 145.0          # safety margin before the 150 s kill

def _time_left():
    return _DEADLINE - (time.perf_counter() - _T0)

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _make_target(n, m, p):
    """Build the matrix-multiplication tensor T[u,v,w]=1 where
    u = i*m+j, v = j*p+k, w = k*n+i."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _check(U, V, W, n, m, p, tol=1e-12):
    """Verify sum_r U[:,r] V[:,r] W[:,r] == T  exactly (within tol)."""
    Tu = np.array(U, dtype=np.float64)
    Tv = np.array(V, dtype=np.float64)
    Tw = np.array(W, dtype=np.float64)
    T = _make_target(n, m, p)
    # contract over r:  result[u,v,w] = sum_r Tu[u,r]*Tv[v,r]*Tw[w,r]
    # do it efficiently:  (Tu @ Tv.T) gives (u,v); then for each w ...
    # Actually:  result = einsum('ur,vr,wr->uvw', Tu, Tv, Tw)
    R = np.einsum('ur,vr,wr->uvw', Tu, Tv, Tw, optimize=True)
    return bool(np.allclose(R, T, atol=tol, rtol=0.0))


# ---------------------------------------------------------------------------
# Rank-44 construction  (4 blocks of 2x3x2, each rank 11)
# ---------------------------------------------------------------------------

def _build_rank44():
    """Return (U, V, W) as lists-of-lists of ints for rank 44."""
    n, m, p = 2, 3, 8
    R = 44
    U = [[0] * R for _ in range(n * m)]   # 6 x 44
    V = [[0] * R for _ in range(m * p)]   # 24 x 44
    W = [[0] * R for _ in range(p * n)]   # 16 x 44

    # Pre-compute the 7 Strassen coefficients for a 2x2x2 block
    # plus 4 outer-product terms for the 2x1x2 part.
    # For each block b (k0=2b, k1=2b+1):
    #
    # Strassen (7 terms) operate on A-submatrix rows 0,1 cols 0,1
    #   and B-submatrix rows 0,1 cols k0,k1
    #
    # Outer product (4 terms) operate on A col 2 and B row 2.

    # Strassen term definitions for a generic 2x2x2:
    #  M1=(a+d)(e+h) -> c00, c11
    #  M2=(c+d)e     -> c10, c11
    #  M3=a(f-h)     -> c01, c11
    #  M4=d(g-e)     -> c00, c10
    #  M5=(a+b)h     -> c00, c01
    #  M6=(c-a)(e+f) -> c11
    #  M7=(b-d)(g+h) -> c00
    # where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    #       e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    #
    # In our indexing for block b:
    #   a = U[0], b = U[1], c = U[3], d = U[4]
    #   e = V[0*p+k0], f = V[0*p+k1], g = V[1*p+k0], h = V[1*p+k1]
    #   c00 = W[k0*n+0], c01 = W[k1*n+0], c10 = W[k0*n+1], c11 = W[k1*n+1]

    strassen_defs = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        # M1 = (a+d)(e+h)
        ((0, 4), (1, 1), (None, None), (1, 1), (None, None), (1, 1)),  # placeholder
        # M2 = (c+d)e
        ((3, 4), (1, 1), (None,), (1,), (None, None), (1, -1)),
        # M3 = a(f-h)
        ((0,), (1,), (None, None), (1, -1), (None, None), (1, 1)),
        # M4 = d(g-e)
        ((4,), (1,), (None, None), (1, -1), (None, None), (1, 1)),
        # M5 = (a+b)h
        ((0, 1), (1, 1), (None,), (1,), (None, None), (-1, 1)),
        # M6 = (c-a)(e+f)
        ((3, 0), (1, -1), (None, None), (1, 1), (None,), (1,)),
        # M7 = (b-d)(g+h)
        ((1, 4), (1, -1), (None, None), (1, 1), (None,), (1,)),
    ]

    col = 0
    for b in range(4):
        k0, k1 = 2 * b, 2 * b + 1
        # V indices for e,f,g,h (B rows 0,1)
        e_i = 0 * p + k0
        f_i = 0 * p + k1
        g_i = 1 * p + k0
        h_i = 1 * p + k1
        # V indices for B row 2
        e2_i = 2 * p + k0
        f2_i = 2 * p + k1
        # W indices
        w00 = k0 * n + 0
        w01 = k1 * n + 0
        w10 = k0 * n + 1
        w11 = k1 * n + 1
        # U indices
        a_i = 0 * m + 0   # A[0,0]
        b_i = 0 * m + 1   # A[0,1]
        c_i = 1 * m + 0   # A[1,0]
        d_i = 1 * m + 1   # A[1,1]
        a2_i = 0 * m + 2  # A[0,2]
        d2_i = 1 * m + 2  # A[1,2]

        # Attempt rank-11 per block by combining outer-product with Strassen.
        # The 4 outer-product terms contribute:
        #   w00 += a2*e2,  w01 += a2*f2,  w10 += d2*e2,  w11 += d2*f2
        # We can try to absorb these into existing Strassen terms by adding
        # a2*e2, a2*f2, d2*e2, d2*f2 as additional rank-1 contributions.
        # But that would increase rank. Instead, try to find a rank-11 scheme.
        #
        # Known result: rank(2x3x2) = 11. Construction:
        # Use 7 Strassen + 4 outer = 11. That's what we have.
        # So rank-44 = 4*11 is tight for this block decomposition.
        #
        # Try a different approach: global rank-43 by sharing one term.
        # For now, keep rank-44 but verify correctness.

        sf = [
            ((a_i, d_i), (1, 1), (e_i, h_i), (1, 1), (w00, w11), (1, 1)),
            ((c_i, d_i), (1, 1), (e_i,),      (1,),  (w10, w11), (1, -1)),
            ((a_i,),     (1,),  (f_i, h_i), (1, -1),(w01, w11), (1, 1)),
            ((d_i,),     (1,),  (g_i, e_i), (1, -1),(w00, w10), (1, 1)),
            ((a_i, b_i), (1, 1), (h_i,),    (1,),  (w00, w01), (-1, 1)),
            ((c_i, a_i), (1, -1),(e_i, f_i),(1, 1), (w11,),    (1,)),
            ((b_i, d_i), (1, -1),(g_i, h_i),(1, 1), (w00,),    (1,)),
        ]

        for (ui, uc, vi, vc, wi, wc) in sf:
            for idx, c in zip(ui, uc):
                U[idx][col] += c
            for idx, c in zip(vi, vc):
                V[idx][col] += c
            for idx, c in zip(wi, wc):
                W[idx][col] += c
            col += 1

        # 4 outer-product terms for A-col-2 * B-row-2
        for (ui, vi, wi) in [(a2_i, e2_i, w00), (a2_i, f2_i, w01), (d2_i, e2_i, w10), (d2_i, f2_i, w11)]:
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Schoolbook fallback (rank 48)
# ---------------------------------------------------------------------------

def _build_schoolbook():
    n, m, p = 2, 3, 8
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_rank43():
    """Try to achieve rank 43 by sharing one rank-1 term across two blocks.
    
    Idea: The outer-product terms for block b use V indices 2*p+k0, 2*p+k1.
    If we can express one of these as a linear combination that allows
    merging with a Strassen term from an adjacent block, we save 1 rank.
    
    Alternative: Use a different 2x3x2 rank-11 scheme where one term
    is shared between adjacent blocks.
    
    For now, try: take the rank-44 construction and attempt to eliminate
    one term by expressing it as a combination of others (unlikely to work
    for exact integer decomposition, but try a small perturbation search).
    """
    n, m, p = 2, 3, 8
    R = 43
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Start with rank-44, then try to zero out one column and redistribute
    base = _build_rank44()
    U44 = np.array(base["U"], dtype=np.float64)
    V44 = np.array(base["V"], dtype=np.float64)
    W44 = np.array(base["W"], dtype=np.float64)
    
    # Try removing each of the 44 columns and see if the remaining 43
    # can be adjusted (by adding small corrections) to still work.
    # This is a linear algebra problem: find x in R^43 such that
    # sum_r x_r (u_r ⊗ v_r ⊗ w_r) = T where {u_r, v_r, w_r} are the
    # remaining 43 factors. This requires the 43 rank-1 tensors to span T.
    
    # Quick check: is T in the span of any 43 of the 44 rank-1 tensors?
    # Build the 44 rank-1 tensors as vectors in R^{6*24*16}
    # Too large for exact check. Instead, try a simpler approach:
    # Check if any column is a linear combination of the others in each
    # of U, V, W simultaneously.
    
    # For U: check if any column is in span of others
    # U is 6x44. Rank of U is at most 6. So many columns are dependent.
    # But we need simultaneous dependence in U, V, AND W.
    
    # Try: for each pair of columns (i, j), check if column i can be
    # expressed as alpha*column_j + beta*(something) in all three matrices.
    # This is complex. Instead, try a randomized approach:
    # Pick a random column to remove, solve for the correction needed.
    
    import random
    random.seed(42)
    
    for trial in range(200):
        if _time_left() < 5:
            break
        # Try removing column `rem`
        rem = random.randint(0, 43)
        keep = [c for c in range(44) if c != rem]
        
        U_k = U44[:, keep]  # 6 x 43
        V_k = V44[:, keep]  # 24 x 43
        W_k = W44[:, keep]  # 16 x 43
        
        # The removed term contributes: U44[:,rem] ⊗ V44[:,rem] ⊗ W44[:,rem]
        # We need to find coefficients c_k such that
        # sum_k c_k (U_k[:,k] ⊗ V_k[:,k] ⊗ W_k[:,k]) = T + U44[:,rem] ⊗ V44[:,rem] ⊗ W44[:,rem]
        # But c_k are already fixed (they are the original coefficients).
        # Actually, the original decomposition is:
        # T = sum_{r=0}^{43} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # If we remove term `rem`, we get T - U[:,rem] ⊗ V[:,rem] ⊗ W[:,rem]
        # which is NOT zero. So we can't just remove a term.
        
        # We need to find NEW coefficients for the remaining 43 terms such that
        # sum_{k in keep} c'_k (U[:,k] ⊗ V[:,k] ⊗ W[:,k]) = T
        # This is a system of 6*24*16 = 2304 equations in 43 unknowns.
        # Overdetermined. Only works if T happens to be in the span.
        
        # Let's check: does T lie in the span of the 43 rank-1 tensors?
        # Build the linear system: for each (u,v,w) in support of T,
        # sum_k c'_k U[u,k] V[v,k] W[w,k] = T[u,v,w]
        
        # The support of T has 2*3*8 = 48 entries.
        # So we have 48 equations in 43 unknowns.
        # Check if consistent.
        
        A_mat = np.zeros((48, 43), dtype=np.float64)
        b_vec = np.zeros(48, dtype=np.float64)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    for ci, k_idx in enumerate(keep):
                        A_mat[idx, ci] = U44[u, k_idx] * V44[v, k_idx] * W44[w, k_idx]
                    b_vec[idx] = 1.0
                    idx += 1
        
        # Solve least squares
        try:
            sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            # Check if the solution is exact
            if np.allclose(A_mat @ sol, b_vec, atol=1e-10, rtol=0):
                # Found a valid rank-43 decomposition!
                U3 = U_k.T.copy()  # 43 x 6 -> we need 6 x 43
                V3 = V_k.T.copy()  # 43 x 24
                W3 = W_k.T.copy()  # 43 x 16
                # But we need to scale by the solution coefficients
                # Actually, the decomposition is sum_k c'_k U[:,k] ⊗ V[:,k] ⊗ W[:,k]
                # We can absorb c'_k into one of the matrices
                for ci in range(43):
                    U3[ci, :] *= sol[ci]
                    # V3 and W3 stay the same
                return {"U": U3.tolist(), "V": V3.T.tolist(), "W": W3.T.tolist()}
        except:
            continue
    
    return None


def construct():
    n, m, p = 2, 3, 8

    # Try rank-43 first
    if _time_left() > 10:
        cand43 = _build_rank43()
        if cand43 is not None:
            if _check(cand43["U"], cand43["V"], cand43["W"], n, m, p):
                return cand43

    # Fast path: rank-44 Strassen
    cand = _build_rank44()
    if _check(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand

    # Fallback: schoolbook rank 48
    fb = _build_schoolbook()
    if _check(fb["U"], fb["V"], fb["W"], n, m, p):
        return fb

    # Should never reach here
    return fb
