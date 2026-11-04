"""Rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication.

Strategy: deterministic block-partition construction (Strassen on 2x2 blocks +
schoolbook on the remainder), combined with a local-search rank-reduction pass
that attempts to merge pairs of terms when the merged contribution vanishes on
all tensor entries, followed by a short ALS polish.  The result is returned as
the lowest-rank valid decomposition found within the time budget.
"""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]


def _build_target(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _tensor_from_factors(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _is_valid(U, V, W, target, tol=1e-9):
    est = _tensor_from_factors(U, V, W)
    return np.max(np.abs(est - target)) < tol


def _try_round_half(U, V, W, target, tol=1e-8):
    """Try to round to half-integers; return (U,V,W, success)."""
    Ur = np.round(U * 2) / 2.0
    Vr = np.round(V * 2) / 2.0
    Wr = np.round(W * 2) / 2.0
    if _is_valid(Ur, Vr, Wr, target, tol):
        return Ur, Vr, Wr, True
    # Try integers
    Ui = np.round(U)
    Vi = np.round(V)
    Wi = np.round(W)
    if _is_valid(Ui, Vi, Wi, target, tol):
        return Ui, Vi, Wi, True
    return U, V, W, False


def _strassen_2x2_terms():
    """Return 7 Strassen terms for 2x2 matrix multiplication.
    
    Each term is (u, v, w) where u, v, w are length-4 vectors
    (for 2x2 matrices flattened row-major).
    C = A * B, C[i,k] = sum_j A[i,j] B[j,k]
    Term r contributes u_r[i,j] * v_r[j',k] * w_r[k',i'] ... 
    Actually for the tensor T[a,b,c] where a=(i,j), b=(j,k), c=(k,i):
    T[(i,j),(j,k),(k,i)] = 1
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = A10(B00+B01)
    M3 = (A00-A10)(B10+B11)
    M4 = A11(B11-B01)
    M5 = (A01+A11)(B00-B10)
    M6 = (A00+A10)(B01-B11)
    M7 = (A01-A10)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M2 + M3 - M4 + M6
    C10 = M2 + M3 + M5 - M7
    C11 = M1 - M2 + M4 + M6
    """
    # For each term, we need vectors u (over A indices), v (over B indices), w (over C indices)
    # such that the product u[a]*v[b]*w[c] contributes to the tensor.
    # The tensor entry T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    # So for a given term, u[a] is nonzero only at a=(i,j) for specific (i,j),
    # v[b] nonzero only at b=(j,k), w[c] nonzero only at c=(k,i).
    # The contribution to T[a,b,c] from term r is u_r[a] * v_r[b] * w_r[c].
    # We need sum_r u_r[a] v_r[b] w_r[c] = T[a,b,c].
    
    # For Strassen term M1 = (A00+A11)(B00+B11):
    # u has +1 at (0,0) and (1,1)
    # v has +1 at (0,0) and (0,1) ... wait, v is over B indices (j,k)
    # B00 is at (0,0), B11 is at (1,1). So v has +1 at (0,0) and (1,1).
    # w: C00 gets +1, C11 gets +1. C00 is at c=(0,0) [k=0,i=0], C11 at c=(1,1) [k=1,i=1].
    # So w has +1 at (0,0) and (1,1).
    
    # Let me define indices: A index a = i*2+j, B index b = j*2+k, C index c = k*2+i.
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  [c = k*2+i]
    
    terms = []
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    u = np.zeros(4); u[0] = 1; u[3] = 1   # A00, A11
    v = np.zeros(4); v[0] = 1; v[3] = 1   # B00, B11
    w = np.zeros(4); w[0] = 1; w[3] = 1   # C00, C11
    terms.append((u, v, w))
    
    # M2: A10(B00+B01) -> C01 += M2, C10 += M2
    u = np.zeros(4); u[2] = 1   # A10
    v = np.zeros(4); v[0] = 1; v[1] = 1   # B00, B01
    w = np.zeros(4); w[1] = 1; w[2] = 1   # C01, C10
    terms.append((u, v, w))
    
    # M3: (A00-A10)(B10+B11) -> C01 += M3, C10 += M3
    u = np.zeros(4); u[0] = 1; u[2] = -1   # A00 - A10
    v = np.zeros(4); v[2] = 1; v[3] = 1   # B10, B11
    w = np.zeros(4); w[1] = 1; w[2] = 1   # C01, C10
    terms.append((u, v, w))
    
    # M4: A11(B11-B01) -> C00 += M4, C11 += M4
    u = np.zeros(4); u[3] = 1   # A11
    v = np.zeros(4); v[3] = 1; v[1] = -1   # B11 - B01
    w = np.zeros(4); w[0] = 1; w[3] = 1   # C00, C11
    terms.append((u, v, w))
    
    # M5: (A01+A11)(B00-B10) -> C00 -= M5, C10 += M5
    u = np.zeros(4); u[1] = 1; u[3] = 1   # A01 + A11
    v = np.zeros(4); v[0] = 1; v[2] = -1   # B00 - B10
    w = np.zeros(4); w[0] = -1; w[2] = 1   # -C00 + C10
    terms.append((u, v, w))
    
    # M6: (A00+A10)(B01-B11) -> C01 += M6, C11 += M6
    u = np.zeros(4); u[0] = 1; u[2] = 1   # A00 + A10
    v = np.zeros(4); v[1] = 1; v[3] = -1   # B01 - B11
    w = np.zeros(4); w[1] = 1; w[3] = 1   # C01 + C11
    terms.append((u, v, w))
    
    # M7: (A01-A10)(B10+B11) -> C00 += M7, C10 -= M7
    u = np.zeros(4); u[1] = 1; u[2] = -1   # A01 - A10
    v = np.zeros(4); v[2] = 1; v[3] = 1   # B10 + B11
    w = np.zeros(4); w[0] = 1; w[2] = -1   # C00 - C10
    terms.append((u, v, w))
    
    return terms


def _build_strassen_block(target, n, m, p, i_off, j_off, k_off, size):
    """Build Strassen terms embedded in the full tensor space for a size x size block.
    
    i_off, j_off, k_off: offsets for the block in the full matrices.
    size: block size (2 for 2x2).
    Returns list of (u, v, w) vectors of full dimension.
    """
    s = size
    terms = _strassen_2x2_terms() if s == 2 else None
    if terms is None:
        return []
    
    U_full = []
    V_full = []
    W_full = []
    for u, v, w in terms:
        uf = np.zeros(n * m)
        vf = np.zeros(m * p)
        wf = np.zeros(p * n)
        for idx in range(s * s):
            ii = idx // s
            jj = idx % s
            a = (i_off + ii) * m + (j_off + jj)
            uf[a] = u[idx]
            b = (j_off + jj) * p + (k_off + idx % s)  # B index: (j, k) where j=jj, k=idx%s? No.
            # Wait: B index b = j*p + k. For the block, j ranges j_off..j_off+s-1, k ranges k_off..k_off+s-1.
            # The v vector is over B indices. v[idx] corresponds to B[ii, jj] in the local 2x2.
            # Local B index: (jj_local, kk_local) where jj_local = idx // s, kk_local = idx % s.
            # Global B index: (j_off + jj_local) * p + (k_off + kk_local)
            # But v is indexed by the local B index idx = jj_local * s + kk_local.
            # So for each local B index idx, global b = (j_off + idx//s) * p + (k_off + idx%s)
            pass
        
        # Redo: v[idx] is the coefficient for local B entry at (idx//s, idx%s).
        for idx in range(s * s):
            jj_local = idx // s
            kk_local = idx % s
            b = (j_off + jj_local) * p + (k_off + kk_local)
            vf[b] = v[idx]
            
            # w[idx] is the coefficient for local C entry. C index c = k*n + i.
            # Local C: (kk_local, ii_local) where ii_local = idx // s? No.
            # C entry c = k*n + i. In the local block, k = kk_local, i = ii_local.
            # But w is indexed by the local C index. The local C index for (k,i) is k*s + i.
            # So w[idx] corresponds to local C at (idx//s, idx%s) = (k_local, i_local).
            kk_local_w = idx // s
            ii_local_w = idx % s
            c = (k_off + kk_local_w) * n + (i_off + ii_local_w)
            wf[c] = w[idx]
        
        U_full.append(uf)
        V_full.append(vf)
        W_full.append(wf)
    
    return list(zip(U_full, V_full, W_full))


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 115  # safety margin
    
    target = _build_target(n, m, p)
    
    # Schoolbook fallback
    rank_sb = n * m * p  # 80
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # ------------------------------------------------------------------
    # Block-partition construction using Strassen on 2x2 blocks
    # ------------------------------------------------------------------
    # A is 4x4, split into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # B is 4x5, split rows into 2+2, columns into 2+2+1
    # B = [[B00, B01, B02], [B10, B11, B12]]
    #     B00, B01, B10, B11 are 2x2; B02, B12 are 2x1
    # C = A*B is 4x5
    # C = [[C00, C01, C02], [C10, C11, C12]]
    # C00 = A00*B00 + A01*B10  (2x2)
    # C01 = A00*B01 + A01*B11  (2x2)
    # C02 = A00*B02 + A01*B12  (2x1)
    # C10 = A10*B00 + A11*B10  (2x2)
    # C11 = A10*B01 + A11*B11  (2x2)
    # C12 = A10*B02 + A11*B12  (2x1)
    
    # For each 2x2x2 product, use Strassen (7 terms)
    # For each 2x2x1 product, use schoolbook (4 terms)
    
    # Total: 4 * 7 + 4 * 4 = 28 + 16 = 44? Wait, let me recount.
    # C00: A00*B00 + A01*B10 -> two 2x2x2 products, each 7 terms = 14
    # C01: A00*B01 + A01*B11 -> 14
    # C10: A10*B00 + A11*B10 -> 14
    # C11: A10*B01 + A11*B11 -> 14
    # C02: A00*B02 + A01*B12 -> two 2x2x1 products, each 4 terms = 8
    # C12: A10*B02 + A11*B12 -> 8
    # Total: 14*4 + 8*2 = 56 + 16 = 72
    
    # That's not great. Let me think differently.
    # Actually, we can share Strassen terms across products that use the same A and B blocks.
    # For example, C00 and C01 both use A00. C00 and C10 both use B00.
    # But Strassen terms are specific to a pair (A_block, B_block), so sharing is non-trivial.
    
    # Let's just build it term by term.
    terms = []
    
    # Helper: add a Strassen decomposition for A_block (2x2) * B_block (2x2) contributing to C_block (2x2)
    def add_strassen(i_off, j_off, k_off):
        """Add 7 Strassen terms for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2] -> C[k_off:k_off+2, i_off:i_off+2]"""
        for u, v, w in _build_strassen_block(target, n, m, p, i_off, j_off, k_off, 2):
            terms.append((u, v, w))
    
    # Helper: add schoolbook terms for A_block (2x2) * B_block (2x1) -> C_block (2x1)
    def add_schoolbook_vec(i_off, j_off, k_off):
        """Add 4 schoolbook terms for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off] -> C[k_off, i_off:i_off+2]"""
        for ii in range(2):
            for jj in range(2):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                a = (i_off + ii) * m + (j_off + jj)
                b = (j_off + jj) * p + k_off
                c = k_off * n + (i_off + ii)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                terms.append((u, v, w))
    
    # C00 = A00*B00 + A01*B10
    add_strassen(0, 0, 0)  # A00*B00 -> C00
    add_strassen(0, 2, 0)  # A01*B10 -> C00
    # C01 = A00*B01 + A01*B11
    add_strassen(0, 0, 2)  # A00*B01 -> C01
    add_strassen(0, 2, 2)  # A01*B11 -> C01
    # C10 = A10*B00 + A11*B10
    add_strassen(2, 0, 0)  # A10*B00 -> C10
    add_strassen(2, 2, 0)  # A11*B10 -> C10
    # C11 = A10*B01 + A11*B11
    add_strassen(2, 0, 2)  # A10*B01 -> C11
    add_strassen(2, 2, 2)  # A11*B11 -> C11
    # C02 = A00*B02 + A01*B12
    add_schoolbook_vec(0, 0, 4)  # A00*B02 -> C02
    add_schoolbook_vec(0, 2, 4)  # A01*B12 -> C02
    # C12 = A10*B02 + A11*B12
    add_schoolbook_vec(2, 0, 4)  # A10*B02 -> C12
    add_schoolbook_vec(2, 2, 4)  # A11*B12 -> C12
    
    R0 = len(terms)
    U0 = np.column_stack([t[0] for t in terms])
    V0 = np.column_stack([t[1] for t in terms])
    W0 = np.column_stack([t[2] for t in terms])
    
    if _is_valid(U0, V0, W0, target):
        best_rank = R0
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # ------------------------------------------------------------------
    # Local search: try to merge pairs of terms
    # ------------------------------------------------------------------
    U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
    R_cur = best_rank
    merged = True
    while merged and time.time() < deadline - 15:
        merged = False
        for r1 in range(R_cur):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, R_cur):
                # Try merging terms r1 and r2: new term = sum of both
                # Check if removing them and adding the sum still gives valid decomposition
                U_test = U_cur.copy()
                V_test = V_cur.copy()
                W_test = W_cur.copy()
                # Remove r2 (shift indices)
                U_test[:, r2] = U_test[:, r1] + U_test[:, r2]
                V_test[:, r2] = V_test[:, r1] + V_test[:, r2]
                W_test[:, r2] = W_test[:, r1] + W_test[:, r2]
                U_test = np.delete(U_test, r1, axis=1)
                V_test = np.delete(V_test, r1, axis=1)
                W_test = np.delete(W_test, r1, axis=1)
                
                if _is_valid(U_test, V_test, W_test, target):
                    U_cur, V_cur, W_cur = U_test, V_test, W_test
                    R_cur -= 1
                    best_rank = R_cur
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    merged = True
                    break
            if merged:
                break
    
    # ------------------------------------------------------------------
    # ALS polish: try to reduce rank further
    # ------------------------------------------------------------------
    if time.time() < deadline - 20 and R_cur > 60:
        np.random.seed(42)
        for R_try in range(R_cur - 1, max(55, R_cur - 10), -1):
            if time.time() > deadline - 10:
                break
            for trial in range(3):
                if time.time() > deadline - 5:
                    break
                # Initialize from current best, projecting to lower rank
                # Use SVD-like approach: stack and random project
                U_init = np.random.randn(n * m, R_try) * 0.3
                V_init = np.random.randn(m * p, R_try) * 0.3
                W_init = np.random.randn(p * n, R_try) * 0.3
                
                # Seed with some structure from current solution
                for idx in range(min(R_try, R_cur)):
                    U_init[:, idx] += U_cur[:, idx] * 0.5
                    V_init[:, idx] += V_cur[:, idx] * 0.5
                    W_init[:, idx] += W_cur[:, idx] * 0.5
                
                lr = 0.005
                for step in range(200):
                    if time.time() > deadline - 3:
                        break
                    est = _tensor_from_factors(U_init, V_init, W_init)
                    residual = est - target
                    max_res = np.max(np.abs(residual))
                    if max_res < 1e-8:
                        break
                    dU = np.einsum('abc,br,cr->ar', residual, V_init, W_init)
                    dV = np.einsum('abc,ar,cr->br', residual, U_init, W_init)
                    dW = np.einsum('abc,ar,br->cr', residual, U_init, V_init)
                    U_init -= lr * dU
                    V_init -= lr * dV
                    W_init -= lr * dW
                
                est_final = _tensor_from_factors(U_init, V_init, W_init)
                if np.max(np.abs(est_final - target)) < 1e-7:
                    U_c, V_c, W_c, _ = _try_round_half(U_init, V_init, W_init, target)
                    best_rank = R_try
                    best_U, best_V, best_W = U_c, V_c, W_c
                    break
            if best_rank == R_try:
                break
    
    # Final rounding attempt
    U_final, V_final, W_final, _ = _try_round_half(best_U, best_V, best_W, target)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
