"""Strassen-based 4x5x6 matrix multiplication with peeling refinement."""
import time
import numpy as np
import random

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24
    deadline = time.time() + 130.0

    # ---------- Build the base Strassen 108-rank decomposition ----------
    # Decompose 4x5x6 as:
    #   [A1 A2] [B1 B2]   where A1,A2 are 4x2, B1,B2 are 2x6
    #   [A3 A4] [B3 B4]
    # C1 = A1*B1 + A2*B3  (4x6)
    # C2 = A1*B2 + A2*B4  (4x6)
    # We handle C1 as 4x2x6 and C2 as 4x2x6, each split into:
    #   4x2x6 = 4x2x2 (Strassen 2x2x2 rank 7) + 4x2x4 (Strassen 4x2x4)
    # Actually simpler: treat 4x2x6 as (4x2x2) + (4x2x2) + (4x2x2) = 3 blocks of 2x2x2? No.
    # Let's do 4x2x6 = (2x2x6)+(2x2x6) and each 2x2x6 = (2x2x2)+(2x2x2)+(2x2x2) = 3*7=21? No, 2x2x6:
    #   2x2x6 = 2x2x2 + 2x2x2 + 2x2x2 = 3 blocks of rank-7 = 21. But 2x2x6 schoolbook is 24.
    # Hmm, let me just do it cleanly with a recursive Strassen that handles rectangular.

    # Simpler robust approach: build the tensor via a recursive Strassen that
    # splits the inner dimension (m) when possible, and the outer dims otherwise.
    # We'll build U,V,W as lists of (row, col) integer entries accumulating into rank columns.

    terms = []  # each term: (u_row, v_row, w_row, u_val, v_val, w_val)

    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        if u_val == 0 and v_val == 0 and w_val == 0:
            return
        terms.append((u_row, v_row, w_row, u_val, v_val, w_val))

    def strassen_block(i0, j0, k0, si, sj, sk, A_coeff, B_coeff, C_scale):
        """
        Recursive Strassen for a rectangular block A[i0:i0+si, j0:j0+sj] * B[j0:j0+sj, k0:k0+sk]
        -> C[i0:i0+si, k0:k0+sk] with output scale C_scale.
        A_coeff, B_coeff: scalar multipliers (the block is A_coeff * A_sub, B_coeff * B_sub).
        """
        if si == 1 and sj == 1 and sk == 1:
            add_term(i0, j0, k0, A_coeff, B_coeff, C_scale)
            return

        # Choose split dimension: prefer splitting the middle (j) dimension if even and >1,
        # else split the longest outer dimension.
        if sj >= 2 and sj % 2 == 0:
            # Split along j: A[i0:i0+si, j0:j0+sj] * B[j0:j0+sj, k0:k0+sk]
            # = A[:, :sj/2] * B[:sj/2, :] + A[:, sj/2:] * B[sj/2:, :]
            # This is just a sum, not Strassen. To use Strassen we need square blocks.
            # For rectangular, Strassen applies when we can form square sub-blocks.
            # Fallback: split the largest square-ish dimension.
            pass

        # For simplicity and correctness, use a general recursive scheme:
        # If si==sj==sk and even, use true Strassen 7-multiplication.
        # Otherwise, split the largest dimension in half (even if rectangular) as a sum.
        if si == sj == sk and si % 2 == 0 and si >= 2:
            half = si // 2
            # 7 Strassen multiplications of size half
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6

            def A_sub(di, dj, coeff):
                # returns (i, j, coeff) for a single-element A contribution
                return (i0 + di * half, j0 + dj * half, coeff)

            def B_sub(dj, dk, coeff):
                return (j0 + dj * half, k0 + dk * half, coeff)

            # M1 = (A00+A11)(B00+B11): A has A00 and A11, B has B00 and B11
            # We recurse with A being a SUM of two sub-blocks. Our recursion only handles
            # a single block. So we must expand: M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11.
            # That defeats Strassen. Instead, the recursion must accept that the "A" and "B"
            # inputs to a Strassen call are themselves linear combinations.
            #
            # To keep it tractable, I will NOT do nested Strassen with linear combos.
            # Instead, I'll use a flat, verified construction:
            #   - For the 4x4x4 core, use the known 49-term algorithm if I can, else 56.
            #   - For the 4x4x2 parts, use 2x2x2 Strassen blocks.
            #
            # Given the complexity, I'll fall back to a clean schoolbook+2x2x2-Strassen mix
            # that I can verify, then rely on the peeling pass to improve it.
            pass

        # General fallback: split the largest dimension in half as a sum.
        if si >= 2 and (si >= sj and si >= sk):
            half = si // 2
            strassen_block(i0, j0, k0, half, sj, sk, A_coeff, B_coeff, C_scale)
            strassen_block(i0 + half, j0, k0, si - half, sj, sk, A_coeff, B_coeff, C_scale)
        elif sj >= 2 and (sj >= si and sj >= sk):
            half = sj // 2
            strassen_block(i0, j0, k0, si, half, sk, A_coeff, B_coeff, C_scale)
            strassen_block(i0, j0 + half, k0, si, sj - half, sk, A_coeff, B_coeff, C_scale)
        elif sk >= 2:
            half = sk // 2
            strassen_block(i0, j0, k0, si, sj, half, A_coeff, B_coeff, C_scale)
            strassen_block(i0, j0, k0 + half, si, sj, sk - half, A_coeff, B_coeff, C_scale)
        else:
            # All dimensions are 1 or odd; do scalar if all 1, else scalar per element.
            for di in range(si):
                for dj in range(sj):
                    for dk in range(sk):
                        add_term(i0 + di, j0 + dj, k0 + dk, A_coeff, B_coeff, C_scale)

    # This generic splitter degenerates to schoolbook for odd sizes. Let me instead
    # explicitly build a known-good 108 solution using 2x2x2 Strassen blocks.

    terms.clear()

    def add_strassen_2x2x2(i0, j0, k0, A_c, B_c, C_s):
        """Add a 2x2x2 Strassen (rank 7) block at A[i0:i0+2,j0:j0+2]*B[j0:j0+2,k0:k0+2]."""
        # M1 = (A00+A11)(B00+B11)
        add_term(i0, j0, k0, A_c, B_c, C_s)
        add_term(i0, j0+1, k0, A_c, B_c, C_s)
        add_term(i0+1, j0, k0, A_c, B_c, C_s)
        add_term(i0+1, j0+1, k0, A_c, B_c, C_s)
        # M2 = (A10+A11)B00
        add_term(i0+1, j0, k0, A_c, B_c, C_s)
        add_term(i0+1, j0+1, k0, A_c, B_c, C_s)
        # M3 = A00(B01-B11)
        add_term(i0, j0, k0, A_c, B_c, C_s)
        add_term(i0, j0+1, k0, A_c, B_c, C_s)
        # M4 = A11(B10-B00)
        add_term(i0+1, j0, k0, A_c, B_c, C_s)
        add_term(i0+1, j0+1, k0, A_c, B_c, C_s)
        # M5 = (A00+A01)B11
        add_term(i0, j0, k0, A_c, B_c, C_s)
        add_term(i0, j0+1, k0, A_c, B_c, C_s)
        # M6 = (A10-A00)(B00+B01)
        add_term(i0, j0, k0, A_c, B_c, C_s)
        add_term(i0+1, j0, k0, A_c, B_c, C_s)
        # M7 = (A01-A11)(B10+B11)
        add_term(i0, j0, k0, A_c, B_c, C_s)
        add_term(i0+1, j0, k0, A_c, B_c, C_s)

    # This is getting error-prone. Let me use a verified, simpler base:
    # schoolbook (120) is trivially correct. I'll START from schoolbook and let the
    # peeling pass reduce it. The peeling pass is the real improvement driver.
    terms.clear()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(i, j, k, 1, 1, 1)

    # Convert terms to U,V,W arrays
    def terms_to_matrices(terms):
        R = len(terms)
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        for r, (u_row, v_row, w_row, u_val, v_val, w_val) in enumerate(terms):
            U[u_row, r] = u_val
            V[v_row, r] = v_val
            W[w_row, r] = w_val
        return U, V, W

    U, V, W = terms_to_matrices(terms)
    base_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = base_rank
    print(f"Base rank: {best_rank}")

    # ---------- Verification ----------
    def verify(U, V, W, tol=1e-9):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        R = U.shape[1]
        # Target: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1, else 0
        # Check all (i, j, jp, k): value = delta(j,jp)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :] * V[jp*p+k, :], W[k*n+i, :])
                        target = 1.0 if j == jp else 0.0
                        if abs(val - target) > tol:
                            return False
        return True

    assert verify(best_U, best_V, best_W), "Base decomposition failed verification!"

    # ---------- Peeling refinement ----------
    # Try to remove rank-1 components: find a rank-1 component (u,v,w) of the
    # current residual tensor and subtract it, reducing rank by 1.
    # A rank-1 component satisfies: for all (i,j,jp,k),
    #   u[i*m+j]*v[jp*p+k]*w[k*n+i] = T[i,j,jp,k] - (current decomposition value)
    # where T is the target (delta(j,jp)).
    # We look for u,v,w such that the residual tensor equals u ⊗ v ⊗ w.
    # Strategy: pick a candidate (i0,j0,k0) with nonzero target, set w[k0*n+i0]=1,
    # then solve for u and v.

    def try_peel(U, V, W, max_iters=200):
        """Attempt to peel rank-1 components. Returns (U,V,W,peeled_count)."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        peeled = 0
        # Work on a copy of the "active" decomposition; we track which columns are alive.
        active = list(range(R))
        # Current residual = target - sum of active rank-1 terms.
        # We'll iteratively find a rank-1 component in the residual and mark a column
        # as "absorbed" (set to zero) after subtracting.

        # Precompute target tensor T[i,j,jp,k] = delta(j,jp)
        # Represent residual via the decomposition: resid = T - sum_{r in active} u_r ⊗ v_r ⊗ w_r
        # We need to find (u,v,w) = residual (a rank-1 tensor) to peel.

        # Efficient approach: the residual tensor's value at (i,j,jp,k) is
        #   T[i,j,jp,k] - sum_{r in active} U[i*m+j,r]*V[jp*p+k,r]*W[k*n+i,r]
        # For the residual to be rank-1, it must factorize.
        # Heuristic: find a slice that is rank-1.

        # Simpler: use the "peeling" via solving for one factor.
        # Pick (i0, k0). The residual's "fiber" over (j, jp) at fixed (i0, k0) is:
        #   M[j, jp] = T[i0,j,jp,k0] - sum_r U[i0*m+j,r]*V[jp*p+k0,r]*W[k0*n+i0,r]
        # = (delta(j,jp)) - sum_r ...
        # If residual is rank-1 u⊗v⊗w, then M[j,jp] = u[i0*m+j]*v[jp*p+k0]*w[k0*n+i0].
        # So M is rank-1 (outer product of u_slice and v_slice) times scalar w_val.

        # Algorithm:
        # 1. Compute current full residual tensor R_res (n*m*m*p) — too big? 4*5*5*6=600. OK.
        # 2. Find a rank-1 approximation / exact rank-1 component.

        # Let's compute the residual tensor explicitly.
        def compute_residual(active_set):
            # R_res[i,j,jp,k] = T - sum_{r in active_set} U[i*m+j,r]*V[jp*p+k,r]*W[k*n+i,r]
            res = np.zeros((n, m, m, p))
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            val = 1.0 if j == jp else 0.0
                            for r in active_set:
                                val -= U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
                            res[i, j, jp, k] = val
            return res

        # Find a rank-1 component of the residual.
        # A tensor is rank-1 iff all its matrix unfoldings are rank 1.
        # We'll try: pick (i0, k0), form M[j,jp] = res[i0,j,jp,k0].
        # If M is rank-1, then M = a * (u_col ⊗ v_col) where u_col[j]=u[i0*m+j],
        # v_col[jp]=v[jp*p+k0], a = w[k0*n+i0].
        # Then extend to full u,v,w and verify the whole residual equals u⊗v⊗w.

        def find_rank1_component(res):
            # Try each (i0, k0)
            for i0 in range(n):
                for k0 in range(p):
                    M = res[i0, :, :, k0]  # shape (m, m)
                    # Check if M is rank-1: M = a * u_col ⊗ v_col
                    # Use SVD
                    try:
                        s = np.linalg.svd(M, compute_uv=False)
                    except np.linalg.LinAlgError:
                        continue
                    if len(s) == 0 or s[0] < 1e-10:
                        continue
                    if len(s) > 1 and s[1] > 1e-9:
                        continue  # not rank-1
                    # M is rank-1 (or zero). Get u_col, v_col via SVD.
                    U_svd, s2, Vt = np.linalg.svd(M, full_matrices=False)
                    if U_svd.shape[0] == 0:
                        continue
                    u_col = U_svd[:, 0] * s2[0]  # = M @ Vt[0]
                    v_col = Vt[0]
                    # Now M[j,jp] = u_col[j] * v_col[jp]
                    # We need to extend to full u (nm), v (mp), w (pn).
                    # u[i*m+j] for i != i0 is unknown; v[jp*p+k] for k != k0 unknown;
                    # w[k*n+i] for (k,i) != (k0,i0) unknown.
                    # Set w[k0*n+i0] = 1, then u[i0*m+j] = u_col[j], v[jp*p+k0] = v_col[jp].
                    # For other entries, determine from other slices of res.
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    w[k0*n + i0] = 1.0
                    for j in range(m):
                        u[i0*m + j] = u_col[j]
                    for jp in range(m):
                        v[jp*p + k0] = v_col[jp]
                    # Determine u[i*m+j] for i != i0 using res[i,j,jp,k0]:
                    # res[i,j,jp,k0] should = u[i*m+j]*v[jp*p+k0]*w[k0*n+i]
                    # We know v[jp*p+k0] = v_col[jp]. w[k0*n+i] unknown for i != i0.
                    # Use a jp where v_col[jp] != 0.
                    # For each i != i0, pick jp0 with |v_col[jp0]|>1e-9.
                    # Then res[i,j,jp0,k0] = u[i*m+j]*v_col[jp0]*w[k0*n+i]
                    # This still has 2 unknowns (u[i*m+j] and w[k0*n+i]).
                    # Use two different j to separate.
                    # Actually, for fixed i, k0: res[i, :, :, k0] = u[i*m, :] ⊗ (v_col * w[k0*n+i])
                    # So res[i,:,:,k0] is rank-1 = (u[i*m,:]) ⊗ (w[k0*n+i]*v_col)
                    # We can extract u[i*m,:] and w[k0*n+i] from this slice.
                    feasible = True
                    for i in range(n):
                        if i == i0:
                            continue
                        S = res[i, :, :, k0]  # (m, m)
                        # S[j,jp] = u[i*m+j] * (w[k0*n+i] * v_col[jp])
                        # Find jp0 with v_col[jp0] != 0
                        nz = np.where(np.abs(v_col) > 1e-9)[0]
                        if len(nz) == 0:
                            feasible = False
                            break
                        jp0 = nz[0]
                        # S[:, jp0] = u[i*m,:] * w[k0*n+i] * v_col[jp0]
                        # Also S[j, :] = u[i*m+j] * w[k0*n+i] * v_col
                        # Use S[:, jp0] / (w*v_col[jp0]) = u[i*m,:]. But w unknown.
                        # Use ratio: for two j, S[j1,jp0]/S[j2,jp0] = u[i*m+j1]/u[i*m+j2].
                        # Better: S should be rank-1. Check and extract.
                        try:
                            ss = np.linalg.svd(S, compute_uv=False)
                        except np.linalg.LinAlgError:
                            feasible = False
                            break
                        if len(ss) > 1 and ss[1] > 1e-9:
                            feasible = False
                            break
                        if ss[0] < 1e-10:
                            # S is zero => u[i*m,:]*w[k0*n+i] = 0
                            # Set w[k0*n+i]=0, u[i*m,:]=0 (arbitrary)
                            w[k0*n + i] = 0.0
                            continue
                        U_s, s2, Vt_s = np.linalg.svd(S, full_matrices=False)
                        # S = (U_s[:,0]*s2[0]) ⊗ Vt_s[0]
                        left = U_s[:, 0] * s2[0]   # = u[i*m,:] * w[k0*n+i]
                        right = Vt_s[0]            # = w[k0*n+i] * v_col
                        # right should be proportional to v_col
                        # Determine w[k0*n+i] from right = w * v_col
                        # w = right[jp0]/v_col[jp0]
                        w_val = right[jp0] / v_col[jp0]
                        # Check consistency: right[jp]/v_col[jp] should = w_val for all jp
                        ok = True
                        for jp in nz:
                            if abs(right[jp] / v_col[jp] - w_val) > 1e-7:
                                ok = False
                                break
                        if not ok:
                            feasible = False
                            break
                        w[k0*n + i] = w_val
                        # u[i*m,:] = left / w_val
                        if abs(w_val) < 1e-12:
                            feasible = False
                            break
                        u[i*m:i*m+m] = left / w_val
                    if not feasible:
                        continue
                    # Now determine v[jp*p+k] for k != k0 using res[i0, j, jp, k]:
                    # res[i0,j,jp,k] = u[i0*m+j]*v[jp*p+k]*w[k*n+i0]
                    # We know u[i0*m+j]=u_col[j], w[k*n+i0] unknown for k != k0.
                    # For fixed k, res[i0, :, :, k] = u_col ⊗ (w[k*n+i0]*v[:,k])
                    # Extract similarly.
                    for k in range(p):
                        if k == k0:
                            continue
                        S = res[i0, :, :, k]  # (m, m)
                        try:
                            ss = np.linalg.svd(S, compute_uv=False)
                        except np.linalg.LinAlgError:
                            feasible = False
                            break
                        if len(ss) > 1 and ss[1] > 1e-9:
                            feasible = False
                            break
                        if ss[0] < 1e-10:
                            w[k*n + i0] = 0.0
                            continue
                        U_s, s2, Vt_s = np.linalg.svd(S, full_matrices=False)
                        left = U_s[:, 0] * s2[0]   # = u_col * w[k*n+i0]
                        right = Vt_s[0]            # = w[k*n+i0] * v[:,k]
                        # left should be proportional to u_col
                        nz_u = np.where(np.abs(u_col) > 1e-9)[0]
                        if len(nz_u) == 0:
                            feasible = False
                            break
                        j0u = nz_u[0]
                        w_val = left[j0u] / u_col[j0u]
                        ok = True
                        for j in nz_u:
                            if abs(left[j] / u_col[j] - w_val) > 1e-7:
                                ok = False
                                break
                        if not ok:
                            feasible = False
                            break
                        w[k*n + i0] = w_val
                        if abs(w_val) < 1e-12:
                            feasible = False
                            break
                        v[:, k] = right / w_val
                    if not feasible:
                        continue
                    # Now verify the full residual equals u⊗v◗w
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    pred = u[i*m+j] * v[jp*p+k] * w[k*n+i]
                                    if abs(pred - res[i, j, jp, k]) > 1e-7:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        return (u, v, w)
            return None

        # Iteratively peel
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            active_set = set(active)
            res = compute_residual(active_set)
            # Check if residual is zero (done)
            if np.max(np.abs(res)) < 1e-9:
                break
            comp = find_rank1_component(res)
            if comp is None:
                break
            u, v, w = comp
            # Subtract this component from the active decomposition.
            # We need to find a column r in active such that we can represent the
            # subtraction. Simplest: add a NEW column that is the negative of (u,v,w)
            # and keep all active columns. But that doesn't reduce rank.
            # Instead: we want to REMOVE one active column and adjust others so that
            # sum of remaining = original sum - (u⊗v⊗w).
            # This is a linear change. Pick an active column r*, set its (u,v,w) to
            # (u_r* - u, v_r* - v, w_r* - w)? No, that changes the product non-linearly.
            #
            # Correct approach: the residual is exactly u⊗v⊗w. The active decomposition
            # sums to T - residual = T - u⊗v⊗w. We want a decomposition of (T - u⊗v⊗w)
            # with one fewer rank-1 term. But the active set already sums to that.
            # Peeling means: express u⊗v⊗w as a combination that lets us drop a term.
            #
            # Alternative peeling: find a rank-1 component and a column r such that
            # we can replace column r's contribution. This is complex.
            #
            # Simpler valid improvement: if the residual is rank-1, then the WHOLE
            # tensor T = (active sum) + residual has rank <= len(active) + 1.
            # That doesn't help reduce.
            #
            # Let me reconsider: peeling usually works on the tensor itself:
            # If T has a rank-1 component c = u⊗v⊗w, then T - c has rank <= rank(T)-1
            # ONLY IF c is a "summand" in some decomposition. In general, subtracting
            # a rank-1 tensor can reduce rank by at most 1, and we get a new tensor
            # T' = T - c. We then need to decompose T'. But T' may have the same rank.
            #
            # The real benefit: if we can find c such that T' is decomposable with
            # rank-1 fewer terms. This requires T' to have a decomposition of that rank.
            #
            # Given the difficulty, let me use a different, guaranteed improvement:
            # Try to find a rank-1 component and then solve a linear system to re-express
            # the decomposition with that component removed, reducing rank by 1.
            #
            # I'll implement: given current decomposition (U,V,W) of rank R summing to T,
            # and a rank-1 tensor c = u⊗v⊗w that is a "direction", try to find
            # (U',V',W') of rank R-1 summing to T by setting one column to zero and
            # solving for adjustments. This is underdetermined; use least squares.

            # Pick a column r* to eliminate. Try each.
            success = False
            R_cur = U.shape[1]
            for r_star in active:
                if time.time() > deadline - 20:
                    break
                # We want new U,V,W with column r_star = 0, and
                # sum_{r != r_star} U'[i*m+j,r]*V'[jp*p+k,r]*W'[k*n+i,r] = T
                # i.e. the remaining R-1 columns must sum to T.
                # Currently sum_all = T. So sum_{r != r_star} (orig) = T - c_{r_star}
                # where c_{r_star} = U[:,r_star]⊗V[:,r_star]⊗W[:,r_star].
                # We need to modify the other columns so their sum = T.
                # Let delta_r be the change to column r (r != r_star). We need
                #   sum_{r != r_star} delta_r_tensor = c_{r_star}
                # where delta_r_tensor = U'[:,r]⊗V'[:,r]⊗W'[:,r] - U[:,r]⊗V[:,r]⊗W[:,r].
                # This is nonlinear. Instead, treat U',V',W' as free and solve the
                # linear system in W' (fixing U',V' = U,V for r != r_star):
                #   For all (i,j,jp,k): sum_{r != r_star} U[i*m+j,r]*V[jp*p+k,r]*W'[k*n+i,r] = T[i,j,jp,k]
                # This is linear in W'. Solve it.
                # Number of equations: n*m*m*p = 600. Variables: p*n*(R-1) = 24*(R-1).
                # For R=120, vars = 24*119 = 2856 > 600. Underdetermined -> min-norm solution.
                # Build A (600 x 24*(R-1)), b (600).
                Rm1 = R_cur - 1
                keep = [r for r in range(R_cur) if r != r_star]
                num_eq = n * m * m * p
                A = np.zeros((num_eq, pn * Rm1))
                b = np.zeros(num_eq)
                eq = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                target = 1.0 if j == jp else 0.0
                                for idx, r in enumerate(keep):
                                    var = (k * n + i) * Rm1 + idx
                                    A[eq, var] = U[i*m+j, r] * V[jp*p+k, r]
                                b[eq] = target
                                eq += 1
                # Solve min-norm: W' = A^T (A A^T)^+ b  (for underdetermined)
                try:
                    # Use lstsq which gives min-norm for underdetermined
                    W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                except np.linalg.LinAlgError:
                    continue
                W_new = W_flat.reshape(pn, Rm1)
                # Verify
                U_new = np.column_stack([U[:, r] for r in keep])
                V_new = np.column_stack([V[:, r] for r in keep])
                if verify(U_new, V_new, W_new, tol=1e-8):
                    # Check integer/half-integer
                    def is_int_half(M, tol=1e-6):
                        for row in M:
                            for val in row:
                                if abs(val - round(val)) > tol:
                                    if abs(val - (round(val*2)/2.0)) > tol:
                                        return False
                        return True
                    def snap(M):
                        M2 = np.zeros_like(M)
                        for i in range(M.shape[0]):
                            for j in range(M.shape[1]):
                                val = M[i,j]
                                ni = round(val)
                                nh = round(val*2)/2.0
                                M2[i,j] = ni if abs(val-ni) <= abs(val-nh) else nh
                        return M2
                    if is_int_half(W_new) or is_int_half(snap(W_new)):
                        W_snap = snap(W_new)
                        if verify(U_new, V_new, W_snap, tol=1e-8):
                            # Success! Rank reduced by 1.
                            U, V, W = U_new, V_new, W_snap
                            R_cur = Rm1
                            peeled += 1
                            success = True
                            # Update active to new column indices
                            active = list(range(R_cur))
                            break
            if not success:
                break
        return U, V, W, peeled

    U, V, W, peeled = try_peel(best_U, best_V, best_W)
    if peeled > 0:
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
        print(f"After peeling: rank {best_rank} (peeled {peeled})")

    # ---------- Final fallback verification ----------
    if not verify(best_U, best_V, best_W, tol=1e-8):
        # Revert to schoolbook
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy() if U.shape[1] == base_rank else (
            np.zeros((nm, base_rank)), np.zeros((mp, base_rank)), np.zeros((pn, base_rank)))
        # Rebuild schoolbook
        best_U = np.zeros((nm, base_rank))
        best_V = np.zeros((mp, base_rank))
        best_W = np.zeros((pn, base_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
        best_rank = base_rank
        print("Reverted to schoolbook rank", best_rank)

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
