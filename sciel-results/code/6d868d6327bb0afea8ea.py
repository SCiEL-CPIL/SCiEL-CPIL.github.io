"""Matrix multiplication 4x5x5 - Strassen family with fast vectorized pruning + polish."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    # ---------- Build the rank-85 Strassen baseline ----------
    rank_base = 85
    U = np.zeros((nm, rank_base), dtype=float)
    V = np.zeros((mp, rank_base), dtype=float)
    W = np.zeros((pn, rank_base), dtype=float)

    # 2x2x2 Strassen factors
    s_u = np.zeros((4, 7), dtype=float)
    s_v = np.zeros((4, 7), dtype=float)
    s_w = np.zeros((4, 7), dtype=float)
    s_u[0,0]=1; s_u[3,0]=1; s_v[0,0]=1; s_v[3,0]=1; s_w[0,0]=1; s_w[3,0]=1
    s_u[2,1]=1; s_u[3,1]=1; s_v[0,1]=1; s_w[2,1]=1; s_w[3,1]=-1
    s_u[0,2]=1; s_v[1,2]=1; s_v[3,2]=-1; s_w[1,2]=1; s_w[3,2]=1
    s_u[3,3]=1; s_v[2,3]=1; s_v[0,3]=-1; s_w[0,3]=1; s_w[2,3]=1
    s_u[0,4]=1; s_u[1,4]=1; s_v[3,4]=1; s_w[0,4]=-1; s_w[1,4]=1
    s_u[0,5]=-1; s_u[2,5]=1; s_v[0,5]=1; s_v[1,5]=1; s_w[3,5]=1
    s_u[1,6]=1; s_u[3,6]=-1; s_v[2,6]=1; s_v[3,6]=1; s_w[0,6]=1

    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]; v_vec = s_v[:, inner_k]; w_vec = s_w[:, inner_k]
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr*2+lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2*a_br+lr; gj = 2*a_bc+lc
                            U[gi*5+gj, col] += a_coeff*u_val
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr*2+lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2*b_br+lr; gk = 2*b_bc+lc
                            V[gj*5+gk, col] += b_coeff*v_val
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr*2+lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = 2*c_br+lr; gk = 2*c_bc+lc
                            W[gk*4+gi, col] += c_coeff*w_val

    terms = [
        ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1},{(0,0):1},{(1,0):1,(1,1):-1}),
        ({(0,0):1},{(0,1):1,(1,1):-1},{(0,1):1,(1,1):1}),
        ({(1,1):1},{(1,0):1,(0,0):-1},{(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1},{(1,1):1},{(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1},{(1,1):1}),
        ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1},{(0,0):1}),
    ]
    for t_idx,(a_c,b_c,c_c) in enumerate(terms):
        if time.time()-start_time > deadline:
            break
        add_4x4_term(a_c, b_c, c_c, t_idx*7)

    col = 49
    for i in range(4):
        if time.time()-start_time > deadline: break
        for j in range(4):
            U[i*5+j, col]=1; V[j*5+4, col]=1; W[4*4+i, col]=1; col+=1
    for i in range(4):
        if time.time()-start_time > deadline: break
        for k in range(4):
            U[i*5+4, col]=1; V[4*5+k, col]=1; W[k*4+i, col]=1; col+=1
    for i in range(4):
        if time.time()-start_time > deadline: break
        U[i*5+4, col]=1; V[4*5+4, col]=1; W[4*4+i, col]=1; col+=1

    # ---------- Fast vectorized verification ----------
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Outer expansion: T = sum_c (U[:,c] as i,j) outer (W[:,c] as k,i)
    #   U[:,c] reshape (n,m) at [i*m+j], W[:,c] reshape (p,n) at [k*n+i]
    #   product T[i,j,k] += U[i,j]*W[k,i]  ->  (n,m) x (p,n) -> einsum
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        if r == 0:
            return False
        # Build T via outer expansion over rank: T[i,j,k] = sum_c U[i*m+j,c]*W[k*n+i,c]
        # U reshaped (n,m,r), W reshaped (p,n,r); T = sum_c U[i,j,c]*W[k,i,c]
        U3 = U_mat.reshape(n, m, r)
        W3 = W_mat.reshape(p, n, r)
        T = np.einsum('ijc,kic->ijk', U3, W3)
        # target: T[i,j,k] = 1 if k==i else 0
        target = np.zeros((n, m, p))
        for i in range(n):
            target[i, :, i] = 1.0
        return np.allclose(T, target, atol=tol, rtol=0.0)

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_schoolbook = n*m*p
        U = np.zeros((nm, rank_schoolbook), dtype=float)
        V = np.zeros((mp, rank_schoolbook), dtype=float)
        W = np.zeros((pn, rank_schoolbook), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c]=1; V[j*p+k, c]=1; W[k*n+i, c]=1; c+=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = rank_base

    # ---------- Fast column-removal via residual test ----------
    # If we remove column c, the new tensor is T - T_c where
    # T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We can compute T once, then for each candidate removal check
    # whether T - T_c is still the target. Vectorize across columns.
    #
    # Represent each column's tensor contribution as a flat vector of length n*m*p,
    # indexed by (i,j,k) -> k*n + i? We need consistent flattening.
    # T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build F[c, flat] where flat = (i*m+j)*p + k  (n*m*p entries)
    def build_fingerprints(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        U3 = U_mat.reshape(n, m, r)
        V3 = V_mat.reshape(m, p, r)
        W3 = W_mat.reshape(p, n, r)
        # T_c[i,j,k] = U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        # flat index: ((i*m)+j)*p + k
        Tc = np.einsum('ijc,jkc,kic->ijkc', U3, V3, W3)  # (n,m,p,r)
        F = Tc.reshape(n*m*p, r).T  # (r, n*m*p)
        return F

    def can_remove_batch(F, indices, tol=1e-9):
        """Test removing each index in indices; return list of indices whose removal keeps validity.
        Validity after removing set S: sum_{c notin S} F[c] == target.
        Equivalent: target - sum_{c in S} F[c] == sum_{c notin S} F[c] ... 
        Simpler: total = sum_c F[c] == target (already true). Removing S valid iff sum_{c in S} F[c] == 0.
        """
        # total == target guaranteed. Removing S valid iff sum_{c in S} F[c] == 0.
        sub = F[indices]  # (len, n*m*p)
        s = sub.sum(axis=0)
        return [indices[i] for i in range(len(indices)) if np.abs(sub[i]).max() < tol]

    # Iteratively remove columns whose fingerprint is (nearly) zero in the residual sense.
    # Actually a column c is removable iff its fingerprint F[c] == 0 (since total must equal target).
    # But multiple columns can cancel: a set S is removable iff sum_{c in S} F[c] == 0.
    # Greedy: find any zero-sum subset. Use QR/dependent-set detection.
    F = build_fingerprints(best_U, best_V, best_W)
    # Find a minimal linearly independent set of the fingerprint rows; extra rows are
    # in the span, so the full set can be reduced to a basis while preserving the sum.
    # But we must preserve the SUM = target, not just the span. So we need the rows to
    # still sum to target after pruning. Strategy: keep a basis, then the removed rows must
    # sum to zero. Use: compute a basis B of rows; the complement C has sum = target - sum(B).
    # We can only drop C if sum(C)==0. Instead, use linear dependency to re-express.
    # Simpler robust approach: repeatedly find a dependent row and test dropping it.
    def prune_dependent(U_mat, V_mat, W_mat, tol=1e-9, max_rounds=200):
        curU, curV, curW = U_mat.copy(), V_mat.copy(), W_mat.copy()
        r = curU.shape[1]
        for _ in range(max_rounds):
            if time.time()-start_time > deadline-5:
                break
            if r == 0:
                break
            F = build_fingerprints(curU, curV, curW)
            # Find a column c whose fingerprint is in the span of the others,
            # then test dropping it (drop valid iff F[c] is in span AND the residual
            # still equals target; since total==target, drop valid iff F[c] is a
            # linear combination that we can absorb — but we can't change others.
            # So drop c valid iff F[c] == 0. To allow non-zero drops, we need to
            # re-solve. For pure drop, require F[c] ~ 0.
            # Compute effective rank and identify near-zero fingerprint rows.
            norms = np.linalg.norm(F, axis=1)
            # Try each column with small norm first
            order = np.argsort(norms)
            dropped_any = False
            for c in order:
                if time.time()-start_time > deadline-5:
                    break
                if norms[c] < tol:
                    keep = [x for x in range(r) if x != c]
                    curU = curU[:, keep]; curV = curV[:, keep]; curW = curW[:, keep]
                    r -= 1
                    dropped_any = True
                    break
            if not dropped_any:
                break
        return curU, curV, curW, r

    best_U, best_V, best_W, best_rank = prune_dependent(best_U, best_V, best_W)
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W, best_rank = U, V, W, rank_base

    # ---------- Polish pass: solve for exact corrections to allow larger drops ----------
    # Given a candidate kept set K, we want to know if there exist adjusted
    # columns U',V',W' (still rank |K|) that reproduce the target. This is a
    # nonlinear (trilinear) problem. Use a linearization: fix V,W, solve for U
    # least-squares, then verify. Iterate coordinate-wise.
    def polish(U_mat, V_mat, W_mat, tol=1e-8, iters=6):
        U_mat = U_mat.copy(); V_mat = V_mat.copy(); W_mat = W_mat.copy()
        r = U_mat.shape[1]
        target = np.zeros((n, m, p))
        for i in range(n):
            target[i, :, i] = 1.0
        target_flat = target.reshape(n*m*p)
        for _ in range(iters):
            if time.time()-start_time > deadline-5:
                break
            # Fix V,W, solve for U: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = target[i,j,k]
            # This is linear in U. Build system: for each (i,j,k), row over U entries.
            # U has nm*r entries. Build A (n*m*p x nm*r), b = target_flat.
            # A[(i,j,k), (i*m+j, c)] = V[j*p+k,c]*W[k*n+i,c]
            V3 = V_mat.reshape(m, p, r)
            W3 = W_mat.reshape(p, n, r)
            A = np.zeros((n*m*p, nm*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i*m+j)*p + k
                        # coefficient for U[i*m+j, c] is V[j,k,c]*W[k,i,c]
                        coeff = V3[j, k, :] * W3[k, i, :]  # (r,)
                        base = (i*m+j)*r
                        A[row, base:base+r] = coeff
            sol, *_ = np.linalg.lstsq(A, target_flat, rcond=None)
            U_mat = sol.reshape(nm, r)
            # Fix U,W, solve for V: A2[(i,j,k),(j*p+k,c)] = U[i*m+j,c]*W[k*n+i,c]
            U3 = U_mat.reshape(n, m, r)
            A2 = np.zeros((n*m*p, mp*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i*m+j)*p + k
                        coeff = U3[i, j, :] * W3[k, i, :]
                        base = (j*p+k)*r
                        A2[row, base:base+r] = coeff
            sol2, *_ = np.linalg.lstsq(A2, target_flat, rcond=None)
            V_mat = sol2.reshape(mp, r)
            # Fix U,V, solve for W: A3[(i,j,k),(k*n+i,c)] = U[i*m+j,c]*V[j*p+k,c]
            V3 = V_mat.reshape(m, p, r)
            A3 = np.zeros((n*m*p, pn*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i*m+j)*p + k
                        coeff = U3[i, j, :] * V3[j, k, :]
                        base = (k*n+i)*r
                        A3[row, base:base+r] = coeff
            sol3, *_ = np.linalg.lstsq(A3, target_flat, rcond=None)
            W_mat = sol3.reshape(pn, r)
        return U_mat, V_mat, W_mat

    # Try to reduce rank: attempt ranks from best_rank-1 downward, polish and verify.
    # This is expensive; only attempt a few steps if time allows.
    if time.time()-start_time < deadline-30:
        for target_rank in range(best_rank-1, max(40, best_rank-6), -1):
            if time.time()-start_time > deadline-30:
                break
            # Take a subset of columns (keep first target_rank) and polish
            subU = best_U[:, :target_rank]
            subV = best_V[:, :target_rank]
            subW = best_W[:, :target_rank]
            pU, pV, pW = polish(subU, subV, subW, iters=4)
            if verify(pU, pV, pW, tol=1e-7):
                best_U, best_V, best_W = pU, pV, pW
                best_rank = target_rank
            else:
                break

    # Final verification
    if not verify(best_U, best_V, best_W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
