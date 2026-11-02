"""Bootstrap MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Strategy: Use block-Strassen (rank 189) as baseline, then attempt rank reduction
using proper tensor product elimination and alternative block decompositions.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses block-Strassen as baseline and attempts rank reduction.
    """
    deadline = time.time() + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build block-Strassen decomposition (rank 189)
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    best_U, best_V, best_W = U_bs.copy(), V_bs.copy(), W_bs.copy()
    best_rank = rank_bs
    
    if time.time() > deadline - 15:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Verify the decomposition is correct
    def verify(U, V, W, tol=1e-9):
        """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta(j matches) for all i,j,k"""
        # Actually, the tensor should satisfy: for all i,j,k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is the (i,k) entry from j
        # The matrix multiplication tensor T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
        # So we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
        # Wait, that's not right either. Let me think again.
        # The tensor for matrix mult: T_{i,j,k} where the contraction is sum_j A_{i,j} B_{j,k} = C_{i,k}
        # In our factorization: C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_k V_{j,k,r} B_{j,k}) W_{k,i,r}
        # Hmm, the indices are getting confused. Let me just check numerically.
        
        # The correct check: for random A (n x m) and B (m x p), verify A@B matches
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        # Compute using decomposition
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
        # Wait, this doesn't look right. Let me re-examine the convention.
        
        # The task says: W[k*n+i, r] is the output factor.
        # Standard tensor decomposition for matrix mult:
        # C_{i,k} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}  -- no, that's not standard.
        
        # Actually, for the matrix multiplication tensor, we have:
        # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # A rank-r decomposition gives:
        # (A*B)_{i,k} = sum_{r=1}^R (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k}) * W_{k,i,r}
        # Hmm, but that has the same j index in both sums which is wrong.
        
        # Let me reconsider. The standard form is:
        # (A*B)_{i,k} = sum_r [sum_j U_{(i,j),r} A_{i,j}] * [sum_l V_{(l,k),r} B_{l,k}] * W_{(k,i),r}
        # No wait, that's still not right because the inner indices need to match.
        
        # I think the correct interpretation is:
        # (A*B)_{i,k} = sum_r W_{(k,i),r} * (sum_j U_{(i,j),r} * A_{i,j}) * (sum_j V_{(j,k),r} * B_{j,k})
        # But this requires the j index to be the same in both sums, which means U and V 
        # are indexed by the same j. Let me just verify numerically with the schoolbook.
        
        # For schoolbook: U[i*m+j, col(i',j',k')] = 1 if (i,j)=(i',j'), etc.
        # So for a given (i,k): sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
        # For the column corresponding to (i,j,k): W[k*n+i,r]=1, U[i*m+j,r]=1 (only if matches), V[j*p+k,r]=1 (only if matches)
        # So the term is 1*1*1 = 1 when (i,j,k) matches, giving sum_j A[i,j]*B[j,k]. Correct!
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(U.shape[1]):
                    if abs(W[k*n+i, r]) < 1e-15:
                        continue
                    u_val = sum(U[i*m+j, r] * A[i,j] for j in range(m))
                    v_val = sum(V[j*p+k, r] * B[j,k] for j in range(m))
                    C_computed[i, k] += W[k*n+i, r] * u_val * v_val
        
        return np.allclose(C_computed, C_expected, atol=tol)
    
    # Verify block-Strassen
    if not verify(U_bs, V_bs, W_bs):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_schoolbook
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt rank reduction via proper tensor elimination
    # A column r can be eliminated if there exist coefficients alpha_s, beta_s, gamma_s 
    # such that the rank-1 tensor u_r ⊗ v_r ⊗ w_r can be expressed as a linear combination
    # of other rank-1 tensors. A sufficient condition: if u_r is in span of other u's,
    # v_r is in span of other v's, and w_r is in span of other w's, AND the combination
    # is consistent.
    
    # Practical approach: try to find if column r equals a linear combination of 
    # at most 2 other columns in ALL three matrices simultaneously.
    
    remaining = list(range(best_rank))
    
    # Sort by norm to eliminate small-norm columns first (more likely redundant)
    col_norms = np.linalg.norm(best_U, axis=0) * np.linalg.norm(best_V, axis=0) * np.linalg.norm(best_W, axis=0)
    remaining.sort(key=lambda r: col_norms[r])
    
    max_elim_attempts = 30
    elim_count = 0
    
    while remaining and elim_count < max_elim_attempts:
        if time.time() > deadline - 10:
            break
        
        eliminated_this_round = False
        
        for idx_r in range(len(remaining)):
            if time.time() > deadline - 10:
                break
            
            r = remaining[idx_r]
            others = [c for c in remaining if c != r]
            
            if len(others) < 2:
                continue
            
            # Try to express U[:,r] as combination of up to 2 other columns
            # and check consistency with V and W
            
            # First, check if U[:,r] is in the span of others
            U_rest = best_U[:, others]
            V_rest = best_V[:, others]
            W_rest = best_W[:, others]
            
            try:
                x_u, _, _, _ = np.linalg.lstsq(U_rest, best_U[:, r], rcond=None)
            except:
                continue
            
            # Check how well this x_u predicts V and W
            v_pred = V_rest @ x_u
            w_pred = W_rest @ x_u
            
            res_u = np.linalg.norm(best_U[:, r] - U_rest @ x_u)
            res_v = np.linalg.norm(best_V[:, r] - v_pred)
            res_w = np.linalg.norm(best_W[:, r] - w_pred)
            
            # For the tensor product to be preserved, we need a stronger condition.
            # The product u_r ⊗ v_r ⊗ w_r should be in the span of {u_s ⊗ v_s ⊗ w_s}.
            # If u_r = U_rest @ x, v_r = V_rest @ x, w_r = W_rest @ x, then
            # u_r ⊗ v_r ⊗ w_r is NOT generally in the span of the products.
            # So this simple check is insufficient.
            
            # Instead, use a necessary condition: the trilinear form must match.
            # For all (i,j,k): U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal
            # sum_s x_u[s] * U[i*m+j,others[s]]*V[j*p+k,others[s]]*W[k*n+i,others[s]]
            # This is equivalent to checking that the tensor entry matches.
            
            # Compute the difference tensor
            diff = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff[i,j,k] = (best_U[i*m+j, r] * best_V[j*p+k, r] * best_W[k*n+i, r] -
                                       np.sum(x_u * best_U[i*m+j, others] * best_V[j*p+k, others] * best_W[k*n+i, others]))
            
            if np.max(np.abs(diff)) < 1e-8:
                # Column r can be eliminated
                remaining.remove(r)
                elim_count += 1
                eliminated_this_round = True
                break
        
        if not eliminated_this_round:
            break
    
    if len(remaining) < best_rank:
        final_U = best_U[:, remaining]
        final_V = best_V[:, remaining]
        final_W = best_W[:, remaining]
        
        # Verify the reduced decomposition
        if verify(final_U, final_V, final_W):
            best_U, best_V, best_W = final_U, final_V, final_W
            best_rank = len(remaining)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
