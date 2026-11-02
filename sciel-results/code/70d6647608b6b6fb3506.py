import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    # Build the base rank-50 solution using Strassen block decomposition
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1

        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1

        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1

        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1

        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1

        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1

        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify base solution
    if not verify(U, V, W):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Tensor Mode-3 Unfolding and Null-Space Projection Strategy
    # The multiplication tensor T is in R^{(n*m) x (m*p) x (p*n)} with
    # T[i*m+j, jp*p+k, k*n+i] = delta(j, jp)
    # 
    # Mode-3 unfolding: flatten to matrix of size (n*m*p) x (p*n) where
    # M[(i*m+j)*(m*p) + (jp*p+k), k*n+i] = T[i*m+j, jp*p+k, k*n+i]
    # Actually, mode-3 unfolding contracts the third mode (size p*n=14) to columns,
    # and the first two modes (sizes n*m=8 and m*p=28) to rows.
    # So M is 224 x 14, with M[(i*m+j)*(m*p) + (jp*p+k), k*n+i] = delta(j,jp)
    
    # Build the mode-3 unfolding matrix
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14
    
    M3 = np.zeros((nm * mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    row = (i * m + j) * mp + (jp * p + k)
                    col_idx = k * n + i
                    if j == jp:
                        M3[row, col_idx] = 1
    
    # Compute the rank and null space of M3
    # M3 is 224 x 14. Its rank is at most 14.
    # The null space of M3^T (size 14 x 224) gives linear dependencies among the rows.
    # But we want to find if we can reduce the rank of the tensor decomposition.
    
    # Key insight: The tensor rank is bounded below by the matrix rank of any unfolding.
    # rank(tensor) >= rank(M3) for any mode-k unfolding.
    # Let's compute rank(M3) to get a lower bound.
    
    # Use SVD to find the rank
    U_svd, s_vals, Vt_svd = np.linalg.svd(M3.astype(float), full_matrices=False)
    rank_M3 = np.sum(s_vals > 1e-10)
    
    # The tensor rank must be at least rank_M3.
    # If rank_M3 < 50, we might be able to reduce the rank.
    
    # Strategy: Use the null space of M3 to find linear combinations of the output
    # monomials that sum to zero. These can be used to "cancel" terms in the decomposition.
    
    # The null space of M3^T has dimension 224 - rank(M3).
    # Each vector in this null space corresponds to a linear relation among the 224
    # "atomic" products U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c] for a fixed c.
    
    # Actually, let's think differently. We have a rank-50 decomposition. We want to
    # find a rank-r decomposition with r < 50. 
    # 
    # Approach: Try to express some of the 50 rank-1 tensors as linear combinations
    # of others, using the null space information.
    # 
    # For each term c in our decomposition, define the vector v_c in R^{nm*mp} by
    # v_c[(i*m+j)*mp + (jp*p+k)] = U[i*m+j,c] * V[jp*p+k,c]
    # Then the contribution of term c to the mode-3 unfolding is v_c * W_c^T (outer product).
    # The sum over all c of v_c * W_c^T should equal M3.
    
    # Let's build the matrix V_mat where V_mat[:, c] = v_c (flattened)
    V_mat = np.zeros((nm * mp, rank), dtype=np.int64)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        row = (i * m + j) * mp + (jp * p + k)
                        V_mat[row, c] = U[i*m+j, c] * V[jp*p+k, c]
    
    # Now we have V_mat (224 x 50) and W (14 x 50).
    # The constraint is: V_mat @ W.T = M3
    # We want to find a lower-rank factorization.
    
    # Check if V_mat has a null space that can be exploited.
    # If there's a vector a in R^50 such that V_mat @ a = 0, then we can replace
    # the 50 terms with 49 terms by combining along a.
    
    # Compute the null space of V_mat
    U_vm, s_vm, Vt_vm = np.linalg.svd(V_mat.astype(float), full_matrices=True)
    rank_V_mat = np.sum(s_vm > 1e-10)
    
    # The null space of V_mat has dimension 50 - rank_V_mat.
    # If this is > 0, we have linear dependencies among the 50 "V-terms".
    
    # For each null space vector a, we have V_mat @ a = 0, which means
    # sum_c a_c * v_c = 0. This doesn't directly help unless W also has structure.
    
    # Better approach: Look for a vector a such that both V_mat @ a = 0 AND W @ a = 0.
    # This would mean the 50th term is a linear combination of the other 49 in a way
    # that doesn't affect the product.
    
    # Actually, the correct condition is: if there exist vectors a, b such that
    # V_mat @ a = 0 and W @ a = 0, then term a can be removed.
    # But this is too restrictive.
    
    # Let's try a different approach: use the fact that the tensor rank is the minimum
    # r such that there exist U, V, W of rank r that satisfy the constraint.
    # 
    # We can formulate this as: minimize r subject to V_mat @ W.T = M3, where
    # V_mat and W are constrained to have rank at most r.
    # 
    # This is a non-convex problem. Let's use a heuristic: try to remove terms one by one
    # and check if the remaining terms can still satisfy the constraint (possibly with
    # adjusted values).
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each term and solving for the remaining
    for remove_idx in range(rank):
        if time.time() > deadline - 10:
            break
        
        # Keep all terms except remove_idx
        keep = [c for c in range(rank) if c != remove_idx]
        r_new = len(keep)
        
        if r_new >= best_rank:
            continue
        
        # Build the constraint matrix for the remaining terms
        # We need: sum_{c in keep} U[:,c] * V[:,c]^T (outer, then contract with W) = M3
        # This is still nonlinear. Let's fix W and solve for U, V, or use a different approach.
        
        # Alternative: check if the removed term is "redundant" by seeing if its contribution
        # can be matched by the other terms.
        
        # Compute the contribution of term remove_idx to M3
        contrib = np.zeros((nm * mp, pn), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        row = (i * m + j) * mp + (jp * p + k)
                        for ki in range(p):
                            for ii in range(n):
                                col_idx = ki * n + ii
                                contrib[row, col_idx] += U[i*m+j, remove_idx] * V[jp*p+k, remove_idx] * W[ki*n+ii, remove_idx]
        
        # Check if this contribution is zero (term is redundant)
        if np.all(contrib == 0):
            # Term is redundant, remove it
            best_rank = r_new
            best_U = np.delete(U, remove_idx, axis=1)
            best_V = np.delete(V, remove_idx, axis=1)
            best_W = np.delete(W, remove_idx, axis=1)
            continue
        
        # Try to see if we can adjust the other terms to compensate
        # This requires solving a linear system, which is complex.
        # For now, skip this and try other removals.
    
    # Try a more systematic approach: use the null space of the full constraint system
    # to find a lower-rank solution.
    
    # The constraint is: for each (i, j, jp, k), sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c] = delta(j,jp)
    # This is 224 equations in the variables U, V, W (with rank constraints).
    
    # Let's try to find a rank-49 solution by checking if the 50 terms are linearly
    # independent as "effective" rank-1 tensors.
    
    # Two rank-1 tensors u_c ⊗ v_c ⊗ w_c and u_d ⊗ v_d ⊗ w_d are linearly dependent
    # if one is a scalar multiple of the other. Check if any pair is dependent.
    
    for c1 in range(rank):
        if time.time() > deadline - 5:
            break
        for c2 in range(c1 + 1, rank):
            # Check if (U[:,c1], V[:,c1], W[:,c1]) is a scalar multiple of (U[:,c2], V[:,c2], W[:,c2])
            # This requires U[:,c1] = alpha * U[:,c2], V[:,c1] = alpha * V[:,c2], W[:,c1] = alpha * W[:,c2]
            # for some scalar alpha.
            
            # Find alpha from the first nonzero element
            alpha = None
            for idx in range(nm):
                if U[idx, c2] != 0:
                    alpha = U[idx, c1] / U[idx, c2]
                    break
            if alpha is None:
                continue
            
            # Check if this alpha works for all elements
            if np.allclose(U[:, c1], alpha * U[:, c2]) and \
               np.allclose(V[:, c1], alpha * V[:, c2]) and \
               np.allclose(W[:, c1], alpha * W[:, c2]):
                # Terms c1 and c2 are linearly dependent
                # We can remove one of them
                best_rank = rank - 1
                # Remove c2
                keep = [c for c in range(rank) if c != c2]
                best_U = U[:, keep]
                best_V = V[:, keep]
                best_W = W[:, keep]
                break
        if best_rank < rank:
            break
    
    # Verify the best solution
    if best_rank < rank and verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If no improvement found, return the original rank-50 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
