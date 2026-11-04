import time
import random
import numpy as np
import sys

DIMENSIONS = [3, 4, 8]

def verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index is k*n + i where k is col (0..p-1), i is row (0..n-1)
    U index is i*m + j where i is row (0..n-1), j is col (0..m-1)
    V index is j*p + k where j is row (0..m-1), k is col (0..p-1)
    
    T[i,j,k] = C[i,k] from A[i,:] * B[:,k] = sum over r of A[i,r]*B[r,k]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            target = 0.0
            # T[i,j,k] contributes: sum_j A[i,j]*B[j,k]
            # In tensor form: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For fixed (i,k): sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            total = 0.0
            for j in range(m):
                u_col = U[i*m+j]
                v_row = V[j*p+k]
                # sum_c u_col[c] * v_row[c]
                total += np.dot(u_col, v_row)
            expected = total * W[k*n+i, :]  # wait no
            # Actually: for fixed (i,j,k), the contribution to T is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # But T[i,j,k] = 1 for all i,j,k (this is the matmul tensor)
            # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
            # Let me re-verify by checking all triples
            
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                s = 0.0
                for c in range(rank):
                    s += u_val[c] * v_val[c] * w_val[c]
                if abs(s - 1.0) > 1e-9:
                    return False, f"Triple ({i},{j},{k}) gives {s}, expected 1"
    return True, "ok"

def schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140  # safety margin
    
    # Start with schoolbook
    U, V, W = schoolbook_decomp(n, m, p)
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = len(U[0])
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    def check_time():
        return time.time() < deadline
    
    # Strategy: try to merge pairs of rank-1 terms
    # Two terms (c1, c2) can potentially be merged if their tensor structure
    # allows a rank-2 -> rank-1 reduction on a sub-tensor
    # 
    # A pair of terms contributes: for all (i,j,k):
    #   U1[i*m+j]*V1[j*p+k]*W1[k*n+i] + U2[i*m+j]*V2[j*p+k]*W2[k*n+i]
    # We want this to equal U3[i*m+j]*V3[j*p+k]*W3[k*n+i]
    #
    # This is possible when the 2D sub-tensor (for fixed free indices)
    # has rank 1, which happens when the two rank-1 components are
    # aligned in a specific way.
    #
    # Key insight: If two terms differ only in one index set, they can be merged.
    # E.g., if U1 and U2 differ, but V1=V2 and W1=W2 (up to scaling), then
    # sum = (U1+U2)*V*W, which is rank 1.
    # More generally, if the outer product structures align on 2 of 3 modes.
    
    # Let's represent each term by (u_vec, v_vec, w_vec) where u_vec has length n*m,
    # v_vec has length m*p, w_vec has length p*n.
    
    def terms_from_matrices(U, V, W):
        rank = U[0].__len__() if U else 0
        terms = []
        for c in range(rank):
            u = [U[i][c] for i in range(n_m)]
            v = [V[i][c] for i in range(m_p)]
            w = [W[i][c] for i in range(p_n)]
            terms.append((u, v, w))
        return terms
    
    def matrices_from_terms(terms):
        rank = len(terms)
        U = [[0]*rank for _ in range(n_m)]
        V = [[0]*rank for _ in range(m_p)]
        W = [[0]*rank for _ in range(p_n)]
        for c, (u, v, w) in enumerate(terms):
            for i in range(n_m):
                U[i][c] = u[i]
            for i in range(m_p):
                V[i][c] = v[i]
            for i in range(p_n):
                W[i][c] = w[i]
        return U, V, W
    
    def is_sparse(vec):
        nonzeros = sum(1 for x in vec if x != 0)
        return nonzeros, nonzeros / len(vec)
    
    def try_merge(terms):
        """Try to find a pair of terms that can be merged into one.
        Returns (new_terms, success) where new_terms has one fewer element."""
        rank = len(terms)
        if rank < 2:
            return terms, False
        
        # For each pair, check if they can be merged
        # Two terms t1=(u1,v1,w1) and t2=(u2,v2,w2) can be merged if:
        # there exists (u3,v3,w3) such that for all (i,j,k):
        # u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k] = u3[i]*v3[j]*w3[k]
        # 
        # This is a rank-2 tensor that must have rank 1.
        # The 3D tensor T[i,j,k] = u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k]
        # has rank 1 iff T = u3 ⊗ v3 ⊗ w3.
        #
        # Check: For all i,j,k: T[i,j,k] = u3[i]*v3[j]*w3[k]
        # Taking ratios: T[i,j,k]*T[i',j',k'] = T[i,j',k]*T[i',j,k'] etc.
        # Simpler: if we fix one index, the resulting 2D matrix must be rank 1.
        # T[:,j0,k0] must be parallel across all (j0,k0) pairs (i.e., all 2D slices have rank 1 and consistent).
        
        # Efficient check: 
        # If u1 and u2 are nonzero: the i-direction structure must be consistent.
        # T[i,j,k] = u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k]
        # = u1[i] * (v1[j]*w1[k]) + u2[i] * (v2[j]*w2[k])
        # For this to factor as u3[i]*v3[j]*w3[k]:
        # The two vectors (v1[j]*w1[k]) and (v2[j]*w2[k]) must be parallel, OR
        # u1 and u2 must be parallel (in which case factor out), OR
        # similar conditions in other modes.
        
        # Let's check: are (v1*w1) and (v2*w2) parallel? (element-wise products)
        v1w1 = [v1[j] * w1[k] for j in range(m) for k in range(p)]  # wait, v is indexed by j*p+k
        # v has length m*p, w has length p*n
        # v[j*p+k], w[k*n+i]
        # The "2D matrix" for mode i is: M_i = v1[j]*w1[k] (but this isn't quite right)
        
        # Let me think differently. For the 3-tensor T[a,b,c] where a indexes U-space (n*m),
        # b indexes V-space (m*p), c indexes W-space (p*n):
        # T[a,b,c] = u[a]*v[b]*w[c] for each term.
        # Sum of two: T[a,b,c] = u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c]
        # For this to be rank 1: T[a,b,c] = u3[a]*v3[b]*w3[c]
        
        # Check method: If there exist (a0,b0), (a1,b1) where T is nonzero at both,
        # then u3[a0]*v3[b0]*w3[c0] = u3[a1]*v3[b1]*w3[c0] implies
        # u3[a0]/u3[a1] = v3[b1]/v3[b0] for all c0 where w3[c0]!=0.
        # So the ratios u3[a0]/u3[a1] must be constant over b.
        
        # Simpler numerical check: build a small sample and check rank.
        # For speed, let's check a few (i,j,k) combinations.
        
        # Actually, let me use a cleaner algebraic approach:
        # Case 1: u1 and u2 are proportional. Then u3 = u1+u2 (scaled), and we need
        #         v1*w1 and v2*w2 to be proportional.
        # Case 2: v1 and v2 are proportional. Then similar in other modes.
        # Case 3: w1 and w2 are proportional.
        # Case 4: More subtle cases.
        
        # Let me check if u1 ∝ u2:
        def vectors_proportional(v1, v2):
            # Check if v1 = alpha * v2 for some alpha
            n1 = next((x for x in v1 if x != 0), None)
            n2 = next((x for x in v2 if x != 0), None)
            if n1 is None and n2 is None:
                return True, 0.0  # both zero
            if n1 is None or n2 is None:
                return False, 0.0
            alpha = n1 / n2
            for a, b in zip(v1, v2):
                if abs(a - alpha * b) > 1e-12:
                    return False, 0.0
            return True, alpha
        
        # Check mode by mode
        # Mode U: u1, u2
        prop_u, alpha_u = vectors_proportional(u1 := terms[idx1][0], u2 := terms[idx2][0])
        prop_v, alpha_v = vectors_proportional(v1 := terms[idx1][1], v2 := terms[idx2][1])
        prop_w, alpha_w = vectors_proportional(w1 := terms[idx1][2], w2 := terms[idx2][2])
        
        # If two modes are proportional, merge is straightforward
        if prop_u and prop_v:
            # u3 = u1+u2, v3 = v1+alpha_u*v2 (wait, let me redo)
            # T[a,b,c] = u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c]
            # If u2 = alpha_u * u1, v2 = alpha_v * v1:
            # T = u1[a]*v1[b]*w1[c] + alpha_u*u1[a]*alpha_v*v1[b]*w2[c]
            # = u1[a]*v1[b]*(w1[c] + alpha_u*alpha_v*w2[c])
            # So u3=u1, v3=v1, w3 = w1 + alpha_u*alpha_v*w2
            # But we need this to equal u3[a]*v3[b]*w3[c], which it does!
            # Wait, but we also need w1 ∝ w2 or the last term must factor.
            # T = u1[a] * v1[b] * (w1[c] + alpha_u*alpha_v*w2[c])
            # This IS rank 1! Because it's u1 ⊗ v1 ⊗ (w1 + alpha_u*alpha_v*w2).
            # So if two modes are proportional, merge works.
            
            u3 = [u1[a] for a in range(n_m)]
            v3 = [v1[b] for b in range(m_p)]
            factor = alpha_u * alpha_v
            w3 = [w1[c] + factor * w2[c] for c in range(p_n)]
            
            new_terms = terms[:idx1] + terms[idx1+1:idx2] + terms[idx2+1:] + [(u3, v3, w3)]
            # Reorder: keep other terms + new merged term
            # Actually let me be more careful
            keep = []
            for idx in range(rank):
                if idx != idx1 and idx != idx2:
                    keep.append(terms[idx])
            keep.append((u3, v3, w3))
            return keep, True
        
        elif prop_u and prop_w:
            # u2 = alpha_u*u1, w2 = alpha_w*w1
            # T = u1[a]*v1[b]*w1[c] + alpha_u*u1[a]*v2[b]*alpha_w*w1[c]
            # = u1[a]*w1[c]*(v1[b] + alpha_u*alpha_w*v2[b])
            # Rank 1!
            u3 = [u1[a] for a in range(n_m)]
            factor = alpha_u * alpha_w
            v3 = [v1[b] + factor * v2[b] for b in range(m_p)]
            w3 = [w1[c] for c in range(p_n)]
            keep = [terms[idx] for idx in range(rank) if idx != idx1 and idx != idx2]
            keep.append((u3, v3, w3))
            return keep, True
        
        elif prop_v and prop_w:
            # v2 = alpha_v*v1, w2 = alpha_w*w1
            # T = u1[a]*v1[b]*w1[c] + u2[a]*alpha_v*v1[b]*alpha_w*w1[c]
            # = v1[b]*w1[c]*(u1[a] + alpha_v*alpha_w*u2[a])
            factor = alpha_v * alpha_w
            u3 = [u1[a] + factor * u2[a] for a in range(n_m)]
            v3 = [v1[b] for b in range(m_p)]
            w3 = [w1[c] for c in range(p_n)]
            keep = [terms[idx] for idx in range(rank) if idx != idx1 and idx != idx2]
            keep.append((u3, v3, w3))
            return keep, True
        
        elif prop_u:
            # u2 = alpha_u * u1
            # T = u1[a] * (v1[b]*w1[c] + alpha_u * v2[b]*w2[c])
            # Need v1*w1 + alpha_u*v2*w2 to factor as v3*w3
            # This requires the 2-tensor v1⊗w1 + alpha_u*v2⊗w2 to have rank 1
            # which happens if v1 ∝ v2 (handled above) or w1 ∝ w2 (handled above)
            # or in general when the 2-term sum of 2-tensors has rank 1
            # v1[b]*w1[c] + alpha_u*v2[b]*w2[c] = v3[b]*w3[c]
            # This is: for all b,c: v1[b]*w1[c] + alpha_u*v2[b]*w2[c] = v3[b]*w3[c]
            # If v1 and v2 are not proportional and w1 and w2 are not proportional,
            # the 2-tensor has rank 2 in general.
            # But special cases: if one of v1,w1,v2,w2 is zero vector.
            pass
        
        return terms, False
    
    # The above function references terms[idx1] etc. incorrectly - let me rewrite properly.
    
    def try_merge_pair(terms, i1, i2):
        """Try to merge terms[i1] and terms[i2]. Returns (new_terms_list, success)."""
        u1, v1, w1 = terms[i1]
        u2, v2, w2 = terms[i2]
        
        def vectors_proportional(x, y):
            nx = next((a for a in x if a != 0), None)
            ny = next((a for a in y if a != 0), None)
            if nx is None and ny is None:
                return True, 0.0
            if nx is None or ny is None:
                return False, 0.0
            alpha = nx / ny
            for a, b in zip(x, y):
                if abs(a - alpha * b) > 1e-12:
                    return False, 0.0
            return True, alpha
        
        prop_u, au = vectors_proportional(u1, u2)
        prop_v, av = vectors_proportional(v1, v2)
        prop_w, aw = vectors_proportional(w1, w2)
        
        count = sum([prop_u, prop_v, prop_w])
        
        if count >= 2:
            # At least two modes proportional -> merge works
            # Use the two proportional modes to determine which to sum
            if prop_u and prop_v:
                u3 = u1[:]
                v3 = v1[:]
                f = au * av
                w3 = [w1[c] + f * w2[c] for c in range(p_n)]
            elif prop_u and prop_w:
                u3 = u1[:]
                f = au * aw
                v3 = [v1[b] + f * v2[b] for b in range(m_p)]
                w3 = w1[:]
            else:  # prop_v and prop_w
                f = av * aw
                u3 = [u1[a] + f * u2[a] for a in range(n_m)]
                v3 = v1[:]
                w3 = w1[:]
            
            keep = [terms[idx] for idx in range(len(terms)) if idx != i1 and idx != i2]
            keep.append((u3, v3, w3))
            return keep, True
        
        # Single mode proportional: check if the 2-tensor sum has rank 1
        if count == 1:
            if prop_u:
                # u2 = au * u1
                # Need v1*w1 + au*v2*w2 to be rank 1
                # i.e., exists v3,w3 such that v1[b]*w1[c] + au*v2[b]*w2[c] = v3[b]*w3[c]
                # Check: pick b0 where (v1[b0], v2[b0]) != (0,0), then for all b:
                # v1[b]*w1 + au*v2[b]*w2 = lambda_b * (v1[b0]*w1 + au*v2[b0]*w2)
                # where lambda_b = v3[b]/v3[b0]
                # This means the vectors {v1[b]*w1 + au*v2[b]*w2 for all b} are all parallel.
                # Check by picking 3 b's and verifying proportionality.
                base = None
                for b in range(m_p):
                    vec = [v1[b]*w1[c] + au*v2[b]*w2[c] for c in range(p_n)]
                    nn = next((x for x in vec if x != 0), None)
                    if nn is None:
                        continue
                    if base is None:
                        base = vec
                        base_nz = next((x for x in vec if x != 0), None)
                    else:
                        alpha = nn / base_nz
                        ok = all(abs(vec[c] - alpha * base[c]) < 1e-12 for c in range(p_n) if base[c] != 0 or abs(vec[c]) > 1e-12)
                        if not ok:
                            return terms, False
                # If we got here, the 2-tensor has rank 1
                u3 = u1[:]
                # v3 and w3: use base vector = v3[b0]*w3 for some normalization
                # Actually, let's just set v3[b] = coefficient and w3 = base pattern
                # v1[b]*w1 + au*v2[b]*w2 = v3[b] * w3
                # So w3 = base (up to scale), v3[b] = ratio
                w3 = base[:]
                w3_norm = next((x for x in w3 if x != 0), 1.0)
                v3 = []
                for b in range(m_p):
                    vec = [v1[b]*w1[c] + au*v2[b]*w2[c] for c in range(p_n)]
                    nn = next((x for x in vec if x != 0), 0)
                    v3.append(nn / w3_norm)
                keep = [terms[idx] for idx in range(len(terms)) if idx != i1 and idx != i2]
                keep.append((u3, v3, w3))
                return keep, True
            
            elif prop_v:
                # v2 = av * v1
                # Need u1*w1 + av*u2*w2 to be rank 1 (2-tensor in modes U and W)
                base = None
                base_nz = 1.0
                for a in range(n_m):
                    vec = [u1[a]*w1[c] + av*u2[a]*w2[c] for c in range(p_n)]
                    nn = next((x for x in vec if x != 0), None)
                    if nn is None:
                        continue
                    if base is None:
                        base = vec
                        base_nz = nn
                    else:
                        alpha = nn / base_nz
                        ok = all(abs(vec[c] - alpha * base[c]) < 1e-12 for c in range(p_n) if abs(base[c]) > 1e-12 or abs(vec[c]) > 1e-12)
                        if not ok:
                            return terms, False
                w3 = base[:]
                u3 = []
                for a in range(n_m):
                    vec = [u1[a]*w1[c] + av*u2[a]*w2[c] for c in range(p_n)]
                    nn = next((x for x in vec if x != 0), 0)
                    u3.append(nn / base_nz)
                v3 = v1[:]
                keep = [terms[idx] for idx in range(len(terms)) if idx != i1 and idx != i2]
                keep.append((u3, v3, w3))
                return keep, True
            
            elif prop_w:
                # w2 = aw * w1
                # Need u1*v1 + aw*u2*v2 to be rank 1 (2-tensor in modes U and V)
                # u1[a]*v1[b] + aw*u2[a]*v2[b] = u3[a]*v3[b]
                base = None
                base_nz = 1.0
                for a in range(n_m):
                    vec = [u1[a]*v1[b] + aw*u2[a]*v2[b] for b in range(m_p)]
                    nn = next((x for x in vec if x != 0), None)
                    if nn is None:
                        continue
                    if base is None:
                        base = vec
                        base_nz = nn
                    else:
                        alpha = nn / base_nz
                        ok = all(abs(vec[b] - alpha * base[b]) < 1e-12 for b in range(m_p) if abs(base[b]) > 1e-12 or abs(vec[b]) > 1e-12)
                        if not ok:
                            return terms, False
                v3 = base[:]
                u3 = []
                for a in range(n_m):
                    vec = [u1[a]*v1[b] + aw*u2[a]*v2[b] for b in range(m_p)]
                    nn = next((x for x in vec if x != 0), 0)
                    u3.append(nn / base_nz)
                w3 = w1[:]
                keep = [terms[idx] for idx in range(len(terms)) if idx != i1 and idx != i2]
                keep.append((u3, v3, w3))
                return keep, True
        
        return terms, False
    
    # Main search loop
    random.seed(42)
    max_iterations = 5000
    
    iter_count = 0
    while check_time():
        iter_count += 1
        terms = terms_from_matrices(
            [[U[i][c] for c in range(best_rank)] for i in range(n_m)] if False else
            [ [best_U[i][c] for c in range(best_rank)] for i in range(n_m)],
            [ [best_V[i][c] for c in range(best_rank)] for i in range(m_p)],
            [ [best_W[i][c] for c in range(best_rank)] for i in range(p_n)]
        )
        
        # Actually, let me work with a local copy
        U_local = [row[:] for row in best_U]
        V_local = [row[:] for row in best_V]
        W_local = [row[:] for row in best_W]
        rank_local = len(U_local[0])
        
        terms_local = []
        for c in range(rank_local):
            u = [U_local[i][c] for i in range(n_m)]
            v = [V_local[i][c] for i in range(m_p)]
            w = [W_local[i][c] for i in range(p_n)]
            terms_local.append((u, v, w))
        
        merged = False
        # Try random pairs
        attempts = 0
        max_attempts = 200
        while not merged and attempts < max_attempts and check_time():
            attempts += 1
            i1 = random.randint(0, len(terms_local) - 1)
            i2 = random.randint(0, len(terms_local) - 1)
            if i1 == i2:
                continue
            new_terms, success = try_merge_pair(terms_local, i1, i2)
            if success and len(new_terms) < len(terms_local):
                merged = True
                terms_local = new_terms
        
        if merged:
            # Update best
            U_new, V_new, W_new = matrices_from_terms(terms_local)
            # Verify
            valid, _ = verify(U_new, V_new, W_new, n, m, p)
            if valid:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = len(U_new[0])
                print(f"Rank reduced to {best_rank}", file=sys.stderr)
            else:
                # Verification failed, skip
                pass
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}


def construct():
    n, m, p = DIMENSIONS
    
    # The previous search strategy (greedy merge of proportional modes) is ineffective
    # for the schoolbook decomposition because no two distinct terms share proportional
    # modes. The search would run until timeout without finding any merges.
    # 
    # To ensure a valid result within the time limit, we return the schoolbook
    # decomposition, which is always exact and has rank n*m*p.
    # For 3x4x8, rank is 96.
    
    U, V, W = schoolbook_decomp(n, m, p)
    
    return {"U": U, "V": V, "W": W}
