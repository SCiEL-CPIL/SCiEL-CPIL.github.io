"""
Matrix multiplication 4x5x5 rank-minimal decomposition.
Uses structured search: start from schoolbook, find linear dependencies
among the 100 rank-1 terms to reduce rank.
"""
import time
import numpy as np
import itertools

DIMENSIONS = [4, 5, 5]
N, M, P = DIMENSIONS

def _schoolbook():
    """Return the schoolbook decomposition (rank = n*m*p = 100)."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    return max_err < tol, max_err

def _schoolbook_terms():
    """Return the 100 schoolbook terms as (u_vec, v_vec, w_vec) tuples.
    Term (i,j,k): u has 1 at i*m+j, v has 1 at j*p+k, w has 1 at k*n+i."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m + j] = 1.0
                v[j*p + k] = 1.0
                w[k*n + i] = 1.0
                terms.append((u, v, w))
    return terms

def _check_single_term_replacement(u_new, v_new, w_new, removed_terms, remaining_terms):
    """Check if replacing removed_terms with a single rank-1 term (u_new,v_new,w_new)
    still satisfies all (i,j,k) constraints.
    The sum over all terms must give 1 at each (i,j,k).
    Removed terms contribute sum of their (u*v*w) at each (i,j,k).
    New term contributes u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i].
    Remaining terms contribute their sum.
    Total must be 1.
    """
    n, m, p = DIMENSIONS
    # Compute the tensor from removed terms
    removed_tensor = np.zeros((n, m, p))
    for u, v, w in removed_terms:
        removed_tensor += np.einsum('a,b,c->abc', u, v, w)
    # Compute the tensor from remaining terms
    remaining_tensor = np.zeros((n, m, p))
    for u, v, w in remaining_terms:
        remaining_tensor += np.einsum('a,b,c->abc', u, v, w)
    # New term tensor
    new_tensor = np.einsum('a,b,c->abc', u_new, v_new, w_new)
    # Check: remaining_tensor + new_tensor == removed_tensor (since total must be 1)
    # Actually: remaining + new should equal 1 - remaining... no.
    # Original: sum of ALL terms = 1 at each (i,j,k)
    # After replacement: remaining + new should = 1 at each (i,j,k)
    # So remaining + new - 1 should be 0
    total = remaining_tensor + new_tensor
    max_err = np.max(np.abs(total - 1.0))
    return max_err < 1e-8, max_err

def _find_mergeable_pairs(terms, deadline):
    """Find pairs of schoolbook terms that can be merged into one rank-1 term.
    Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if their sum
    equals a single rank-1 tensor u*v*w.
    
    For terms to merge: the sum tensor must have rank 1 in the appropriate sense.
    Specifically, if we have terms t1=(u1,v1,w1) and t2=(u2,v2,w2),
    their sum is u1*v1*w1 + u2*v2*w2. This is rank-1 iff the "matrix" formed
    by combining them has rank 1 in a specific way.
    
    Actually, a simpler check: the sum of two rank-1 tensors is rank-1 iff
    they are "aligned" in a specific way. For matmul terms, two terms (i,j,k) and (i',j',k')
    can be merged if they share certain index structure.
    
    Let me think differently. The sum of two terms is:
    delta[i,j]delta[j,k] * delta[i',j']delta[j',k'] summed... no.
    
    Term (i,j,k) contributes 1 only at position (i,j,k) in the tensor.
    So the sum of terms (i1,j1,k1) and (i2,j2,k2) has 1 at (i1,j1,k1) and 1 at (i2,j2,k2),
    0 elsewhere. This is rank-1 iff... a rank-1 tensor u*v*w has the property that
    T[i,j,k] = u[i,j]*v[j,k]*w[k,i].
    
    For the sum to be rank-1, we need: there exist u,v,w such that
    u[i,j]*v[j,k]*w[k,i] = 1 if (i,j,k)==(i1,j1,k1) or (i2,j2,k2), 0 otherwise.
    
    This is only possible if the two positions are "compatible" in a strong way.
    Specifically, if (i1,j1,k1) != (i2,j2,k2), then we need u[i1,j1]*v[j1,k1]*w[k1,i1] = 1
    and u[i2,j2]*v[j2,k2]*w[k2,i2] = 1, and all other products = 0.
    
    For all other (i,j,k) not equal to either, we need u[i,j]*v[j,k]*w[k,i] = 0.
    This means for each (i,j,k) not in {(i1,j1,k1),(i2,j2,k2)}, at least one of
    u[i,j], v[j,k], w[k,i] must be 0.
    
    This is a very strong constraint. Let's check: can two distinct schoolbook terms ever merge?
    
    Consider terms (0,0,0) and (0,0,1). 
    Need: u[0,0]*v[0,0]*w[0,0] = 1, u[0,0]*v[0,1]*w[1,0] = 1
    And for all other (i,j,k): u[i,j]*v[j,k]*w[k,i] = 0.
    
    From first two: u[0,0] != 0, v[0,0] != 0, w[0,0] != 0, v[0,1] != 0, w[1,0] != 0.
    Now consider (1,0,0): u[1,0]*v[0,0]*w[0,1] = 0. Since v[0,0]!=0, need u[1,0]=0 or w[0,1]=0.
    Consider (0,1,0): u[0,1]*v[1,0]*w[0,0] = 0. Since w[0,0]!=0, need u[0,1]=0 or v[1,0]=0.
    Consider (0,0,2): u[0,0]*v[0,2]*w[2,0] = 0. Since u[0,0]!=0, need v[0,2]=0 or w[2,0]=0.
    Consider (1,0,1): u[1,0]*v[0,1]*w[1,1] = 0. Since v[0,1]!=0, need u[1,0]=0 or w[1,1]=0.
    
    This is getting complex. Let me just try to solve it as a system.
    """
    n, m, p = DIMENSIONS
    num_terms = len(terms)
    merges = []
    
    for idx1 in range(num_terms):
        if time.time() > deadline - 30:
            break
        for idx2 in range(idx1 + 1, num_terms):
            if time.time() > deadline - 30:
                break
            
            t1 = terms[idx1]
            t2 = terms[idx2]
            
            # Determine the (i,j,k) for each term
            # Term index: idx = i*m*p + j*p + k
            i1, j1, k1 = idx1 // (m*p), (idx1 // p) % m, idx1 % p
            i2, j2, k2 = idx2 // (m*p), (idx2 // p) % m, idx2 % p
            
            # For the sum to be rank-1, we need to find u,v,w such that
            # u[i,j]*v[j,k]*w[k,i] equals the sum tensor.
            # The sum tensor is 1 at (i1,j1,k1) and (i2,j2,k2), 0 elsewhere.
            
            # Key insight: if i1==i2 and j1==j2 but k1!=k2, or similar partial matches,
            # we might be able to merge. Let's try a direct approach.
            
            # Build the system: for each (i,j,k), u[i,j]*v[j,k]*w[k,i] = target[i,j,k]
            # where target is 1 at the two positions, 0 elsewhere.
            
            # This is a nonlinear system. Let's try a simpler heuristic:
            # Check if the two terms share at least one index component.
            
            # Actually, let me try a different approach: use the fact that
            # a rank-1 tensor T[i,j,k] = u[i,j]*v[j,k]*w[k,i] must satisfy
            # T[i,j,k]*T[i',j',k'] = T[i,j',k']*T[i',j,k'] when certain conditions hold.
            # Specifically, the "matrix" M_{(i,k),(j)} = T[i,j,k] must have rank 1
            # for fixed... this is getting complicated.
            
            # Let me just try: if the two terms differ in exactly one index,
            # check if merging is possible.
            
            diffs = sum([i1!=i2, j1!=j2, k1!=k2])
            if diffs != 1:
                continue
            
            # Try to solve for u,v,w
            # Case: same i,j, different k (k1 != k2)
            if i1 == i2 and j1 == j2:
                # u[i,j]*v[j,k1]*w[k1,i] = 1
                # u[i,j]*v[j,k2]*w[k2,i] = 1
                # All other (i',j',k'): u[i',j']*v[j',k']*w[k',i'] = 0
                
                # Set u[i,j] = 1 (scale freedom)
                # Then v[j,k1]*w[k1,i] = 1, v[j,k2]*w[k2,i] = 1
                # We can set v[j,k1]=1, w[k1,i]=1, v[j,k2]=1, w[k2,i]=1
                # But then for (i,j,k) with k != k1,k2: u[i,j]*v[j,k]*w[k,i] = v[j,k]*w[k,i] must be 0
                # So for all k not in {k1,k2}: v[j,k]=0 or w[k,i]=0
                # And for (i',j',k) with i'!=i or j'!=j: u[i',j']*v[j',k]*w[k,i'] = 0
                # So u[i',j']=0 for (i',j') != (i,j), OR v[j',k]=0 or w[k,i']=0 for those.
                
                # Simplest: set u[i',j']=0 for all (i',j') != (i,j)
                # Then for (i,j,k) with k not in {k1,k2}: need v[j,k]*w[k,i] = 0
                # Set v[j,k]=0 for k not in {k1,k2}
                # Then for (i',j',k) with j'!=j: u[i',j']=0 already handles it
                # For (i,j',k) with j'!=j: u[i,j']=0 handles it
                # For (i',j,k) with i'!=i: u[i',j]=0 handles it
                # For (i,j,k) with k in {k1,k2}: already set to 1
                
                # Check: does this work?
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i1*m + j1] = 1.0
                v[j1*p + k1] = 1.0
                w[k1*n + i1] = 1.0
                v[j1*p + k2] = 1.0
                w[k2*n + i1] = 1.0
                
                # Verify
                ok, err = _check_single_term_replacement(u, v, w, [t1, t2], [])
                # Need to check against ALL other terms
                other_terms = [terms[idx] for idx in range(num_terms) if idx != idx1 and idx != idx2]
                ok, err = _check_single_term_replacement(u, v, w, [t1, t2], other_terms)
                if ok:
                    merges.append((idx1, idx2, u, v, w))
            
            # Case: same i, different j,k
            elif i1 == i2 and j1 != j2 and k1 != k2:
                # u[i,j1]*v[j1,k1]*w[k1,i] = 1
                # u[i,j2]*v[j2,k2]*w[k2,i] = 1
                # All others = 0
                # Set u[i,j1]=1, v[j1,k1]=1, w[k1,i]=1
                # Set u[i,j2]=1, v[j2,k2]=1, w[k2,i]=1
                # But w[k1,i] and w[k2,i] might conflict if k1!=k2 (they're different entries, OK)
                # Check: for (i,j1,k2): u[i,j1]*v[j1,k2]*w[k2,i] = 1*v[j1,k2]*1 = v[j1,k2] must be 0
                # For (i,j2,k1): u[i,j2]*v[j2,k1]*w[k1,i] = 1*v[j2,k1]*1 = v[j2,k1] must be 0
                # For (i',j1,k1) with i'!=i: u[i',j1]*v[j1,k1]*w[k1,i'] = u[i',j1]*1*w[k1,i'] must be 0
                # Set u[i',j1]=0 for i'!=i, and w[k1,i']=0 for i'!=i (but w[k1,i]=1 already)
                # This is getting complex. Let me just try it.
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i1*m + j1] = 1.0
                v[j1*p + k1] = 1.0
                w[k1*n + i1] = 1.0
                u[i1*m + j2] = 1.0
                v[j2*p + k2] = 1.0
                w[k2*n + i1] = 1.0
                
                other_terms = [terms[idx] for idx in range(num_terms) if idx != idx1 and idx != idx2]
                ok, err = _check_single_term_replacement(u, v, w, [t1, t2], other_terms)
                if ok:
                    merges.append((idx1, idx2, u, v, w))
            
            # Case: same j, different i,k
            elif j1 == j2 and i1 != i2 and k1 != k2:
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i1*m + j1] = 1.0
                v[j1*p + k1] = 1.0
                w[k1*n + i1] = 1.0
                u[i2*m + j1] = 1.0
                v[j1*p + k2] = 1.0
                w[k2*n + i2] = 1.0
                
                other_terms = [terms[idx] for idx in range(num_terms) if idx != idx1 and idx != idx2]
                ok, err = _check_single_term_replacement(u, v, w, [t1, t2], other_terms)
                if ok:
                    merges.append((idx1, idx2, u, v, w))
            
            # Case: same k, different i,j
            elif k1 == k2 and i1 != i2 and j1 != j2:
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i1*m + j1] = 1.0
                v[j1*p + k1] = 1.0
                w[k1*n + i1] = 1.0
                u[i2*m + j2] = 1.0
                v[j2*p + k1] = 1.0
                w[k1*n + i2] = 1.0
                
                other_terms = [terms[idx] for idx in range(num_terms) if idx != idx1 and idx != idx2]
                ok, err = _check_single_term_replacement(u, v, w, [t1, t2], other_terms)
                if ok:
                    merges.append((idx1, idx2, u, v, w))
    
    return merges

def _apply_merges(terms, merges):
    """Apply a set of merges to the terms. Returns new list of terms."""
    merged_indices = set()
    new_terms = []
    for idx1, idx2, u, v, w in merges:
        if idx1 not in merged_indices and idx2 not in merged_indices:
            merged_indices.add(idx1)
            merged_indices.add(idx2)
            new_terms.append((u.copy(), v.copy(), w.copy()))
    for idx, term in enumerate(terms):
        if idx not in merged_indices:
            new_terms.append(term)
    return new_terms

def _find_more_merges(terms, deadline):
    """Given a set of terms (not necessarily schoolbook), find pairs that can be merged.
    This is harder because the terms are now arbitrary rank-1 tensors.
    We check if the sum of two terms is rank-1.
    """
    n, m, p = DIMENSIONS
    num_terms = len(terms)
    merges = []
    
    for idx1 in range(num_terms):
        if time.time() > deadline - 20:
            break
        for idx2 in range(idx1 + 1, num_terms):
            if time.time() > deadline - 20:
                break
            
            t1 = terms[idx1]
            t2 = terms[idx2]
            
            # Check if t1 + t2 is rank-1
            # Sum tensor: S[i,j,k] = t1.u[i,j]*t1.v[j,k]*t1.w[k,i] + t2.u[i,j]*t2.v[j,k]*t2.w[k,i]
            # For S to be rank-1: S = u*v*w for some u,v,w
            # Necessary condition: the "matricization" must have rank 1
            # S as a matrix with rows (i,j) and cols k: S[i,j,k]
            # Or rows i, cols (j,k), etc.
            
            # Build the full tensor
            S = np.einsum('a,b,c->abc', t1[0], t1[1], t1[2]) + np.einsum('a,b,c->abc', t2[0], t2[1], t2[2])
            
            # Check rank via SVD of matricizations
            # Matricization 1: rows=(i,j), cols=k
            M1 = S.reshape(n*m, p)
            # Matricization 2: rows=i, cols=(j,k)
            M2 = S.reshape(n, m*p)
            # Matricization 3: rows=(i,k), cols=j
            M3 = S.transpose(0,2,1).reshape(n*p, m)
            
            # Rank-1 tensor: all matricizations have rank 1
            # (approximately)
            try:
                r1 = np.linalg.matrix_rank(M1, tol=1e-8)
                r2 = np.linalg.matrix_rank(M2, tol=1e-8)
                r3 = np.linalg.matrix_rank(M3, tol=1e-8)
            except:
                continue
            
            if r1 > 1 or r2 > 1 or r3 > 1:
                continue
            
            # The sum is rank-1. Extract u,v,w from S.
            # S[i,j,k] = u[i,j]*v[j,k]*w[k,i]
            # Find a non-zero entry
            nonzero = np.where(S != 0)
            if len(nonzero[0]) == 0:
                continue
            
            # Use the first nonzero entry to set scale
            # This is tricky. Let's use a simple approach:
            # If S is rank-1, then S = outer product in some sense.
            # Actually, for a rank-1 tensor in this format, we can extract:
            # Pick i0,j0,k0 with S[i0,j0,k0] != 0
            # Then u[i,j] = S[i,j,k0] / (v[j,k0]*w[k0,i])... circular.
            
            # Alternative: since we just need SOME u,v,w that work, and the sum is rank-1,
            # we can use the fact that for rank-1 tensors, the entries factorize.
            # Let's try: u[i,j] = S[i,j,0] if S[i,j,0] != 0, else use another k.
            
            # Simpler: just use the SVD approach to get approximate u,v,w
            # For now, mark this pair as mergeable and extract later
            
            # Extract u,v,w from S
            # S[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # Take logs (where non-zero) to linearize
            # This is complex. Let's try a heuristic:
            
            # Find all non-zero entries
            nz = np.where(S != 0)
            if len(nz[0]) == 0:
                continue
            
            # Try to find u,v,w by solving
            # For a rank-1 tensor, S[i,j,k] * S[i',j',k'] = S[i,j',k'] * S[i',j,k']
            # (when the indices are compatible)
            
            # Let's just try: set u[i,j] = S[i,j,k*] for some fixed k*,
            # v[j,k] = S[i*,j,k] / u[i*,j] for some fixed i*,
            # w[k,i] = S[i,j*,k] / (u[i,j*] * v[j*,k]) for some fixed j*
            
            # Find good i*, j*, k*
            i_star, j_star, k_star = nz[0][0], nz[1][0], nz[2][0]
            
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Set u[i,j] = S[i,j,k_star] / (v[j,k_star] * w[k_star,i])
            # This is circular. Let's use a different approach.
            
            # For rank-1 tensor S, we can write S = alpha * outer(a, b, c) in some sense.
            # Actually, let's just verify the merge works by checking all (i,j,k) constraints
            # with the new term replacing the two old ones.
            
            # To extract u,v,w: use the fact that if S is rank-1, then
            # S[i,j,k] = S[i,j,k0] * S[i0,j,k] * S[i0,j0,k] / (S[i0,j0,k0]^2) ... not quite.
            
            # Let me try: u[i,j] = S[i,j,k_star] / S[i_star,j_star,k_star] * S[i_star,j_star,k_star]
            # No, let's just do it properly.
            
            # For a rank-1 tensor T[i,j,k] = u[i,j]*v[j,k]*w[k,i]:
            # T[i,j,k] * T[i',j',k'] = T[i,j',k'] * T[i',j,k'] (when indices align properly)
            # Specifically: T[i,j,k] * T[i,j',k'] = T[i,j',k] * T[i,j,k'] (same i)
            # And: T[i,j,k] * T[i',j,k'] = T[i',j,k] * T[i,j,k'] (same j)
            
            # Let's just try to solve it numerically.
            # Pick a reference point (i0,j0,k0) with T[i0,j0,k0] != 0
            # Set u[i0,j0] = 1
            # Then v[j0,k0] * w[k0,i0] = T[i0,j0,k0]
            # For same i0: T[i0,j,k] = u[i0,j] * v[j,k] * w[k,i0]
            #   => u[i0,j] = T[i0,j,k] / (v[j,k] * w[k,i0])
            # Still circular.
            
            # OK, let's use a completely different approach:
            # Since S is rank-1, we can get u,v,w from the SVD of matricizations.
            # M1 = S.reshape(n*m, p). If rank 1, M1 = a * b^T where a is (n*m)x1, b is px1.
            # Then S[i,j,k] = a[i*m+j] * b[k]
            # But we need S[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # So a[i*m+j] * b[k] = u[i,j] * v[j,k] * w[k,i]
            # This means a[i*m+j] = u[i,j] * c_j and b[k] = d_k * (1/w[k,i])... not clean.
            
            # For now, let's just store the merge and extract u,v,w later using optimization.
            # Or, more practically: if the sum is rank-1, we can just use the SVD vectors.
            
            # Get left and right singular vectors of M1
            try:
                U_svd, s, Vt_svd = np.linalg.svd(M1, full_matrices=False)
                if s[0] < 1e-10:
                    continue
                a = U_svd[:, 0] * np.sqrt(s[0])
                b = Vt_svd[0, :] * np.sqrt(s[0])
            except:
                continue
            
            # Now S[i,j,k] ≈ a[i*m+j] * b[k]
            # We need to factor a[i*m+j] * b[k] = u[i,j] * v[j,k] * w[k,i]
            # This is possible if the factorization is consistent.
            # Set u[i,j] = a[i*m+j] / f[j] for some f, v[j,k] = g[j] * h[k], w[k,i] = b[k] / (h[k] * f[j] * ...)
            # This is underdetermined. Let's just try:
            # u[i,j] = a[i*m+j], v[j,k] = 1, w[k,i] = b[k]
            # Check: u[i,j]*v[j,k]*w[k,i] = a[i*m+j] * 1 * b[k] = S[i,j,k] ✓
            
            u = a.reshape(n, m)
            v = np.ones((m, p))
            w = b.reshape(p, 1)  # w[k,i] = b[k] for all i
            
            # Convert to flat vectors
            u_flat = u.reshape(-1)
            v_flat = v.reshape(-1)
            w_flat = np.tile(b, n)  # w[k*n+i] = b[k] for all i
            
            # Verify
            other_terms = [terms[idx] for idx in range(num_terms) if idx != idx1 and idx != idx2]
            ok, err = _check_single_term_replacement(u_flat, v_flat, w_flat, [t1, t2], other_terms)
            if ok:
                merges.append((idx1, idx2, u_flat, v_flat, w_flat))
    
    return merges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    terms = _schoolbook_terms()
    current_rank = len(terms)
    best_rank = current_rank
    best_terms = terms[:]
    
    # Iteratively find and apply merges
    iterations = 0
    while time.time() < deadline - 30 and iterations < 50:
        iterations += 1
        merges = _find_more_merges(terms, deadline)
        if not merges:
            break
        
        # Apply all non-conflicting merges
        merged_indices = set()
        new_terms = []
        applied = 0
        for idx1, idx2, u, v, w in merges:
            if idx1 not in merged_indices and idx2 not in merged_indices:
                merged_indices.add(idx1)
                merged_indices.add(idx2)
                new_terms.append((u.copy(), v.copy(), w.copy()))
                applied += 1
        
        for idx, term in enumerate(terms):
            if idx not in merged_indices:
                new_terms.append(term)
        
        if applied == 0:
            break
        
        terms = new_terms
        current_rank = len(terms)
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_terms = terms[:]
    
    # Build the final U, V, W matrices
    n, m, p = DIMENSIONS
    rank = len(best_terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u, v, w) in enumerate(best_terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
    
    # Verify
    ok, err = _verify(U.tolist(), V.tolist(), W.tolist())
    if not ok:
        # Fall back to schoolbook
        result = _schoolbook()
        return result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

