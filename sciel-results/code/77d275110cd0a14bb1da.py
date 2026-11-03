import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - start_time < DEADLINE

    def build_schoolbook():
        """Return list of (u_dict, v_dict, w_dict) for schoolbook rank-72."""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, coeff in u_dict.items():
                U[idx, r] = coeff
            for idx, coeff in v_dict.items():
                V[idx, r] = coeff
            for idx, coeff in w_dict.items():
                W[idx, r] = coeff
        return U, V, W

    def verify(U, V, W):
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # Check that each (i,j,k) entry equals 1
        return np.allclose(result, 1.0, atol=1e-9)

    def try_merge(terms, a, b):
        """
        Try to merge terms[a] and terms[b] into a single rank-2 term.
        Returns (new_term, remaining_terms) if successful, else None.
        
        Strategy: If U_a and U_b differ in exactly one coordinate (one has value at idx_x, other doesn't),
        or V_a and V_b differ in exactly one coordinate, we can use a rank-2 construction.
        """
        ua, va, wa = terms[a]
        ub, vb, wb = terms[b]
        
        # Get sets of nonzero indices
        ua_keys = set(ua.keys())
        ub_keys = set(ub.keys())
        va_keys = set(va.keys())
        vb_keys = set(vb.keys())
        
        # Check if U factors differ in exactly 1 element
        u_diff_a = ua_keys - ub_keys  # in ua but not ub
        u_diff_b = ub_keys - ua_keys  # in ub but not ua
        u_common = ua_keys & ub_keys
        
        # Check if V factors differ in exactly 1 element
        v_diff_a = va_keys - vb_keys
        v_diff_b = vb_keys - va_keys
        v_common = va_keys & vb_keys
        
        # Case 1: U differs in exactly 1 element on each side, V is identical
        if len(u_diff_a) == 1 and len(u_diff_b) == 1 and va_keys == vb_keys:
            # ua = common + {x: ca}, ub = common + {y: cb}
            x = list(u_diff_a)[0]
            y = list(u_diff_b)[0]
            ca = ua[x]
            cb = ub[y]
            
            # V is same: v_val
            # wa and wb are single-entry dicts
            wa_keys = set(wa.keys())
            wb_keys = set(wb.keys())
            
            # For the merge to work, we need:
            # (common + x*ca) * v * w_new = common*v*wa + x*ca*v*wa + ...
            # This gets complex. Let's try a specific construction.
            # 
            # Actually, let's think differently. We want:
            # term_a + term_b = new_term
            # ua*va*wa + ub*vb*wb
            #
            # If va == vb = v, then: (ua+ub)*v*(?) 
            # But ua+ub has 2 extra elements, so we'd need rank 2 in W.
            # 
            # Alternative: If ua and ub share all but 1 element, and va==vb:
            # Let ua = uc + x*ca, ub = uc + y*cb where uc is common part
            # Then ua*va*wa + ub*vb*wb = uc*v*(wa+wb) + x*ca*v*wa + y*cb*v*wb
            # We can write this as:
            # uc*v*(wa+wb) + v*(x*ca*wa + y*cb*wb)
            # But that's still 2 terms unless wa+wb and the second part are related.
            #
            # This approach is getting complicated. Let me try a different structural approach.
            pass
        
        # Case 2: V differs in exactly 1 element on each side, U is identical
        if len(v_diff_a) == 1 and len(v_diff_b) == 1 and ua_keys == ub_keys:
            pass
        
        # Case 3: Both U and V differ - more complex
        # For now, return None to indicate no merge found
        return None

    # Better approach: Use a linear algebra method
    # Represent each rank-1 term as a vector in a high-dimensional space
    # and look for linear dependencies that allow rank reduction
    
    def rank_reduction_search(terms, max_iter=5000):
        """
        Systematically try to reduce rank by finding pairs that can be merged.
        Uses the observation that if we have terms where U vectors are "close" 
        (differ by a rank-1 update), we can potentially combine them.
        """
        current_terms = list(terms)
        rank = len(current_terms)
        
        for iteration in range(max_iter):
            if not time_ok():
                break
            
            merged = False
            # Try all pairs (but limit for time)
            n_terms = len(current_terms)
            # Sample pairs to save time
            if n_terms > 40:
                pairs = list(combinations(range(n_terms), 2))
                # Shuffle and take a subset
                np.random.shuffle(pairs)
                pairs = pairs[:min(200, len(pairs))]
            else:
                pairs = list(combinations(range(n_terms), 2))
            
            for a, b in pairs:
                if a >= len(current_terms) or b >= len(current_terms):
                    continue
                if a == b:
                    continue
                    
                result = try_merge(current_terms, a, b)
                if result is not None:
                    new_term, remaining = result
                    current_terms = remaining + [new_term]
                    rank = len(current_terms)
                    merged = True
                    break
            
            if not merged:
                break
        
        return current_terms, rank

    # Even better structural approach: 
    # Use the fact that for matrix multiplication, we can apply 
    # "row/column splitting" techniques.
    # 
    # Key insight: For a 3x4*4x6 multiplication, we can think of it as:
    # C = A * B where A is 3x4, B is 4x6
    # 
    # We can split B's columns into groups and use different decompositions
    # for each group, potentially sharing some rank-1 terms.
    #
    # Alternative structural approach: Use the "laser method" or 
    # "border rank" ideas from algebraic complexity.
    #
    # Let me try a completely different approach: 
    # Greedy rank reduction using linear algebra.
    # Start with schoolbook, then for each pair of terms, check if their
    # tensor (outer product of u, v, w vectors) is in the span of the others.
    
    def tensor_vector(u_dict, v_dict, w_dict):
        """Convert a rank-1 term to a flat vector representation."""
        vec = {}
        for ui, uc in u_dict.items():
            for vi, vc in v_dict.items():
                for wi, wc in w_dict.items():
                    key = (ui, vi, wi)
                    vec[key] = vec.get(key, 0) + uc * vc * wc
        return vec

    def try_rank_reduction_linalg(terms):
        """
        Try to reduce rank by finding linear dependencies among the tensor vectors.
        If term_i = sum_j c_j * term_j for j != i, we can eliminate term_i.
        """
        current_terms = list(terms)
        
        for iteration in range(20):  # Limit iterations
            if not time_ok():
                break
            
            n_terms = len(current_terms)
            if n_terms <= 1:
                break
            
            # Build matrix of tensor vectors
            # Get all unique (ui, vi, wi) indices
            all_indices = set()
            for t in current_terms:
                tv = tensor_vector(t[0], t[1], t[2])
                all_indices.update(tv.keys())
            
            if len(all_indices) == 0:
                break
            
            index_list = list(all_indices)
            index_to_pos = {idx: pos for pos, idx in enumerate(index_list)}
            
            # Build matrix M where M[i][j] = coefficient of index j in term i
            M = np.zeros((n_terms, len(index_list)))
            for i, (ud, vd, wd) in enumerate(current_terms):
                tv = tensor_vector(ud, vd, wd)
                for idx, val in tv.items():
                    M[i, index_to_pos[idx]] = val
            
            # Try to find a row that's a linear combination of others
            # Use rank computation
            rank_M = np.linalg.matrix_rank(M, tol=1e-9)
            
            if rank_M < n_terms:
                # There's a linear dependency
                # Find which row to eliminate
                # Use QR with column pivoting
                Q, R = np.linalg.qr(M.T, mode='complete')
                # The last row of R being ~0 indicates dependency
                # Actually, let's find the null space of M
                _, S, Vt = np.linalg.svd(M, full_matrices=True)
                # Null space vectors
                null_dim = n_terms - rank_M
                if null_dim > 0:
                    # Get null space basis
                    null_vecs = Vt[rank_M:].T
                    # Find a null vector with the largest coefficient
                    for nv in null_vecs:
                        # Check if we can eliminate a term
                        max_idx = np.argmax(np.abs(nv))
                        if abs(nv[max_idx]) > 1e-6:
                            # Try to eliminate term max_idx
                            # But we need the coefficients to be "nice"
                            # For now, just mark as potential reduction
                            pass
                break  # Conservative: break after first check
            
            break  # No dependency found, stop
        
        return current_terms, len(current_terms)

    # Let me try yet another structural approach that's more likely to work:
    # Use the "splitting" method from algebraic complexity theory.
    # 
    # For 3x4 * 4x6, we can view this as a sum of smaller matrix multiplications
    # with shared structure.
    #
    # Actually, the most promising structural change is to use a 
    # "parametric" decomposition where we allow rational coefficients
    # and search over a parameter space.
    #
    # But for a concrete improvement, let me try:
    # 1. Start with Strassen-based rank 66
    # 2. Try to further reduce by exploiting the specific dimensions
    #
    # For 3x4*4x6, note that 3 = 2+1 and 6 = 2*3.
    # We can use Strassen for the 2x2 part and handle the remaining row separately.
    # But we can also try to share terms between the Strassen blocks and the extra row.
    
    def build_strassen_block(i1, i2, j1, j2, k1, k2):
        """
        Generate 7 Strassen terms for a 2x2x2 block.
        A rows i1,i2; A cols j1,j2; B rows j1,j2; B cols k1,k2; C rows i1,i2; C cols k1,k2
        """
        terms = []
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        terms.append(({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
        terms.append(({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
        terms.append(({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
        terms.append(({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
        terms.append(({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}))
        terms.append(({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
        terms.append(({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
        return terms

    def build_schoolbook_row(i, j_start, j_end, k_start, k_end):
        terms = []
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    def build_full_schoolbook():
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    # Strategy 1: Strassen on rows 1-2 (as before, rank 66)
    def strategy_strassen_rows12():
        terms = []
        terms.extend(build_schoolbook_row(0, 0, m, 0, p))  # 24 terms
        for j1 in range(0, m, 2):
            for k1 in range(0, p, 2):
                terms.extend(build_strassen_block(1, 2, j1, j1+1, k1, k1+1))  # 4*7=28 terms
        return terms  # 24 + 28 = 52? Wait, 4 j-blocks * 3 k-blocks... no, m=4 so 2 j-blocks, p=6 so 3 k-blocks
        # Actually: j1 in [0,2], k1 in [0,2,4] -> 2*3=6 blocks * 7 = 42, plus 24 = 66

    # Strategy 2: Strassen on rows 0-1
    def strategy_strassen_rows01():
        terms = []
        terms.extend(build_schoolbook_row(2, 0, m, 0, p))  # 24 terms
        for j1 in range(0, m, 2):
            for k1 in range(0, p, 2):
                terms.extend(build_strassen_block(0, 1, j1, j1+1, k1, k1+1))
        return terms  # 24 + 42 = 66

    # Strategy 3: Try to use a "hybrid" where we split columns differently
    # For the 4x6 part, split into 2+2+2 columns
    # Use Strassen on 2x2x2 blocks but with different groupings
    
    # Strategy 4: Use a 3-way split on the middle dimension
    # A is 3x4, B is 4x6. Split the 4 middle dimension into 2+2.
    # C = A[:,:2]*B[:,:2] + A[:,2:]*B[:,2:]
    # Each is 3x2 * 2x6 = 3x6
    # For 3x2 * 2x6: we can use Strassen-like methods
    
    def build_3x2_2x6_block(i0, j0, k0):
        """
        Decompose 3x2 * 2x6 matrix multiplication.
        A: rows i0..i0+2, cols j0..j0+1
        B: rows j0..j0+1, cols k0..k0+5
        C: rows i0..i0+2, cols k0..k0+5
        """
        terms = []
        # A has 3x2 = 6 elements, B has 2x6 = 12 elements, C has 3x6 = 18 elements
        # Schoolbook: 6*6 = 36 terms... no, for each (i,k) pair, sum over j
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # For fixed i,k: 2 terms (j=0,1)
        # Total: 3*6*2 = 36 schoolbook terms
        
        # Can we do better? 3x2 * 2x6 can be seen as:
        # C = [A[:,0]; A[:,1]] * [B[0,:]; B[1,:]]
        # = A[:,0] * B[0,:] + A[:,1] * B[1,:]
        # Each is an outer product: 3x6, so 18 elements each
        # But we need rank-1 terms...
        
        # Actually, for 3x2 * 2x6, the tensor rank is at most 36 (schoolbook).
        # Can we do better? 
        # 3x2 * 2x6 is equivalent to computing 18 dot products of length 2.
        # Each dot product needs 2 multiplications, so 36 total.
        # But we might share some structure.
        
        # Let's just use schoolbook for this sub-block
        for i in range(3):
            for j in range(2):
                for k in range(6):
                    u_idx = (i0 + i) * m + (j0 + j)
                    v_idx = (j0 + j) * p + (k0 + k)
                    w_idx = (k0 + k) * n + (i0 + i)
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    # Strategy 5: More aggressive - try to find a decomposition using
    # the "laser method" or known results for rectangular matrix multiplication.
    # 
    # For 3x4 * 4x6, the exponent is log_2(omega(3,4,6)).
    # Known bounds suggest rank around 50-60 might be achievable.
    # 
    # Let me try a completely different approach: use the "border rank" 
    # or "subspace" methods.
    #
    # Actually, let me try a greedy algorithm that works on the actual
    # tensor entries and tries to find low-rank approximations that are exact.
    
    def greedy_tensor_decomposition():
        """
        Greedy algorithm to find a low-rank exact decomposition.
        Start with the full tensor and iteratively extract rank-1 components.
        """
        # Build the full tensor T[i,j,k] = 1 for all valid (i,j,k)
        # T has shape (n, m, p) = (3, 4, 6)
        T = np.ones((n, m, p))
        
        U_list = []
        V_list = []
        W_list = []
        
        remaining = T.copy()
        
        for rank in range(1, 80):
            if not time_ok():
                break
            
            # Find the best rank-1 approximation of remaining
            # Use SVD on the matricization
            # T[i,j,k] -> matricize as (i) x (j,k)
            T_mat = remaining.reshape(n, m * p)
            
            # SVD
            U_svd, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
            
            if len(S) == 0 or S[0] < 1e-10:
                break
            
            # Take the top singular vector
            u_vec = U_svd[:, 0]  # shape (n,)
            vk_vec = Vt[0, :]  # shape (m*p,)
            
            # Scale
            scale = S[0]
            
            # u_vec is shape (n,), vk_vec is shape (m*p,)
            # We need u to be shape (n*m,), v to be shape (m*p,), w to be shape (p*n,)
            # 
            # The rank-1 term is: u[i,j] * v[j,k] * w[k,i]
            # With u_vec[i] and vk_vec[j,k], we have:
            # u_vec[i] * vk_vec[j,k] = u[i,j] * v[j,k] * w[k,i]
            #
            # This is underdetermined. We need to factor vk_vec[j,k] into v[j,k]*w[k,i]*u[i,j]/u_vec[i]
            #
            # Let's set u[i,j] = u_vec[i] (same for all j)
            # Then v[j,k]*w[k,i] = vk_vec[j,k] / u_vec[i]
            # But this must hold for all i, which is generally impossible.
            
            # Alternative: Set u[i,j] = 1 for all (i,j), then:
            # v[j,k]*w[k,i] = u_vec[i] * vk_vec[j,k]
            # This is a rank-1 approximation of the matrix M[i,(j,k)] = u_vec[i]*vk_vec[j,k]
            # which is already rank 1! So we can factor it.
            
            # Set u[i,j] = 1 for all i,j
            # Then we need v[j,k]*w[k,i] = u_vec[i] * vk_vec[j,k]
            # Let v[j,k] = vk_vec[j,k] and w[k,i] = u_vec[i]
            # But w should be shape (p*n,) indexed by (k,i), so w[k*n+i] = u_vec[i]
            
            # Check: sum over j of u[i,j]*v[j,k]*w[k,i] = sum_j 1 * vk_vec[j,k] * u_vec[i]
            # = u_vec[i] * sum_j vk_vec[j,k]
            # This should equal remaining[i,j,k] = 1 for the current step.
            # This only works if u_vec[i] * sum_j vk_vec[j,k] = 1 for all (i,k),
            # which is generally not true.
            
            # Better approach: Use the actual SVD components
            # The rank-1 approximation is: u_vec[i] * vk_vec[j,k]
            # We want to express this as sum over the rank-1 structure.
            
            # Let's just subtract the best rank-1 approximation and continue
            # For now, use a simpler greedy: pick the largest entry and create a rank-1 term
            
            # Find the largest absolute entry in remaining
            idx = np.unravel_index(np.argmax(np.abs(remaining)), remaining.shape)
            i, j, k = idx
            val = remaining[i, j, k]
            
            if abs(val) < 1e-10:
                break
            
            # Create a rank-1 term that matches this entry
            # u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = val
            # But this will also affect other entries.
            
            # For exactness, let's use a different approach:
            # Just build the schoolbook and return it.
            break
        
        # Fallback to schoolbook
        return build_full_schoolbook()

    # Given the complexity of finding a better decomposition algorithmically,
    # let me focus on the most promising structural improvement:
    # 
    # The key insight is that for 3x4 * 4x6, we can use a "mixed" decomposition
    # where we apply Strassen to some blocks and schoolbook to others, but with
    # careful sharing of terms.
    #
    # Specifically, note that:
    # - Strassen saves 1 term per 2x2x2 block (4 -> 7 instead of 8... wait, 4*2=8 schoolbook, 7 Strassen)
    # - For rows 1-2, we have 2*4*6 = 48 schoolbook terms, Strassen gives 42
    # - Can we save more by sharing between adjacent Strassen blocks?
    #
    # Let me try: instead of independent Strassen blocks, use a "tiling" approach
    # where the Strassen multiplications share some intermediate results.
    #
    # Actually, the most impactful structural change would be to use a 
    # "recursive" or "hierarchical" decomposition.
    #
    # For 3x4 * 4x6:
    # Split rows into 2+1, columns into 2+2, k into 3+3
    # C[0:2, 0:6] = A[0:2, :] * B[:, 0:6]
    # C[2, 0:6] = A[2, :] * B[:, 0:6]
    #
    # For the 2x4 * 4x6 part, we can use Strassen recursively:
    # Split the 4 middle dimension into 2+2:
    # C = A[:,:2]*B[:,:2] + A[:,2:]*B[:,2:]
    # Each is 2x2 * 2x6, which can be further decomposed.
    
    def build_2x2_2x6(j0, k0):
        """
        2x2 * 2x6 matrix multiplication.
        A: 2x2, B: 2x6, C: 2x6
        Schoolbook: 2*2*6 = 24 terms
        Can we do better?
        
        C[i,k] = sum_j A[i,j]*B[j,k]
        For fixed i: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
        This is a linear combination of two rows of B.
        
        We can use: M1 = A[0,0]*B[0,:], M2 = A[0,1]*B[1,:], etc.
        But that's 24 terms.
        
        Alternatively, use Strassen-like:
        M1 = (A[0,0]+A[1,1])*(B[0,:]+B[1,:])
        M2 = A[1,0]*B[0,:]
        M3 = A[1,1]*B[1,:]
        M4 = (A[0,0]+A[0,1])*B[1,:]
        M5 = (A[1,0]-A[0,0])*B[0,:]
        
        Then:
        C[0,0] = M1 - M3 - M5 ... this gets complicated for the 6 columns.
        
        Let me just use schoolbook for this sub-block.
        """
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    u_idx = i * m + (j0 + j)
                    v_idx = (j0 + j) * p + (k0 + k)
                    w_idx = (k0 + k) * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms

    # I think the most promising approach given the time constraints is to
    # stick with the Strassen-based rank 66 but try a few variations and
    # return the best one.
    
    strategies = [
        strategy_strassen_rows12,
        strategy_strassen_rows01,
        build_full_schoolbook,
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for strat_fn in strategies:
        if not time_ok():
            break
        terms = strat_fn()
        rank = len(terms)
        if rank >= best_rank:
            continue
        U, V, W = terms_to_matrices(terms)
        if verify(U, V, W):
            best_result = (U, V, W, rank)
            best_rank = rank
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    terms = build_full_schoolbook()
    U, V, W = terms_to_matrices(terms)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
