import time
import numpy as np

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60
    
    # Build a large pool of candidate rank-1 tensors
    # A rank-1 tensor is (u_vec, v_vec, w_vec) where:
    #   u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # and the contribution to the (i,j,k) entry is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    candidates = []  # list of (u_vec, v_vec, w_vec, contrib_vec)
    
    def add_candidate(u_sparse, v_sparse, w_sparse):
        """Add a candidate defined by sparse dicts."""
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        
        # Compute contribution vector
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        # Only add if non-zero
        if np.any(contrib):
            candidates.append((u_vec, v_vec, w_vec, contrib))
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})
    
    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        terms = []
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms
    
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
    
    # Type 3: Generate diverse structured candidates for wider blocks
    # For 2x3x3 blocks, generate all combinations of +/-1 patterns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                # Generate candidates with various sign patterns
                # Pattern: sum of A entries times sum of B entries times sum of W entries
                for i_pat in [[1,1],[1,-1],[-1,1]]:
                    for j_pat in [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]:
                        for k_pat in [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]:
                            u_sparse = {}
                            v_sparse = {}
                            w_sparse = {}
                            for ii in range(2):
                                u_sparse[(io+ii, jo)] = i_pat[ii]
                                u_sparse[(io+ii, jo+1)] = i_pat[ii]
                                u_sparse[(io+ii, jo+2)] = i_pat[ii]
                            for jj in range(3):
                                v_sparse[(jo+jj, ko)] = j_pat[jj]
                                v_sparse[(jo+jj, ko+1)] = j_pat[jj]
                                v_sparse[(jo+jj, ko+2)] = j_pat[jj]
                            for kk in range(3):
                                w_sparse[(ko+kk, io)] = k_pat[kk]
                                w_sparse[(ko+kk, io+1)] = k_pat[kk]
                            add_candidate(u_sparse, v_sparse, w_sparse)
    
    # Type 4: Diagonal-like patterns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                for shift in range(3):
                    u_s = {}
                    v_s = {}
                    w_s = {}
                    for ii in range(2):
                        for jj in range(3):
                            if (jj - ii) % 3 == shift:
                                u_s[(io+ii, jo+jj)] = 1
                    for jj in range(3):
                        for kk in range(3):
                            if (kk - jj) % 3 == shift:
                                v_s[(jo+jj, ko+kk)] = 1
                    for kk in range(3):
                        for ii in range(2):
                            if (ii - kk) % 3 == shift:
                                w_s[(ko+kk, io+ii)] = 1
                    add_candidate(u_s, v_s, w_s)
    
    # Type 5: S3 group-theoretic candidates for 2x2x2 blocks.
    # The S3 group (order 6) acts simultaneously on (i,j,k) indices of a 2x2x2
    # block. Its regular representation decomposes into irreps of dimensions 1,1,2,
    # giving rank bound 1+1+4=6 (matching Strassen). We generate all 6 group
    # elements as structured candidates, plus their products with diagonal scalings,
    # to enrich the candidate pool for the greedy solver.
    S3_perms = [
        (0, 0, 0),  # identity
        (1, 1, 1),  # swap both
        (0, 1, 1),  # swap j,k
        (1, 0, 0),  # swap i,k
        (1, 1, 0),  # swap i,j  (note: this maps (i,j,k)->(j,i,k) but we need consistency)
        (0, 0, 1),  # swap i,j,k cycle
    ]
    # More carefully: S3 elements as permutations of 3 positions.
    # For a 2x2x2 block at (io, jo, ko), a group element g acts as:
    #   (i, j, k) -> (pi, pj, pk) where (pi,pj,pk) is the permutation applied
    # The rank-1 tensor for group element g has:
    #   u[i,j] = 1 if (i,j) is in the "support" pattern
    #   v[j,k] = 1 if (j,k) is in the "support" pattern
    #   w[k,i] = 1 if (k,i) is in the "support" pattern
    # For the regular representation, each group element gives a character-based term.
    # We'll generate candidates for each S3 element acting on each 2x2x2 block.
    
    # S3 elements as permutations of {0,1,2}
    S3_elements = [
        [0, 1, 2],  # identity
        [1, 0, 2],  # swap 0,1
        [0, 2, 1],  # swap 1,2
        [2, 1, 0],  # reverse
        [1, 2, 0],  # cycle 0->1->2->0
        [2, 0, 1],  # cycle 0->2->1->0
    ]
    
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for perm in S3_elements:
                    # For each group element, create a structured candidate
                    # The group element perm acts on the 3 "dimensions" (i,j,k)
                    # of the 2x2x2 block. We create a rank-1 tensor where:
                    # u[i,j] = 1 for entries consistent with perm
                    # v[j,k] = 1 for entries consistent with perm
                    # w[k,i] = 1 for entries consistent with perm
                    
                    # Interpretation: perm tells us how to "mix" the indices.
                    # For perm = [p0, p1, p2], the group element maps:
                    #   position 0 -> p0, position 1 -> p1, position 2 -> p2
                    # In the 2x2x2 block, we have indices (0,1) for each dim.
                    # The group element gives a permutation matrix P (3x3) where
                    # P[a,b] = 1 if perm[a] = b.
                    
                    # For the regular representation approach, each group element g
                    # contributes a term where:
                    #   u[i,j] = delta_{i, pi(j)} for the permutation part
                    # This is getting complex; let's use a simpler structured approach.
                    
                    # Simpler: for each S3 element, generate a candidate where
                    # the three factor vectors are determined by the permutation.
                    # This gives diverse structured candidates beyond basic Strassen.
                    u_s = {}
                    v_s = {}
                    w_s = {}
                    
                    # perm = [p0, p1, p2] means the group element maps
                    # e0->e_{p0}, e1->e_{p1}, e2->e_{p2}
                    # For the 2x2x2 block, we interpret this as:
                    #   u[i,j] = 1 if the pair (i,j) is "aligned" with perm
                    #   v[j,k] = 1 if the pair (j,k) is "aligned" with perm
                    #   w[k,i] = 1 if the pair (k,i) is "aligned" with perm
                    
                    # "Aligned" means: for perm, the support is on entries where
                    # the index transformation matches the permutation structure.
                    
                    # For identity [0,1,2]: diagonal in each factor
                    # For swap [1,0,2]: anti-diagonal in first two
                    # etc.
                    
                    # Let's define: for perm, u[i,j] = 1 if perm maps the "role"
                    # of i to the "role" of j consistently.
                    # Role of i in 2x2x2: i is dim 0, j is dim 1, k is dim 2.
                    # perm says dim a goes to dim perm[a].
                    # So u[i,j] gets 1 if the mapping is consistent:
                    #   i (dim 0) maps to dim perm[0], j (dim 1) maps to dim perm[1]
                    # For u to be non-zero at (i,j), we need perm[0] and perm[1]
                    # to be "compatible" with the 2x2 structure.
                    
                    # Simplified: just generate all 6 S3 elements as
                    # permutation-based structured tensors.
                    
                    if perm == [0, 1, 2]:  # identity
                        u_s = {(io, jo): 1, (io+1, jo+1): 1}
                        v_s = {(jo, ko): 1, (jo+1, ko+1): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [1, 0, 2]:  # swap i,j roles
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo, ko): 1, (jo+1, ko+1): 1}
                        w_s = {(ko, io+1): 1, (ko+1, io): 1}
                    elif perm == [0, 2, 1]:  # swap j,k roles
                        u_s = {(io, jo): 1, (io+1, jo+1): 1}
                        v_s = {(jo, ko+1): 1, (jo+1, ko): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [2, 1, 0]:  # reverse
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo, ko+1): 1, (jo+1, ko): 1}
                        w_s = {(ko, io+1): 1, (ko+1, io): 1}
                    elif perm == [1, 2, 0]:  # cycle
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo+1, ko): 1, (jo, ko+1): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [2, 0, 1]:  # inverse cycle
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo+1, ko): 1, (jo, ko+1): 1}
                        w_s = {(ko+1, io): 1, (ko, io+1): 1}
                    
                    add_candidate(u_s, v_s, w_s)
    
    # Type 6: 2x3x2 and 3x2x2 structured blocks using group-theoretic ideas
    # For 2x3x2 blocks, use S3 acting on the 3 middle indices
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                for perm in S3_elements:
                    u_s = {}
                    v_s = {}
                    w_s = {}
                    # u: 2x3, v: 3x2, w: 2x2
                    # perm acts on the 3 j-indices
                    for jj in range(3):
                        u_s[(io, jo+jj)] = 1
                        u_s[(io+1, jo+jj)] = 1
                    for jj in range(3):
                        for kk in range(2):
                            if perm[jj] == kk:  # simplified alignment
                                v_s[(jo+jj, ko+kk)] = 1
                    for kk in range(2):
                        for ii in range(2):
                            w_s[(ko+kk, io+ii)] = 1
                    add_candidate(u_s, v_s, w_s)
    
    # Now use greedy to find a good decomposition
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    num_candidates = len(candidates)
    
    # Precompute norms of all candidates
    norms = np.zeros(num_candidates)
    for idx in range(num_candidates):
        vec = candidates[idx][3]
        norms[idx] = np.dot(vec, vec)
    
    # Build a matrix of all candidate contributions for faster dot products
    # cand_matrix: shape (total_entries, num_candidates)
    cand_matrix = np.zeros((total_entries, num_candidates))
    for idx in range(num_candidates):
        cand_matrix[:, idx] = candidates[idx][3]
    
    max_iter = 150
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Vectorized: compute all dot products at once
        dots = cand_matrix.T @ residual  # shape (num_candidates,)
        coeffs = np.zeros(num_candidates)
        coeffs[norms > 1e-12] = dots[norms > 1e-12] / norms[norms > 1e-12]
        reductions = dots * coeffs
        
        # Mask out invalid
        reductions[norms < 1e-12] = -1e18
        
        best_idx = int(np.argmax(reductions))
        best_reduction = reductions[best_idx]
        best_coeff = coeffs[best_idx]
        
        if best_idx == -1 or best_reduction < 1e-9:
            break
        
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * candidates[best_idx][3]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Refinement pass: try to reduce rank by checking if any selected term
    # can be removed and its contribution absorbed by others
    if len(selected_indices) > 10:
        for trial_remove in range(len(selected_indices)):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Temporarily remove this term and see if residual is still small
            test_residual = T_target.copy()
            for r, idx in enumerate(selected_indices):
                if r == trial_remove:
                    continue
                test_residual -= selected_coeffs[r] * candidates[idx][3]
            
            if np.linalg.norm(test_residual) < 1e-8:
                # This term is redundant, remove it
                selected_indices.pop(trial_remove)
                selected_coeffs.pop(trial_remove)
                break
    
    # Check if we got an exact solution
    if np.linalg.norm(residual) < 1e-9 and len(selected_indices) <= 60:
        rank_found = len(selected_indices)
        
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec
        
        # Exact rank reduction via Smith Normal Form on the combined factor matrix.
        # The decomposition is exact if for all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # If some columns are linearly dependent in a way that preserves the tensor,
        # we can merge them. We check for linear dependencies among the "effective"
        # rank-1 tensors.
        
        # Build the effective contribution matrix: each selected term r contributes
        # a vector of length n*m*p. If these vectors are linearly dependent, we can
        # reduce rank.
        eff_matrix = np.zeros((total_entries, rank_found))
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            eff_matrix[:, r] = coeff * candidates[idx][3]
        
        # Find rank of eff_matrix via SVD
        try:
            s_vals, _, Vt = np.linalg.svd(eff_matrix, full_matrices=False)
            tol = 1e-8 * max(s_vals) if len(s_vals) > 0 else 1e-8
            num_indep = np.sum(s_vals > tol)
            
            if num_indep < rank_found:
                # Some columns are linearly dependent. Reconstruct using only
                # the independent directions.
                # Use the SVD: eff_matrix ≈ U_svd * diag(s) * Vt
                # The tensor is sum_r (coeff_r * cand_r) = eff_matrix @ ones
                # = U_svd * diag(s) * Vt @ ones
                # = U_svd * diag(s) * (Vt @ ones)
                # The effective rank-1 terms are the columns of U_svd * diag(s)
                # weighted by (Vt @ ones).
                
                ones_vec = np.ones(rank_found)
                weights = Vt @ ones_vec  # shape (rank_found,)
                
                # New decomposition: sum_{a=1}^{num_indep} (U_svd[:,a] * s[a] * weights[a])
                # Each term is a vector of length total_entries that equals
                # U_svd[:,a] * (s[a] * weights[a])
                # But we need to express this as rank-1 tensor products!
                # This is the hard part - the SVD gives us vectors in the tensor space,
                # not necessarily rank-1 tensors.
                
                # Alternative: just keep the greedy solution as-is if SVD doesn't
                # directly give rank-1 factors. The linear dependence might not
                # correspond to removable rank-1 terms.
                # For now, skip this reduction and rely on the greedy + removal pass.
                pass
        except Exception:
            pass
        
        # Verify
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val
        
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the known 54-rank solution
    n, m, p = 2, 5, 6
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        # r=0: M1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=1: M2
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        # r=2: M3
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=3: M4
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=4: M5
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        # r=5: M6
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=6: M7
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
