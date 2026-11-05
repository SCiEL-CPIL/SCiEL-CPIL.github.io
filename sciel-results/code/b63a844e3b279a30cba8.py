<<<<<<< SEARCH
    # Greedy post-pass: merge pairs of terms whose outer products sum to a rank-1 term
    def try_merge(terms):
        """Try to find pairs (i,j) where sum of their rank-1 tensors is itself rank-1."""
        n_terms = len(terms)
        if n_terms < 2:
            return terms
        deadline_merge = start_time + 140.0
        
        # Build sparse representation for quick outer-product checking
        # For each term, store as dict of (u_idx, v_idx) -> w_idx contributions
        # Actually, we need: for terms i and j, check if T_i + T_j is rank-1
        # T_i = U_i ⊗ V_i ⊗ W_i (3-way tensor)
        # T_i + T_j is rank-1 iff there exist a,b,c (scalars) and rank-1 U',V',W' s.t.
        # U_i⊗V_i⊗W_i + U_j⊗V_j⊗W_j = a*U'⊗V'⊗W'
        # This is equivalent to: the 2-dimensional subspace spanned by {U_i⊗V_i, U_j⊗V_j}
        # being 1-dimensional AND the corresponding W vectors being proportional... 
        # More practically: check if there exist scalars α,β,γ,δ such that
        # α*U_i = γ*U', β*V_i = δ*V', etc.
        
        # Simpler approach: for each pair, check if the combined 3-tensor has rank 1
        # by checking if all 2x2 minors of the matrix unfoldings vanish.
        
        merged = list(terms)
        removed = [False] * n_terms
        
        for i in range(n_terms):
            if removed[i] or time.time() > deadline_merge:
                continue
            for j in range(i + 1, n_terms):
                if removed[j] or time.time() > deadline_merge:
                    continue
                
                # Check if term_i + term_j is rank-1
                # Collect all (u,v) pairs that appear in either term
                uv_pairs = {}
                for u_idx, u_val in terms[i][0].items():
                    for v_idx, v_val in terms[i][1].items():
                        key = (u_idx, v_idx)
                        uv_pairs[key] = uv_pairs.get(key, 0) + u_val * v_val
                for u_idx, u_val in terms[j][0].items():
                    for v_idx, v_val in terms[j][1].items():
                        key = (u_idx, v_idx)
                        uv_pairs[key] = uv_pairs.get(key, 0) + u_val * v_val
                
                # For the sum to be rank-1, all (u,v) pairs with non-zero coefficient
                # must have W contributions that are proportional
                w_vectors = {}
                for (u_idx, v_idx), coeff in uv_pairs.items():
                    if coeff == 0:
                        continue
                    # Get W contribution from term i at this (u,v)
                    w_i = 0
                    if u_idx in terms[i][0] and v_idx in terms[i][1]:
                        w_i = terms[i][0][u_idx] * terms[i][1][v_idx]
                    # Get W contribution from term j
                    w_j = 0
                    if u_idx in terms[j][0] and v_idx in terms[j][1]:
                        w_j = terms[j][0][u_idx] * terms[j][1][v_idx]
                    w_vectors[(u_idx, v_idx)] = (w_i, w_j)
                
                # For rank-1: there must exist λ such that for all (u,v),
                # the W vector is proportional to a single W'
                # i.e., (w_i, w_j) = s_{uv} * (w'_i, w'_j) for some common direction
                # This means all (w_i, w_j) pairs are collinear in R^2
                if len(w_vectors) <= 1:
                    # Trivially rank-1
                    pass
                else:
                    # Check collinearity of all (w_i, w_j) vectors
                    first = list(w_vectors.values())[0]
                    collinear = True
                    for w_pair in w_vectors.values():
                        # Cross product in 2D: w_i1*w_j2 - w_i2*w_j1
                        cross = first[0] * w_pair[1] - first[1] * w_pair[0]
                        if cross != 0:
                            collinear = False
                            break
                    if not collinear:
                        continue
                
                # If we get here, the sum is rank-1. Create merged term.
                # The merged term's U, V, W need to be determined.
                # We need U'⊗V'⊗W' = T_i + T_j
                # Find a non-zero (u,v) pair to anchor
                anchor = None
                for (u_idx, v_idx), coeff in uv_pairs.items():
                    if coeff != 0:
                        anchor = (u_idx, v_idx)
                        break
                if anchor is None:
                    continue
                
                u_anchor, v_anchor = anchor
                # U'[u_anchor] * V'[v_anchor] * W'[k] = (T_i + T_j)[u_anchor, v_anchor, k]
                # We can set U'[u_anchor] = 1, then V'[v_anchor] * W'[k] = combined[k]
                # combined[k] = T_i[u,v,k] + T_j[u,v,k]
                
                # Get the combined W vector at (u_anchor, v_anchor)
                combined_w = {}
                for u_idx, u_val in terms[i][0].items():
                    if u_idx != u_anchor:
                        continue
                    for v_idx, v_val in terms[i][1].items():
                        if v_idx != v_anchor:
                            continue
                        for w_idx, w_val in terms[i][2].items():
                            combined_w[w_idx] = combined_w.get(w_idx, 0) + u_val * v_val * w_val
                for u_idx, u_val in terms[j][0].items():
                    if u_idx != u_anchor:
                        continue
                    for v_idx, v_val in terms[j][1].items():
                        if v_idx != v_anchor:
                            continue
                        for w_idx, w_val in terms[j][2].items():
                            combined_w[w_idx] = combined_w.get(w_idx, 0) + u_val * v_val * w_val
                
                # Now we need to factor U'⊗V'⊗W' = sum over (u,v,k) of M[u,v,k] e_u⊗e_v⊗e_k
                # where M is the combined 3-tensor.
                # This is complex. Let's use a simpler criterion:
                # Only merge if one term's U is proportional to the other's, etc.
                
                # Actually, let's use a much simpler sufficient condition:
                # If U_i = c * U_j for some scalar c, and V_i = d * V_j, and W_i = e * W_j,
                # then the sum is also rank-1.
                
                # Check if U_i is proportional to U_j
                u_i = terms[i][0]
                u_j = terms[j][0]
                if set(u_i.keys()) != set(u_j.keys()):
                    continue
                # Find ratio
                ratio_u = None
                proportional_u = True
                for key in u_i:
                    if u_j[key] == 0:
                        if u_i[key] != 0:
                            proportional_u = False
                            break
                    else:
                        r = u_i[key] / u_j[key]
                        if ratio_u is None:
                            ratio_u = r
                        elif abs(r - ratio_u) > 1e-9:
                            proportional_u = False
                            break
                if not proportional_u or ratio_u is None:
                    continue
                
                # Check V proportionality
                v_i = terms[i][1]
                v_j = terms[j][1]
                if set(v_i.keys()) != set(v_j.keys()):
                    continue
                ratio_v = None
                proportional_v = True
                for key in v_i:
                    if v_j[key] == 0:
                        if v_i[key] != 0:
                            proportional_v = False
                            break
                    else:
                        r = v_i[key] / v_j[key]
                        if ratio_v is None:
                            ratio_v = r
                        elif abs(r - ratio_v) > 1e-9:
                            proportional_v = False
                            break
                if not proportional_v or ratio_v is None:
                    continue
                
                # Check W proportionality
                w_i = terms[i][2]
                w_j = terms[j][2]
                if set(w_i.keys()) != set(w_j.keys()):
                    continue
                ratio_w = None
                proportional_w = True
                for key in w_i:
                    if w_j[key] == 0:
                        if w_i[key] != 0:
                            proportional_w = False
                            break
                    else:
                        r = w_i[key] / w_j[key]
                        if ratio_w is None:
                            ratio_w = r
                        elif abs(r - ratio_w) > 1e-9:
                            proportional_w = False
                            break
                if not proportional_w or ratio_w is None:
                    continue
                
                # All proportional: merge into single term
                # (U_i + U_j) ⊗ (V_i + V_j) ⊗ (W_i + W_j) is NOT correct in general
                # But T_i + T_j = U_i⊗V_i⊗W_i + U_j⊗V_j⊗W_j
                # = ratio_u*U_j ⊗ ratio_v*V_j ⊗ W_i + U_j ⊗ V_j ⊗ W_j
                # = U_j ⊗ V_j ⊗ (ratio_u*ratio_v*W_i + W_j)
                # So merged term: U=U_j, V=V_j, W=ratio_u*ratio_v*W_i + W_j
                merged_U = dict(u_j)
                merged_V = dict(v_j)
                merged_W = {}
                all_w_keys = set(w_i.keys()) | set(w_j.keys())
                for k in all_w_keys:
                    val = ratio_u * ratio_v * w_i.get(k, 0) + w_j.get(k, 0)
                    if val != 0:
                        merged_W[k] = val
                
                if merged_W:
                    # Replace term i with merged, remove term j
                    merged[i] = (merged_U, merged_V, merged_W)
                    removed[j] = True
        
        result = [merged[i] for i in range(n_terms) if not removed[i]]
        return result
=======
    # Greedy post-pass: merge pairs of terms whose sum is rank-1
    def try_merge(terms):
        """Find pairs (i,j) where T_i + T_j is rank-1, and merge them."""
        n_terms = len(terms)
        if n_terms < 2:
            return terms
        deadline_merge = start_time + 145.0
        
        def check_rank1_sum(term_i, term_j):
            """Check if term_i + term_j is a rank-1 tensor.
            Returns (True, merged_term) or (False, None)."""
            Ui, Vi, Wi = term_i
            Uj, Vj, Wj = term_j
            
            # Build the combined 3-tensor as a dict: (u,v,w) -> value
            combined = {}
            for u, uv in Ui.items():
                for v, vv in Vi.items():
                    for w, wv in Wi.items():
                        key = (u, v, w)
                        combined[key] = combined.get(key, 0) + uv * vv * wv
            for u, uv in Uj.items():
                for v, vv in Vj.items():
                    for w, wv in Wj.items():
                        key = (u, v, w)
                        combined[key] = combined.get(key, 0) + uv * vv * wv
            
            # Remove zeros
            combined = {k: v for k, v in combined.items() if v != 0}
            if not combined:
                return True, ({}, {}, {})
            
            # A 3-way tensor M is rank-1 iff for any two u-indices u1,u2 with M[u1,*,*] != 0 and M[u2,*,*] != 0,
            # the matrices M[u1,:,:] and M[u2,:,:] are proportional.
            # Similarly for v-slices and w-slices.
            
            # Get all u-indices with non-zero slices
            u_slices = {}
            for (u, v, w), val in combined.items():
                if u not in u_slices:
                    u_slices[u] = {}
                u_slices[u][(v, w)] = val
            
            u_indices = list(u_slices.keys())
            if len(u_indices) == 1:
                # Only one u-slice, so trivially rank-1 in u-direction
                # Check if the single slice is rank-1 (as a 2D matrix)
                slice_data = u_slices[u_indices[0]]
                # Check if slice is rank-1: all 2x2 minors vanish
                v_w_keys = list(slice_data.keys())
                if len(v_w_keys) <= 1:
                    return True, _extract_rank1(combined)
                # Check proportionality of rows (v-indices)
                v_slices = {}
                for (v, w), val in slice_data.items():
                    if v not in v_slices:
                        v_slices[v] = {}
                    v_slices[v][w] = val
                v_indices = list(v_slices.keys())
                if len(v_indices) == 1:
                    return True, _extract_rank1(combined)
                # Check if all v-rows are proportional
                ref_v = v_indices[0]
                ref_row = v_slices[ref_v]
                for v in v_indices[1:]:
                    row = v_slices[v]
                    # Check proportionality
                    common_w = set(ref_row.keys()) & set(row.keys())
                    if not common_w:
                        # If no common w, check if one is zero
                        if any(val != 0 for val in ref_row.values()) and any(val != 0 for val in row.values()):
                            return False, None
                        continue
                    ratio = None
                    for w in common_w:
                        if ref_row[w] != 0:
                            if ratio is None:
                                ratio = row[w] / ref_row[w]
                            elif abs(row[w] / ref_row[w] - ratio) > 1e-9:
                                return False, None
                        else:
                            if row[w] != 0:
                                return False, None
                    # Check w's in row but not in ref_row
                    for w in row.keys():
                        if w not in ref_row and row[w] != 0:
                            return False, None
                    # Check w's in ref_row but not in row
                    for w in ref_row.keys():
                        if w not in row and ref_row[w] != 0:
                            return False, None
                return True, _extract_rank1(combined)
            
            # Multiple u-slices: check proportionality
            ref_u = u_indices[0]
            ref_slice = u_slices[ref_u]
            for u in u_indices[1:]:
                sl = u_slices[u]
                # Check proportionality of ref_slice and sl
                common_vw = set(ref_slice.keys()) & set(sl.keys())
                if not common_vw:
                    if any(val != 0 for val in ref_slice.values()) and any(val != 0 for val in sl.values()):
                        return False, None
                    continue
                ratio = None
                for vw in common_vw:
                    if ref_slice[vw] != 0:
                        if ratio is None:
                            ratio = sl[vw] / ref_slice[vw]
                        elif abs(sl[vw] / ref_slice[vw] - ratio) > 1e-9:
                            return False, None
                    else:
                        if sl[vw] != 0:
                            return False, None
                for vw in sl.keys():
                    if vw not in ref_slice and sl[vw] != 0:
                        return False, None
                for vw in ref_slice.keys():
                    if vw not in sl and ref_slice[vw] != 0:
                        return False, None
            
            # All u-slices are proportional, so the tensor is rank-1
            return True, _extract_rank1(combined)
        
        def _extract_rank1(combined):
            """Extract U, V, W dicts from a rank-1 combined tensor."""
            # Find a non-zero entry to anchor
            anchor = None
            for (u, v, w), val in combined.items():
                if val != 0:
                    anchor = (u, v, w, val)
                    break
            if anchor is None:
                return ({}, {}, {})
            u0, v0, w0, val0 = anchor
            
            # U[u] = combined[u, v0, w0] / val0 (if non-zero)
            # V[v] = combined[u0, v, w0] / val0
            # W[w] = combined[u0, v0, w] / val0
            
            U = {}
            V = {}
            W = {}
            
            for (u, v, w), val in combined.items():
                if v == v0 and w == w0 and val != 0:
                    U[u] = val / val0
                if u == u0 and w == w0 and val != 0:
                    V[v] = val / val0
                if u == u0 and v == v0 and val != 0:
                    W[w] = val / val0
            
            # Clean up: remove zeros
            U = {k: v for k, v in U.items() if v != 0}
            V = {k: v for k, v in V.items() if v != 0}
            W = {k: v for k, v in W.items() if v != 0}
            
            return (U, V, W)
        
        merged = list(terms)
        removed = [False] * n_terms
        
        for i in range(n_terms):
            if removed[i] or time.time() > deadline_merge:
                continue
            for j in range(i + 1, n_terms):
                if removed[j] or time.time() > deadline_merge:
                    continue
                
                is_rank1, merged_term = check_rank1_sum(terms[i], terms[j])
                if is_rank1 and merged_term[0] and merged_term[1] and merged_term[2]:
                    merged[i] = merged_term
                    removed[j] = True
                elif is_rank1:
                    # Empty term, just remove j
                    removed[j] = True
        
        result = [merged[i] for i in range(n_terms) if not removed[i]]
        return result
>>>>>>> REPLACE

Now let me also add a second pass that tries to merge the newly merged terms with others:

