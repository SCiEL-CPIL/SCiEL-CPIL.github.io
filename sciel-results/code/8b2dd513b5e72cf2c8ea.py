<<<<<<< SEARCH
    def can_add_fast(s_idx, selected):
        if len(selected) < 2:
            return True
        
        # Use inter matrix for fast access
        # selected is a list of indices
        n_sel = len(selected)
        
        # We can't easily vectorize in Python loop without numpy.
        # Let's stick to the numpy version for larger sets, 
        # but the numpy version creates temporary arrays which is slow for small N.
        
        # Let's try a pure Python loop using the inter matrix (list of lists? No, numpy array).
        # Accessing numpy array in Python loop is slow due to type conversion.
        # Converting selected to numpy array and doing vectorized ops is better for N > ~20.
        
        if n_sel < 20:
            # Python loop
            for i in range(n_sel):
                a_idx = selected[i]
                ia_s = inter[s_idx, a_idx]
                for j in range(i + 1, n_sel):
                    b_idx = selected[j]
                    ia_b = inter[a_idx, b_idx]
                    ib_s = inter[s_idx, b_idx]
                    if ia_b == ia_s == ib_s:
                        return False
            return True
        else:
            # NumPy vectorized
            sel_arr = np.array(selected, dtype=np.int32)
            V = inter[s_idx, sel_arr]
            M = inter[sel_arr[:, None], sel_arr[None, :]]
            
            # Check condition
            # M[i,j] == V[i] and V[i] == V[j]
            # Note: V is 1D, M is 2D.
            # V[:, None] is column vector.
            
            # We can do:
            # mask1 = (M == V[:, None])
            # mask2 = (V[:, None] == V[None, :])
            # return np.any(mask1 & mask2)
            
            # Optimization: 
            # If V has all unique values, then V[i] != V[j] for i!=j, so no conflict.
            # We can check this quickly.
            if len(np.unique(V)) == len(V):
                return True
            
            mask1 = (M == V[:, None])
            mask2 = (V[:, None] == V[None, :])
            return np.any(mask1 & mask2)
=======
    def can_add_fast(s_idx, selected):
        if len(selected) < 2:
            return True
        
        n_sel = len(selected)
        
        if n_sel < 30:
            # Python loop for small sizes to avoid NumPy overhead
            for i in range(n_sel):
                a_idx = selected[i]
                ia_s = inter[s_idx, a_idx]
                for j in range(i + 1, n_sel):
                    b_idx = selected[j]
                    ia_b = inter[a_idx, b_idx]
                    ib_s = inter[s_idx, b_idx]
                    if ia_b == ia_s == ib_s:
                        return False
            return True
        else:
            # NumPy vectorized with group optimization
            sel_arr = np.array(selected, dtype=np.int32)
            V = inter[s_idx, sel_arr]
            
            # Optimization: Group indices by their intersection value with s.
            # A conflict exists if there are i, j in the same group (V[i]==V[j]==v)
            # such that inter[i, j] == v.
            
            unique_vals, inverse = np.unique(V, return_inverse=True)
            
            for v, group_id in zip(unique_vals, range(len(unique_vals))):
                # Get indices in this group
                idxs = np.where(inverse == group_id)[0]
                if len(idxs) < 2:
                    continue
                
                # Check if any pair in this group has intersection v
                # M_sub[i, j] = inter[sel_arr[idxs[i]], sel_arr[idxs[j]]]
                # We need M_sub[i, j] == v
                sub_idx = sel_arr[idxs]
                M_sub = inter[sub_idx[:, None], sub_idx[None, :]]
                
                # Check if any entry in M_sub is equal to v
                # Diagonal entries are inter[x, x] = mask_x. 
                # As argued before, mask_x cannot equal v (intersection of x and s) 
                # unless x == s, which is not possible since s is not in selected.
                # So we can just check np.any(M_sub == v)
                if np.any(M_sub == v):
                    return False
            
            return True
>>>>>>> REPLACE

We also need to adjust the time limit to be safer. The previous limit was 80s. The hard limit is 90s. 80s is risky if the first few iterations are slow or if there is overhead. I will set it to 75s.

