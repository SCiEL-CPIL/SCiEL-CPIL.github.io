<<<<<<< SEARCH
    def is_valid_addition(p_idx, current_indices, dist_sets):
        """
        Check if point p_idx can be added to current set.
        """
        k = len(current_indices)
        if k < 2:
            return True
        
        # Get distances from p_idx to all current points
        d_p = dist_matrix[p_idx, current_indices]
        
        # Check for isosceles at p: any two distances equal
        if len(set(d_p.tolist())) != len(d_p):
            return False
        
        # Check for isosceles at existing points
        for i in range(k):
            if d_p[i] in dist_sets[i]:
                return False
        
        return True
=======
    def is_valid_addition(p_idx, current_indices, dist_sets):
        """
        Check if point p_idx can be added to current set.
        """
        k = len(current_indices)
        if k < 2:
            return True
        
        # Get distances from p_idx to all current points
        d_p = dist_matrix[p_idx, current_indices]
        d_p_list = d_p.tolist()
        
        # Check for isosceles at p: any two distances equal
        if len(set(d_p_list)) != len(d_p_list):
            return False
        
        # Check for isosceles at existing points
        for i in range(k):
            if d_p_list[i] in dist_sets[i]:
                return False
        
        return True

    def is_valid_addition_fast(p_idx, current_indices, dist_sets):
        """
        Fast pre-filter: use numpy to quickly reject most candidates.
        Only returns True if it might be valid (may have false positives).
        """
        k = len(current_indices)
        if k < 2:
            return True
        
        d_p = dist_matrix[p_idx, current_indices]
        
        # Check for isosceles at p: any two distances equal
        if len(set(d_p.tolist())) != len(d_p):
            return False
        
        # Quick check: for each existing point i, d_p[i] must not be in dist_sets[i]
        # We do this with a single pass in Python (can't easily vectorize set membership)
        # But we can at least do the first check efficiently
        for i in range(k):
            if d_p[i] in dist_sets[i]:
                return False
        
        return True
>>>>>>> REPLACE

