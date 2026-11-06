        old_vals = counts[old_slice].copy()
        new_vals = counts[new_slice].copy()
        
        counts[old_slice] -= 1
        lost = np.sum(old_vals == 1)
        
        counts[new_slice] += 1
        gained = np.sum(new_vals == 0)
        
        delta_uncovered = lost - gained
