   def _can_add_idx(p_idx, S_indices):
       if len(S_indices) < 2:
           return True
       
       d_p_s = D2[p_idx, S_indices]
       
       # Cond 1
       vals, counts = np.unique(d_p_s, return_counts=True)
       if np.any(counts > 1):
           return False
           
       # Cond 2/3
       k = len(S_indices)
       # Generate pairs
       ii, jj = np.triu_indices(k, 1)
       if len(ii) == 0:
           return True
           
       i_pts = S_indices[ii]
       j_pts = S_indices[jj]
       
       d_ij = D2[i_pts, j_pts]
       d_pi = D2[p_idx, i_pts]
       d_pj = D2[p_idx, j_pts]
       
       if np.any(d_ij == d_pi) or np.any(d_ij == d_pj):
           return False
           
       return True
   