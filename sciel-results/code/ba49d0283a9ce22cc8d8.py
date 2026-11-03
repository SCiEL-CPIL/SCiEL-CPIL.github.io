def add_subproblem(sub_U, sub_V, sub_W, 
                   u_offset_i, u_offset_j, # A local (i,j) -> global (u_offset_i + i, u_offset_j + j)
                   v_offset_j, v_offset_k, # B local (j,k) -> global (v_offset_j + j, v_offset_k + k)
                   w_offset_k, w_offset_i, # C local (k,i) -> global (w_offset_k + k, w_offset_i + i)
                   n_local, m_local, p_local,
                   terms_list):
    # sub_U, sub_V, sub_W are lists of (idx, val) for the local problem
    # We need to map local indices to global indices
    pass
