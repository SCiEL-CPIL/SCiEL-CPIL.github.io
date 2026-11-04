def schoolbook_1x2xp(i_idx, j_off, k_range):
    cols = []
    for local_j in range(2):
        for k in k_range:
            U_dict = {(i_idx, j_off + local_j): 1}
            V_dict = {(j_off + local_j, k): 1}
            W_dict = {(i_idx, k): 1}
            cols.append(add_term(U_dict, V_dict, W_dict))
    return cols
