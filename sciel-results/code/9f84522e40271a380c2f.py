def schoolbook_221(i_off, j_off, k_off):
    cols = []
    for local_j in range(2):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for local_i in range(2):
            U_dict[(local_i+i_off, local_j+j_off)] = 1.0
        V_dict[(local_j+j_off, k_off)] = 1.0
        for local_i in range(2):
            W_dict[(local_i+i_off, k_off)] = 1.0
        cols.append(add_term(U_dict, V_dict, W_dict))
    return cols
