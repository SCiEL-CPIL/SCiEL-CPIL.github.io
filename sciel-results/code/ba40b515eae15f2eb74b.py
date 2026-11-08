for i_inner in range(2):
    for j_inner in range(2):
        ci = v_in[l_inner][i_inner * 2 + j_inner]
        for b in range(4):
            co = v_out[k_outer][b]
            br, bc = b // 2, b % 2
            jg = 2 * br + i_inner
            kg = 2 * bc + j_inner
            v_vec[jg * 7 + kg] += ci * co
