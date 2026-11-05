def add_1_4_2(i_off, j_off, k_off):
    for k in range(2):
        u_indices = [(i_off) * m + (j_off + j) for j in range(4)]
        u_vals = [1] * 4
        v_indices = [(j_off + j) * p + (k_off + k) for j in range(4)]
        v_vals = [1] * 4
        w_indices = [(k_off + k) * n + i_off]
        w_vals = [1]
        add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
