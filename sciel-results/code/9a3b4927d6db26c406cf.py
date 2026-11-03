for c_block_name, l1_combos in level1_c_combo.items():
    w_total = 0.0
    for l1_mult_idx, l1_coeff in l1_combos:
        if l1_mult_idx == l1_idx:
            if c_block_name in l2_c_contribs:
                w_total += l1_coeff * l2_c_contribs[c_block_name]
