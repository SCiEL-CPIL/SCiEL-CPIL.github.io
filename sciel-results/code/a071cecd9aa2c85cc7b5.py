                    for i_loc in range(si):
                        for j_loc in range(sj):
                            for k_loc in range(sk):
                                gi = bi_list[i_loc]
                                gj = bj_list[j_loc]
                                gk = bk_list[k_loc]
                                add_term([(gi * m + gj, 1)],
                                         [(gj * p + gk, 1)],
                                         [(gk * n + gi, 1)])
