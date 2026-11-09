                    for ap_id in ap_index[i]:
                        a, d = all_aps[ap_id]
                        idxs = [a + j * d for j in range(4)]
                        others = []
                        for idx in idxs:
                            if idx == i:
                                continue
                            others.append(colors[idx])
                        is_v_new = len({c_rand, others[0], others[1], others[2]}) < 3
