        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    
                    if si == 2 and sj == 4 and sk == 4:
                        # ...
                        total_rank += r # r=7 per sub-block, 4 sub-blocks -> 28
                    elif si == 2 and sj == 4 and sk == 2:
                        # ...
                        total_rank += r # r=7 per sub-block, 2 sub-blocks -> 14
                    elif si == 2 and sj == 1:
                        r = si * sj * sk
                        total_rank += r
                    else:
                        r = si * sj * sk
                        total_rank += r
