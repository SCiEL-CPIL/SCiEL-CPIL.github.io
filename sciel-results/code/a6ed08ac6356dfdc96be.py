        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            # check no t-subset becomes uncovered
            ok = True
            while m:
                lb = m & (-m)
                if count[lb.bit_length() - 1] == 1:
                    ok = False
                    break
                m &= m - 1
