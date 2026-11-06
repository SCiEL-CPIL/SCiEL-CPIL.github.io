        # 3-sum check
        nonzero_keys = [k for k in memo_keys if k != 0]
        for i in range(len(nonzero_keys)):
            m1 = nonzero_keys[i]
            rem1 = mask ^ m1
            for j in range(len(nonzero_keys)):
                if i == j: continue
                m2 = nonzero_keys[j]
                m3 = rem1 ^ m2
                if m3 in memo:
                    w1 = memo[m1]
                    w2 = memo[m2]
                    w3 = memo[m3]
                    w_temp = add_gate(w1, w2)
                    w_new = add_gate(w_temp, w3)
                    memo[mask] = w_new
                    return w_new
