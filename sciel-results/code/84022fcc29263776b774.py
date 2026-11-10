        c_dis = c_disrupts[c]
        not_c_dis = np.bitwise_not(c_dis)
        blocked = (valid_dirs & not_c_dis) != 0
        if not np.any(blocked):
            S_bits[c // 64] |= (1 << (c % 64))
            valid_dirs &= not_c_dis
