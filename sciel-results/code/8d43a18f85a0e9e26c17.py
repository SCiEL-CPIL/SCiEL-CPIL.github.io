    def hungarian_init(a, b):
        # H_ij = sum_k A_ik B_jk
        # Minimize sum_i H_i, p(i)
        H = a @ b.T
        # linear_sum_assignment minimizes sum cost[i, p[i]]
        # We want to minimize QAP.
        # If H is such that low H corresponds to good assignment, we minimize H.
        # H_ij is "cost" of assigning i to j in this linear approx.
        row_ind, col_ind = linear_sum_assignment(H)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
