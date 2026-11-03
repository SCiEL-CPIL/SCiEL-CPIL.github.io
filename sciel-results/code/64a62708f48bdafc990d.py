    diff_a = a_sum[i] - a_sum[j]
    diff_D = D[j] - D[i]
    S = np.dot(diff_a, diff_D)
    # ... scalar ops
    