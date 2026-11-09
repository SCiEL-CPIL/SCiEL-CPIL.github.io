        union_mid = w_mid + np.sum(np.minimum(gap_mid, w_mid))
        total_area += union_mid * (y_hi - y_lo)
    