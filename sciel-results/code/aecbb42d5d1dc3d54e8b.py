    rows = np.arange(n)[:, None]
    
    def calc_max_discrepancy_fast(signs):
        max_disc = 0
        for step in range(1, n + 1):
            max_len = (n - 1) // step + 1
            cols = np.arange(max_len)[None, :] * step
            idx = rows + cols
            mask = idx < n
            safe_idx = np.clip(idx, 0, n - 1)
            vals = signs[safe_idx]
            vals[~mask] = 0.0
            csums = np.cumsum(vals, axis=1)
            m = np.max(np.abs(csums))
            if m > max_disc:
                max_disc = m
        return max_disc
