def _max_abs_ap_sum(signs):
    n = len(signs)
    if not isinstance(signs, np.ndarray):
        signs = np.asarray(signs, dtype=np.int32)
    else:
        signs = signs.astype(np.int32)
        
    max_disc = 1
    for d in range(1, n):
        # Process all residues for this step d
        # We can loop over r, but vectorize the Kadane part
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            csum = np.cumsum(seq)
            # min_prefix[j] = min(csum[0..j-1])
            # csum has length L. We need min of csum[0..j-1] for each j in 1..L
            # min_prev = [min(csum[0..0]), min(csum[0..1]), ...] is not quite right.
            # For j=0, min is -inf? No, subarray must be non-empty.
            # Standard Kadane:
            # max_ending_here = max(x, max_ending_here + x)
            # Vectorized:
            # Let csum[0] = x0, csum[1] = x0+x1...
            # We want max(csum[j] - min(csum[i]) for i<j).
            # min_csum_before_j = min(csum[0..j-1])
            # For j=0, no i<j.
            # Let's compute min_csum_prefix = np.minimum.accumulate(csum)
            # This gives min(csum[0..j]) for each j.
            # We need min(csum[0..j-1]).
            # So min_before = np.empty(L)
            # min_before[0] = -inf (or a large negative number, but csum[0] is +/-1)
            # Actually, for j=0, the only subarray ending at 0 is [0]. Sum = csum[0].
            # So we can just handle j=0 separately or use a shift.
            
            # Let's use the shift trick.
            # min_before = np.concatenate(([-10**9], np.minimum.accumulate(csum[:-1])))
            # max_before = np.concatenate(([10**9], np.maximum.accumulate(csum[:-1])))
            
            min_before = np.concatenate(([-10**9], np.minimum.accumulate(csum[:-1])))
            max_before = np.concatenate(([10**9], np.maximum.accumulate(csum[:-1])))
            
            max_sub = np.max(csum - min_before)
            min_sub = np.min(csum - max_before)
            
            if max_sub > max_disc:
                max_disc = max_sub
            if -min_sub > max_disc:
                max_disc = -min_sub
                
    return int(max_disc)
