def evaluate(x):
    max_disc = 0
    n = len(x)
    for d in range(1, n + 1):
        # Max length for this step
        max_len = (n - 1) // d + 1
        if max_len <= max_disc:
            # Even if all signs are same, sum is max_len.
            # If max_len <= current max_disc, this d cannot increase the max.
            # But it might be the one that *defines* the max if others are smaller?
            # No, we are looking for the MAX over all APs.
            # If we already found an AP with sum max_disc, and this d's max possible sum is <= max_disc,
            # it cannot be the *new* maximum if the new maximum is > max_disc.
            # But we need to know if the new max is *lower* than the old one?
            # Wait. We are evaluating a *new* sequence.
            # We want the max discrepancy of *this* sequence.
            # If we skip d, we might miss a discrepancy that is, say, 5, while we have 10 from other d's.
            # That's fine, 10 is the max.
            # But if we skip d, and the true max is 5 (from this d), and we missed it, we return 10 (from other d's).
            # That is correct if 10 is indeed present in other d's.
            # The risk is: We skip d where max_possible=5.
            # We compute other d's and find max=4.
            # We return 4.
            # But the true max is 5 (from the skipped d).
            # So we underestimate the discrepancy.
            # This leads to accepting bad moves.
            # So we CANNOT skip based on max_possible < current_max_found_so_far.
            # We MUST compute all.
            pass
        
        # Construct matrix for this d
        # We can use slicing
        # x is 1D.
        # We want x[r::d] for r in 0..d-1
        # This creates a list of arrays of varying lengths.
        # Padding to max_len is required for 2D array.
        
        # Efficient way:
        # Create a 2D array of shape (d, max_len) filled with 0
        mat = np.zeros((d, max_len), dtype=np.int8)
        for r in range(d):
            # Slice x
            vals = x[r::d]
            L = len(vals)
            mat[r, :L] = vals
        
        # Cumsum along axis 1
        prefix = np.cumsum(mat, axis=1)
        # Max and min along axis 1
        max_p = np.max(prefix, axis=1)
        min_p = np.min(prefix, axis=1)
        diffs = max_p - min_p
        cur_max = np.max(diffs)
        
        if cur_max > max_disc:
            max_disc = cur_max
            
    return max_disc
