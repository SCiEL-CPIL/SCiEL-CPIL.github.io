    def construct():
        start_time = time.time()
        deadline = start_time + 55.0  # safety margin before 60s limit
        
        # Use a smaller N to ensure the sieve completes in time
        # 2.5M is a safe estimate for Python/NumPy sieve on typical hardware
        N = 2_500_000
        
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve: for i from 1 to N, add i to sigma[j] for all j multiple of i
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        if time.time() > deadline - 5:
            return [11, 6]
            
        sig_vals = sigma[1:]
        indices = np.arange(1, N + 1)
        
        order = np.argsort(sig_vals, kind='mergesort')
        sorted_sig = sig_vals[order]
        sorted_idx = indices[order]
        
        changes = np.where(np.diff(sorted_sig) != 0)[0] + 1
        boundaries = np.concatenate(([0], changes, [len(sorted_sig)]))
        
        best_diff = float('inf')
        best_a = 11
        best_b = 6
        
        for k in range(len(boundaries) - 1):
            if time.time() > deadline - 2:
                break
                
            lo = boundaries[k]
            hi = boundaries[k + 1]
            if hi - lo < 2:
                continue
                
            group_idx = sorted_idx[lo:hi]
            n_group = len(group_idx)
            
            # Vectorized search for best pair in this group
            # We want a/b close to 1.5 => 2a close to 3b
            # For each b, find a in group_idx closest to 1.5*b
            targets = 1.5 * group_idx
            pos = np.searchsorted(group_idx, targets)
            
            # Candidate 1: pos (clamped to max index)
            pos1 = np.minimum(pos, n_group - 1)
            a1 = group_idx[pos1]
            err1 = np.abs(2 * a1 - 3 * group_idx).astype(np.float64) / (2.0 * group_idx)
            
            # Candidate 2: pos - 1 (clamped to min index)
            pos2 = np.maximum(pos - 1, 0)
            a2 = group_idx[pos2]
            err2 = np.abs(2 * a2 - 3 * group_idx).astype(np.float64) / (2.0 * group_idx)
            
            # Filter out zero errors (invalid)
            valid1 = err1 > 0
            valid2 = err2 > 0
            
            if not np.any(valid1) and not np.any(valid2):
                continue
                
            min_err1 = np.min(err1[valid1]) if np.any(valid1) else float('inf')
            min_err2 = np.min(err2[valid2]) if np.any(valid2) else float('inf')
            
            group_min_err = min(min_err1, min_err2)
            
            if group_min_err < best_diff:
                if min_err1 <= min_err2:
                    # Find the index of the minimum error in the valid subset
                    valid_indices = np.where(valid1)[0]
                    min_idx_in_valid = np.argmin(err1[valid1])
                    idx_in_group = valid_indices[min_idx_in_valid]
                    b_val = group_idx[idx_in_group]
                    a_val = a1[idx_in_group]
                else:
                    valid_indices = np.where(valid2)[0]
                    min_idx_in_valid = np.argmin(err2[valid2])
                    idx_in_group = valid_indices[min_idx_in_valid]
                    b_val = group_idx[idx_in_group]
                    a_val = a2[idx_in_group]
                
                best_diff = group_min_err
                best_a = int(a_val)
                best_b = int(b_val)
                
        return [best_a, best_b]
    