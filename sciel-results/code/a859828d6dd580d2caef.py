import numpy as np

def construct():
    N = 10_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    # Create array of (sigma[n], n) for n=1..N
    # Sort by sigma[n]
    indices = np.arange(1, N + 1, dtype=np.int64)
    sigma_vals = sigma[1:N+1]
    
    # Sort indices by sigma value
    order = np.argsort(sigma_vals, kind='stable')
    sorted_indices = indices[order]
    sorted_sigma = sigma_vals[order]
    
    # Find group boundaries using diff
    diff = np.diff(sorted_sigma)
    cut_points = np.nonzero(diff)[0] + 1
    groups = np.split(sorted_indices, cut_points)
    
    best_ratio = float('inf')
    best_a, best_b = None, None
    
    for group in groups:
        if len(group) < 2:
            continue
        
        # group is already sorted by index? No, argsort was stable, so within same sigma, order is original (1..N).
        # So group is already sorted.
        # g = np.sort(group) # Just in case, but should be sorted.
        
        # Vectorized search
        # We want to minimize |a/b - 1.5|
        # This is equivalent to finding a close to 1.5*b
        targets = 1.5 * group
        idxs = np.searchsorted(group, targets)
        
        # Candidate 1: index = idxs
        idxs1 = np.clip(idxs, 0, len(group) - 1)
        a1 = group[idxs1]
        b1 = group
        mask1 = a1 != b1
        if np.any(mask1):
            err1 = np.abs(a1[mask1] / b1[mask1] - 1.5)
            min_err1 = np.min(err1)
            if min_err1 < best_ratio:
                best_ratio = min_err1
                idx_min = np.argmin(err1)
                # Need to map back to original indices in mask1
                # This is tricky. Let's just find the pair directly.
                # Re-evaluate for the specific best candidate?
                # Or just store the best a,b from this vectorized op.
                # To avoid complex indexing, we can do:
                valid_indices = np.nonzero(mask1)[0]
                best_local_idx = valid_indices[idx_min]
                best_a_candidate = int(a1[best_local_idx])
                best_b_candidate = int(b1[best_local_idx])
                # Check if this is actually better than current best (it is, by min_err1)
                # But we need to be careful if min_err1 is 0? Task says 0 is invalid.
                if best_ratio > 0:
                    best_a, best_b = best_a_candidate, best_b_candidate

        # Candidate 2: index = idxs - 1
        idxs2 = np.clip(idxs - 1, 0, len(group) - 1)
        a2 = group[idxs2]
        b2 = group
        mask2 = a2 != b2
        if np.any(mask2):
            err2 = np.abs(a2[mask2] / b2[mask2] - 1.5)
            min_err2 = np.min(err2)
            if min_err2 < best_ratio:
                best_ratio = min_err2
                valid_indices = np.nonzero(mask2)[0]
                idx_min = np.argmin(err2)
                best_local_idx = valid_indices[idx_min]
                best_a_candidate = int(a2[best_local_idx])
                best_b_candidate = int(b2[best_local_idx])
                if best_ratio > 0:
                    best_a, best_b = best_a_candidate, best_b_candidate

    return [best_a, best_b]
