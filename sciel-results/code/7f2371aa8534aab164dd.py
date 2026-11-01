<<<<<<< SEARCH
def _greedy_bitmask(spheres, full_mask, time_budget, start_time, random_order=None):
    """Greedy set-cover using bitmask operations."""
    if random_order is not None:
        candidates = list(range(len(spheres)))
        random.shuffle(candidates)
    else:
        candidates = list(range(len(spheres)))
    
    covered = 0
    chosen = []
    while covered != full_mask:
        if time.time() - start_time > time_budget:
            break
        best_gain = -1
        best_idx = -1
        for idx in candidates:
            new_covered = spheres[idx] & ~covered
            gain = new_covered.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        covered |= spheres[best_idx]
    return chosen
=======
def _greedy_bitmask(spheres, full_mask, time_budget, start_time, random_order=None):
    """Greedy set-cover using numpy boolean array for coverage."""
    total = spheres.shape[0]
    
    if random_order is not None:
        candidates = list(range(total))
        random.shuffle(candidates)
    else:
        candidates = list(range(total))
    
    covered = np.zeros(total, dtype=bool)
    chosen = []
    covered_count = 0
    
    while covered_count < total:
        if time.time() - start_time > time_budget:
            break
        best_gain = -1
        best_idx = -1
        # Vectorized: compute gain for all candidates at once
        # gain[i] = number of words in sphere[i] not yet covered
        # spheres[candidates] is a 2D array [len(candidates), total]
        # We want: for each row, count of (row & ~covered)
        # This is expensive for large candidate sets. Use sampling for speed.
        if len(candidates) > 200:
            # Sample a subset for speed
            sample = random.sample(candidates, 200)
            gains = spheres[sample][:, ~covered].sum(axis=1)
            best_local = gains.argmax()
            best_gain = gains[best_local]
            best_idx = sample[best_local]
        else:
            gains = spheres[candidates][:, ~covered].sum(axis=1)
            best_local = gains.argmax()
            best_gain = gains[best_local]
            best_idx = candidates[best_local]
        
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        new_cover = spheres[best_idx]
        newly = new_cover & ~covered
        covered_count += newly.sum()
        covered |= new_cover
    return chosen
>>>>>>> REPLACE

