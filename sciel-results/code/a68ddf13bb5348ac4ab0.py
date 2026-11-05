<<<<<<< SEARCH
    # --- Also run a couple of pure-greedy random orderings for diversity ---
    for it in range(4):
        if stopped():
            break
        full = list(range(1, cap + 1))
        rng.shuffle(full)
        # simple fast greedy with pre-allocated array
        chosen = []
        cnp = np.empty(cap, dtype=np.int64)
        n = 0
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        for x in full:
            if n == 0:
                ns = np.array([2 * x], dtype=np.int64)
            else:
                ns = np.concatenate([cnp[:n] + x, np.array([2 * x], dtype=np.int64)])
            if (counts[ns] + 1).max() <= 2:
                chosen.append(x)
                cnp[n] = x
                n += 1
                counts[ns] += 1
        consider(chosen)
=======
    # --- Also run a couple of pure-greedy random orderings for diversity ---
    # Use batched vectorized greedy for speed
    for it in range(6):
        if stopped():
            break
        full = list(range(1, cap + 1))
        rng.shuffle(full)
        full_arr = np.asarray(full, dtype=np.int64)
        m = len(full_arr)
        chosen = []
        cnp = np.empty(m, dtype=np.int64)
        n = 0
        counts = np.zeros(2 * cap + 1, dtype=np.int32)
        remaining = np.arange(m)
        
        while remaining.size > 0:
            rem_vals = full_arr[remaining]
            if n == 0:
                new_sums = 2 * rem_vals[:, None]
                gathered = counts[new_sums] + 1
                cand_max = gathered.max(axis=1)
            else:
                c_cur = cnp[:n]
                cross = rem_vals[:, None] + c_cur[None, :]
                doubles = 2 * rem_vals[:, None]
                new_sums = np.concatenate([cross, doubles], axis=1)
                gathered = counts[new_sums] + 1
                cand_max = gathered.max(axis=1)
            
            feasible = np.where(cand_max <= 2)[0]
            if feasible.size == 0:
                break
            
            tight = gathered[feasible].mean(axis=1)
            bi = int(feasible[int(np.argmin(tight))])
            x = int(rem_vals[bi])
            
            if n == 0:
                ns = np.array([2 * x], dtype=np.int64)
            else:
                ns = np.concatenate([cnp[:n] + x, np.array([2 * x], dtype=np.int64)])
            
            chosen.append(x)
            cnp[n] = x
            n += 1
            counts[ns] += 1
            remaining = remaining[remaining != bi]
        
        consider(chosen)
>>>>>>> REPLACE
