def construct():
    import time
    import bisect
    deadline = time.time() + 85
    
    n = 10000
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, n + 1) if sieve[i]]
    nprimes = len(primes)
    
    # dp[i] = list of (last_gap, length) sorted by last_gap
    # For quick max query: we'll also maintain a list of prefix max lengths
    # dp_lg[i] = sorted list of last_gaps
    # dp_len[i] = corresponding lengths
    # dp_pref[i] = prefix max of lengths
    
    dp_lg = [[] for _ in range(nprimes)]
    dp_len = [[] for _ in range(nprimes)]
    dp_pref = [[] for _ in range(nprimes)]
    dp_parent = [{} for _ in range(nprimes)]  # last_gap -> (prev_idx, prev_lg)
    
    for j in range(nprimes):
        # Base case: single prime
        dp_lg[j].append(0)
        dp_len[j].append(1)
        dp_pref[j].append(1)
    
    best_len = 1
    best_end = 0
    best_gap = 0
    
    for j in range(1, nprimes):
        if time.time() > deadline:
            break
        pj = primes[j]
        new_lg = []
        new_len = {}
        
        for i in range(j):
            gap = pj - primes[i]
            # Find max length in dp[i] with last_gap <= gap
            lg_list = dp_lg[i]
            if not lg_list:
                continue
            # Binary search for rightmost index where lg <= gap
            idx = bisect.bisect_right(lg_list, gap)
            if idx == 0:
                continue
            # The best is at index idx-1 in prefix max
            best_ln = dp_pref[i][idx - 1]
            # Find which entry gives this
            # Actually, we need to know the specific (lg, ln) to track parent
            # Let's just track the best length and the corresponding lg
            # The prefix max tells us the max length, but we need the lg
            # Let me just iterate... no, that defeats the purpose.
            
            # Alternative: store for each i, a list of (lg, ln) and when we find the max,
            # we know it's at some position. Let me just store the best (ln, lg) pair.
            
            # Actually, let me simplify: just do the naive inner loop but with early termination
            # or just accept O(n^2 * avg_size) and hope it's fast enough in Python with small constants.
            
            # Let me try a different approach: for each j, build up the best by iterating i
            # and for each i, just find the best entry in dp[i] with lg <= gap.
            # To make this fast, keep dp[i] as sorted list with prefix max.
            
            if idx > 0:
                # The max length is dp_pref[i][idx-1]
                # We need to find which lg achieves this. Let's just use the last one before idx
                # that has the max. Actually, the prefix max is non-decreasing, so the max
                # in [0, idx) is dp_pref[i][idx-1]. The corresponding lg is... we need to track it.
                
                # Let me just do a linear scan of dp[i] for entries with lg <= gap,
                # but only up to the relevant portion.
                
                best_local_ln = 0
                best_local_lg = 0
                for k in range(idx):
                    if dp_len[i][k] > best_local_ln:
                        best_local_ln = dp_len[i][k]
                        best_local_lg = lg_list[k]
                
                if best_local_ln > 0:
                    new_ln = best_local_ln + 1
                    if gap not in new_len or new_ln > new_len[gap]:
                        new_len[gap] = new_ln
                        dp_parent[j][gap] = (i, best_local_lg)
                        if new_ln > best_len:
                            best_len = new_ln
                            best_end = j
                            best_gap = gap
        
        # Merge new entries into dp[j]
        for gap, ln in new_len.items():
            # Insert into dp_lg[j], dp_len[j] maintaining sorted order
            pos = bisect.bisect_right(dp_lg[j], gap)
            dp_lg[j].insert(pos, gap)
            dp_len[j].insert(pos, ln)
        
        # Rebuild prefix max
        pref = []
        cur_max = 0
        for ln in dp_len[j]:
            if ln > cur_max:
                cur_max = ln
            pref.append(cur_max)
        dp_pref[j] = pref
    
    # Reconstruct
    seq = [primes[best_end]]
    cur_idx = best_end
    cur_gap = best_gap
    while cur_idx >= 0:
        if cur_gap not in dp_parent[cur_idx]:
            break
        pi, plg = dp_parent[cur_idx][cur_gap]
        if pi == -1:
            break
        cur_idx = pi
        cur_gap = plg
        seq.append(primes[cur_idx])
    
    seq.reverse()
    
    # Validate
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        valid = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
        if not valid:
            values = []
            for p in primes:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            return values
    
    return seq
