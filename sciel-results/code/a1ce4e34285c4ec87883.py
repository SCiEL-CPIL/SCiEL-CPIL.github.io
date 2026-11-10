import bisect
import time

def construct():
    start_time = time.time()
    DEADLINE = 50  # Safety margin

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = 0
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    # dp[i] is a list of (gap, length, prev_idx, prev_gap)
    # We want to keep only the "Pareto optimal" entries?
    # No, we need to query max length for gap <= G.
    # So for each i, we maintain:
    #   sorted_gaps[i]: sorted list of gaps in dp[i]
    #   prefix_max_len[i]: prefix max of lengths corresponding to sorted_gaps
    #   prefix_max_parent[i]: parent info corresponding to the prefix max
    
    # Actually, simpler:
    # For each i, store a dict: gap -> best_length
    # And a list of (gap, best_length, prev_idx, prev_gap)
    # Then sort by gap and compute prefix max.
    
    # Let's use:
    # dp_info[i] = {
    #   'gaps': sorted list of gaps,
    #   'lens': list of lengths,
    #   'pidxs': list of prev indices,
    #   'pgaps': list of prev gaps,
    #   'prefix_lens': prefix max of lens,
    #   'prefix_pidxs': pidxs corresponding to prefix_lens,
    #   'prefix_pgaps': pgaps corresponding to prefix_lens
    # }
    
    dp_info = [None] * n
    best_len = 1
    best_i = 0
    best_gap = 0 # sentinel for first element
    
    for i in range(n):
        # Base case: sequence of length 1 ending at i
        # We can represent this as gap=0, len=1, prev=None
        # But gap=0 is not a real gap. Let's handle length 1 separately or use a sentinel.
        # If we use gap=0, it will be the smallest gap, so prefix max queries for any gap >= 0 will pick it up if no other gap is smaller?
        # Wait, gaps are positive. So gap=0 is strictly less than any real gap.
        # So if we include (0, 1, -1, -1) in the list, it works perfectly.
        
        # Collect candidates for dp[i]
        # Start with the base case
        candidates = {0: (1, -1, -1)} # gap: (length, prev_idx, prev_gap)
        
        for j in range(i):
            gap = primes[i] - primes[j]
            if dp_info[j] is None:
                continue
            
            info_j = dp_info[j]
            gaps_j = info_j['gaps']
            prefix_lens_j = info_j['prefix_lens']
            prefix_pidxs_j = info_j['prefix_pidxs']
            prefix_pgaps_j = info_j['prefix_pgaps']
            
            # Find rightmost index in gaps_j where gap_j <= gap
            idx = bisect.bisect_right(gaps_j, gap)
            if idx > 0:
                best_prev_len = prefix_lens_j[idx - 1]
                best_prev_idx = prefix_pidxs_j[idx - 1]
                best_prev_gap = prefix_pgaps_j[idx - 1]
                
                new_len = best_prev_len + 1
                # Update candidates for i with this gap
                if new_len > candidates.get(gap, (0, -1, -1))[0]:
                    candidates[gap] = (new_len, j, best_prev_gap)
        
        # Build dp_info[i] from candidates
        if not candidates:
            dp_info[i] = None
            continue
            
        items = sorted(candidates.items()) # sorted by gap
        gaps_i = [g for g, v in items]
        lens_i = [v[0] for g, v in items]
        pidxs_i = [v[1] for g, v in items]
        pgaps_i = [v[2] for g, v in items]
        
        # Compute prefix max
        prefix_lens_i = [0] * len(lens_i)
        prefix_pidxs_i = [0] * len(lens_i)
        prefix_pgaps_i = [0] * len(lens_i)
        
        prefix_lens_i[0] = lens_i[0]
        prefix_pidxs_i[0] = pidxs_i[0]
        prefix_pgaps_i[0] = pgaps_i[0]
        
        for k in range(1, len(lens_i)):
            if lens_i[k] > prefix_lens_i[k-1]:
                prefix_lens_i[k] = lens_i[k]
                prefix_pidxs_i[k] = pidxs_i[k]
                prefix_pgaps_i[k] = pgaps_i[k]
            else:
                prefix_lens_i[k] = prefix_lens_i[k-1]
                prefix_pidxs_i[k] = prefix_pidxs_i[k-1]
                prefix_pgaps_i[k] = prefix_pgaps_i[k-1]
                
        dp_info[i] = {
            'gaps': gaps_i,
            'lens': lens_i,
            'pidxs': pidxs_i,
            'pgaps': pgaps_i,
            'prefix_lens': prefix_lens_i,
            'prefix_pidxs': prefix_pidxs_i,
            'prefix_pgaps': prefix_pgaps_i
        }
        
        # Update global best
        # The best length ending at i is the max in lens_i, which is prefix_lens_i[-1]
        current_best_len = prefix_lens_i[-1]
        if current_best_len > best_len:
            best_len = current_best_len
            # We need to find which gap gave this max length to reconstruct
            # It's the one at index where prefix_lens achieved the max.
            # Since prefix_lens is non-decreasing, the last element is the max.
            # But we need the specific gap that achieved it.
            # Actually, prefix_pidxs[-1] gives the index in pidxs_i that corresponds to the max.
            # Wait, prefix_pidxs stores the *original* pidx, not the index in the list.
            # We need to reconstruct the path.
            # The "state" is (i, gap).
            # If the max length is L, it might come from any gap g where lens[g] == L.
            # Let's just pick the one that is stored in the prefix max structure.
            # The value at index -1 in prefix_* arrays corresponds to the best choice for any gap >= gaps_i[-1].
            # But we want the state that actually achieved best_len.
            # It is one of the states in candidates.
            # Let's find the gap in gaps_i that has lens == best_len.
            # Since prefix_lens is non-decreasing, the max is at the end.
            # We can just use the last element's parent info if it represents the max.
            # However, it's possible that an earlier gap had the same max length.
            # It doesn't matter which one we pick for reconstruction, as long as it's valid.
            # Let's pick the one corresponding to the last element in the prefix arrays.
            # This corresponds to the largest gap that achieved the max length (or just the one tracked).
            
            # To be safe, let's find the gap in gaps_i that maximizes lens_i.
            # Since we only care about the sequence, any valid path works.
            # Let's store best_end_i and best_end_gap.
            
            # Find index of max in lens_i
            # It's guaranteed that max(lens_i) == prefix_lens_i[-1]
            # Let's just pick the last one that updated the max?
            # Actually, the prefix arrays track the "best so far".
            # The entry at index k in prefix_* corresponds to the best state among gaps_i[0..k].
            # So prefix_*[-1] is the best state among all gaps for i.
            # So we can use:
            best_end_i = i
            # The gap associated with this best state is NOT necessarily gaps_i[-1].
            # It is the gap that was used to update the prefix max at some point.
            # But we didn't store which gap index in gaps_i was the "winner" for the prefix max.
            # We stored pidxs and pgaps, but not the gap itself.
            # We need to reconstruct the sequence.
            # The sequence ends at primes[i]. The previous prime is primes[pidx].
            # The gap between them is primes[i] - primes[pidx].
            # So we don't strictly need to store "best_gap" as a separate variable if we can derive it from pidx.
            # Wait, the state is defined by (current_prime, last_gap).
            # But last_gap = primes[i] - primes[prev].
            # So if we know prev, we know last_gap.
            # So we just need to track `prev_idx` and `prev_gap`?
            # Actually, `prev_gap` is the gap *before* the last one?
            # No, the constraint is on the gaps of the sequence.
            # Sequence: ... p_k, p_{k+1}, p_{k+2}
            # Gaps: ... g_k, g_{k+1}
            # Constraint: g_k <= g_{k+1}
            # State at p_{k+2}: last_gap = g_{k+1}.
            # Parent state at p_{k+1}: last_gap = g_k.
            # So yes, we need to know the parent's last_gap.
            
            # So `parent[i][gap] = (j, gap_j)` where gap_j is the last gap of the sequence ending at j.
            # And gap = primes[i] - primes[j].
            # And we require gap_j <= gap.
            
            # In our dp_info, for a specific state (i, gap_i), the parent is (j, gap_j).
            # We stored `pidxs` (which is j) and `pgaps` (which is gap_j).
            # So we have all we need.
            
            # To reconstruct, we start at (best_end_i, best_end_gap).
            # We need to know `best_end_gap`.
            # The `prefix_*` arrays at index -1 give us the best `pidx` and `pgap`.
            # But do they give us the `gap` for the current node?
            # No, the current node's gap is determined by `i` and `pidx`.
            # gap_current = primes[i] - primes[pidx].
            # So we can reconstruct without storing the "best gap" explicitly, just the best `pidx` and `pgap`.
            
            # However, we need to identify WHICH state (i, gap) was the best.
            # Multiple gaps for the same i might yield the same best length.
            # It doesn't matter which one we pick.
            # Let's pick the one that corresponds to the last update in the prefix max.
            # But the prefix max array just holds the values.
            # We need to map back to the gap.
            # Let's modify the storage to also store `prefix_gap_idx` (index in gaps_i).
            
            # Let's add `prefix_gap_idx` to the info.
            
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]
        
    # We need to find the specific state (i, gap) that achieved best_len.
    # Let's scan dp_info[best_end_i] to find a gap with length == best_len.
    info = dp_info[best_end_i]
    # Find any gap with lens == best_len
    # Since prefix_lens is non-decreasing, the max is at the end.
    # But the gap that achieved it might be earlier.
    # Let's just find the first gap in gaps_i where lens_i == best_len.
    best_gap = None
    for g, l in zip(info['gaps'], info['lens']):
        if l == best_len:
            best_gap = g
            break
            
    if best_gap is None:
        # Fallback
        best_gap = info['gaps'][-1]
        
    seq = [primes[best_end_i]]
    curr_i = best_end_i
    curr_gap = best_gap
    
    while True:
        info = dp_info[curr_i]
        # Find the state (curr_i, curr_gap) in info
        # We need to find its parent.
        # The parent info is stored in the arrays.
        # But the arrays are parallel to gaps_i.
        # We need to find the index k such that gaps_i[k] == curr_gap.
        # Since gaps_i is sorted, we can binary search.
        idx = bisect.bisect_left(info['gaps'], curr_gap)
        if idx < len(info['gaps']) and info['gaps'][idx] == curr_gap:
             # Found the state
             # But wait, the parent info stored in `pidxs` and `pgaps` corresponds to the state at index idx?
             # YES, `pidxs[k]` is the parent index for the state with gap `gaps[k]`.
             # So we can get parent directly.
             pidx = info['pidxs'][idx]
             pgap = info['pgaps'][idx]
             
             if pidx == -1:
                 break
                 
             seq.append(primes[pidx])
             curr_i = pidx
             curr_gap = pgap
        else:
             # Should not happen
             break
             
    seq.reverse()
    return seq
