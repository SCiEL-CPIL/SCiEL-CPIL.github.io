import time

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit
    
    best_seq = []
    best_len = 0
    
    # Greedy initialization to get a strong starting point
    def greedy_init():
        vals = []
        sums_set = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals

    initial_seq = greedy_init()
    if initial_seq:
        best_seq = initial_seq.copy()
        best_len = len(initial_seq)
    
    # Maintain suffix sums for efficient new-sum computation
    # suffix_sums[i] = sum of seq[i:]
    # When appending x, new suffix sums are: x, x+suffix_sums[-1], x+suffix_sums[-2], ...
    
    def compute_suffix_sums(seq):
        n = len(seq)
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + seq[i]
        return suffix  # suffix[i] = sum of seq[i:], suffix[n] = 0
    
    # Compute the sums mask from a sequence
    def calc_sums_mask(seq):
        mask = 0
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                mask |= (1 << s)
        return mask
    
    # DFS with suffix sums maintained
    def dfs(seq, sums_mask, last_val, suffix_sums):
        nonlocal best_seq, best_len, start_time, time_limit
        
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
            
        # Time check
        if time.time() - start_time > time_limit:
            return
        
        # Pruning: if even using all remaining values can't beat best, prune
        # Max additional elements we can add is at most 256 - last_val
        # But that's a very weak bound. Let's use a tighter one.
        # The minimum sum of k additional elements starting from last_val+1 is:
        # (last_val+1) + (last_val+2) + ... + (last_val+k)
        # = k * (2*last_val + k + 1) / 2
        # But this doesn't directly help with distinct sums.
        
        # Try to extend: candidates from last_val + 1 to 256
        # To speed up, try smaller candidates first (they keep sums lower)
        
        for x in range(last_val + 1, 257):
            # Check time periodically
            if (x - last_val) % 32 == 0 and time.time() - start_time > time_limit:
                return
            
            # Compute new sums mask efficiently using suffix_sums
            # New suffix sums when appending x:
            # new_suffix[0] = x + suffix_sums[-1] (which is 0, so just x)
            # Actually, suffix_sums has length len(seq)+1, with suffix_sums[i] = sum(seq[i:])
            # When we append x, the new sequence is seq + [x]
            # New suffix sums: 
            #   For the new element x: just x
            #   For seq[-1] + x: suffix_sums[-1] + x = 0 + x = x (wait, suffix_sums[-1] should be 0)
            # Let me re-index: suffix_sums[i] for i in 0..n where n=len(seq)
            # suffix_sums[n] = 0
            # suffix_sums[n-1] = seq[n-1]
            # suffix_sums[n-2] = seq[n-2] + seq[n-1]
            # ...
            # When appending x, new suffix sums for the new sequence of length n+1:
            # new_suffix[n] = x
            # new_suffix[n-1] = seq[n-1] + x = suffix_sums[n-1] + x
            # new_suffix[n-2] = seq[n-2] + seq[n-1] + x = suffix_sums[n-2] + x
            # ...
            # new_suffix[0] = sum(seq) + x = suffix_sums[0] + x
            
            new_sums_mask = 0
            total = x
            new_sums_mask |= (1 << total)
            for i in range(len(seq) - 1, -1, -1):
                total += seq[i]
                new_sums_mask |= (1 << total)
            
            if sums_mask & new_sums_mask:
                continue
            
            # Build new suffix sums
            new_suffix = [0] * (len(seq) + 2)
            new_suffix[len(seq) + 1] = 0
            new_suffix[len(seq)] = x
            for i in range(len(seq) - 1, -1, -1):
                new_suffix[i] = suffix_sums[i] + x
            
            dfs(seq + [x], sums_mask | new_sums_mask, x, new_suffix)
            
            # Early termination if time is up
            if time.time() - start_time > time_limit:
                return

    # Start DFS from empty
    dfs([], 0, 0, [0])
    
    return best_seq
