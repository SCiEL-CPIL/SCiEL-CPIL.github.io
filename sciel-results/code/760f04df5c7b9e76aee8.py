import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIMEOUT = 55.0  # Safety margin for 60s limit
    
    def get_sums(seq):
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid_append(seq, x):
        # Check if appending x to seq creates duplicate consecutive sums
        # New sums are x, x+seq[-1], x+seq[-1]+seq[-2], ...
        # We need to check these against the existing set of sums of seq
        # It's faster to maintain the set of sums incrementally in the DFS
        
        pass # Placeholder, will implement incrementally in DFS

    # 1. Greedy initialization
    values = []
    sums = set()
    # Pre-calculate prefix sums for fast tail sum calculation?
    # Actually, just maintain the list of "tails" (sums ending at the last element)
    # If we append x, new tails are [x, x + old_tail_1, x + old_tail_2, ...]
    
    def greedy_build():
        vals = []
        sums_set = set()
        # tails[i] is sum of vals[i:]
        tails = [] 
        # Actually, let's just use the logic from the seed but cleaner
        for x in range(1, 257):
            # Calculate new tails if we append x
            # Current tails are sums of suffixes of vals
            # New tails: x, x + tail_0, x + tail_1 ...
            # Wait, the seed logic:
            # tails, total = [x], x
            # for a in reversed(values):
            #   total += a
            #   tails.append(total)
            # This calculates sums of x, x+last, x+last+second_last, etc.
            # These are the NEW sums introduced by adding x.
            # We must ensure none of these are in `sums`.
            
            new_tails = []
            total = x
            new_tails.append(total)
            for a in reversed(vals):
                total += a
                new_tails.append(total)
                
            if not sums_set.intersection(new_tails):
                vals.append(x)
                sums_set.update(new_tails)
                # Update tails for next iteration? 
                # The seed recalculates from scratch for each candidate. 
                # That is O(N^2) per step, O(N^3) total. N is small, so it's fine.
        return vals

    best_seq = greedy_build()
    best_len = len(best_seq)
    
    # 2. DFS to improve
    # State: current sequence, set of all sums, list of suffix sums of current sequence
    # suffix_sums[i] = sum(seq[i:])
    
    def dfs(seq, sums_set, suffix_sums, depth):
        nonlocal best_seq, best_len, start_time, TIMEOUT
        
        if time.time() - start_time > TIMEOUT:
            return False # Signal timeout
        
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)
            
        # Pruning: if we can't beat best_len, stop
        # Max possible extension: we can add at most (256 - seq[-1]) numbers? 
        # No, we can skip numbers. But we must be increasing.
        # So max length is len(seq) + (256 - seq[-1]) if we pick every number.
        if seq:
            max_possible = len(seq) + (256 - seq[-1])
        else:
            max_possible = 256
            
        if max_possible <= best_len:
            return False

        last_val = seq[-1] if seq else 0
        
        # Try next values
        for x in range(last_val + 1, 257):
            # Calculate new suffix sums
            # new_suffix_sums = [x] + [x + s for s in suffix_sums]
            # Wait, suffix_sums are sums of seq[i:].
            # If we append x, the new suffixes are:
            # x
            # x + seq[-1]
            # x + seq[-1] + seq[-2]
            # ...
            # So new_suffix_sums = [x + s for s in [0] + suffix_sums] ?
            # Let's check:
            # seq = [a, b]. suffix_sums = [a+b, b].
            # Append c.
            # New suffixes: c, c+b, c+b+a.
            # Formula: [c + s for s in [0, b, a+b]] -> [c, c+b, c+a+b]. Correct.
            
            # We need to check if any of these new sums are in sums_set.
            # Also, the new sums_set will be sums_set U new_suffix_sums.
            
            # Optimization: Check intersection first
            # We can generate the list of new sums and check intersection.
            new_sums_list = [x + s for s in [0] + suffix_sums]
            
            # Check for collision
            # If any s in new_sums_list is in sums_set, invalid.
            # Also, new_sums_list must be distinct internally? 
            # Since x > 0 and suffix_sums are strictly decreasing? 
            # No, suffix_sums are strictly decreasing? 
            # seq is increasing positive. 
            # s_0 = sum(seq)
            # s_1 = sum(seq[1:])
            # s_0 > s_1.
            # So new_sums_list is strictly increasing? 
            # x + 0 < x + s_1 < x + s_0. Yes.
            # So internal distinctness is guaranteed.
            
            if any(s in sums_set for s in new_sums_list):
                continue
            
            # Valid move
            new_sums_set = sums_set.union(new_sums_list)
            new_suffix_sums = new_sums_list # These are the suffix sums of the new sequence
            
            # Recurse
            seq.append(x)
            # Pass copies? No, mutate and backtrack.
            # But sums_set is a set, copying is expensive.
            # We can pass a reference and undo? 
            # Or just create a new set. N is small, set size is ~100. Copying is fast.
            
            dfs(seq, new_sums_set, new_suffix_sums, depth + 1)
            seq.pop()
            
            if time.time() - start_time > TIMEOUT:
                return True # Timeout reached

        return False

    # Start DFS
    # Initial state: empty seq, empty sums, empty suffixes
    dfs([], set(), [], 0)
    
    return best_seq
# EVOLVE-BLOCK-END
