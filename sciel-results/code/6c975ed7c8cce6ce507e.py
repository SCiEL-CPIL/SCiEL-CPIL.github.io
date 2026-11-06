import time

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 55  # 55s safety margin
    
    # Max possible sum is 256 * 256 = 65536
    MAX_SUM = 65536 + 1
    
    # Global best
    best_seq = []
    best_len = 0
    
    # Greedy initialization to get a good starting point
    def greedy():
        values = []
        used_sums = bytearray(MAX_SUM)
        for x in range(1, N + 1):
            # Calculate suffix sums if we append x
            # Suffix sums are: x, x + last, x + last + second_last, ...
            # We need to check if any of these are already used.
            # Also, we need to check if x itself is used (it's the first suffix sum).
            
            # Efficient check:
            # The new suffix sums are x + suffix_sum_i for all existing suffix sums, plus x.
            # Let's just compute them.
            new_sums = []
            total = 0
            # Iterate backwards through values to compute suffix sums ending at x
            # Actually, suffix sums of the new sequence are:
            # x, x + v_last, x + v_last + v_second_last, ...
            
            # Let's precompute the suffix sums of the current 'values'
            # But 'values' changes.
            
            # Simpler:
            # Check if x is in used_sums.
            if used_sums[x]:
                continue
                
            # Check other suffix sums
            s = x
            valid = True
            for a in reversed(values):
                s += a
                if used_sums[s]:
                    valid = False
                    break
            
            if valid:
                # Commit
                values.append(x)
                # Update used_sums
                s = x
                used_sums[x] = 1
                for a in reversed(values[:-1]): # values[:-1] is the list before x
                    s += a
                    used_sums[s] = 1
        return values, len(values)
    
    init_seq, init_len = greedy()
    best_seq = init_seq
    best_len = init_len
    
    # DFS with pruning
    # state:
    # seq: list of current values
    # used_sums: bytearray of used sums
    # current_suffix_sums: list of suffix sums of the current seq
    # next_val: minimum value to try next (must be > seq[-1])
    
    def dfs(next_val, seq, used_sums, current_suffix_sums):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: max possible length
        # We can add at most (N - next_val + 1) more elements
        if len(seq) + (N - next_val + 1) <= best_len:
            return
        
        # Try all candidates from next_val to N
        for x in range(next_val, N + 1):
            # Calculate new suffix sums
            # New suffix sums are: x, x + current_suffix_sums[-1], x + current_suffix_sums[-2], ...
            # Wait, current_suffix_sums contains sums of suffixes of 'seq'.
            # If seq = [a, b, c], suffix sums are [c, b+c, a+b+c].
            # If we append x, new suffix sums are [x, x+c, x+b+c, x+a+b+c].
            
            new_suffix_sums = [x]
            valid = True
            if used_sums[x]:
                valid = False
            else:
                # Check the rest
                # The new sums are x + s for s in current_suffix_sums
                # Note: current_suffix_sums is in order [last, last+prev, ...]
                # So x + current_suffix_sums[i]
                for s in current_suffix_sums:
                    ns = x + s
                    new_suffix_sums.append(ns)
                    if used_sums[ns]:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Commit
            new_seq = seq + [x]
            # Update used_sums
            for ns in new_suffix_sums:
                used_sums[ns] = 1
            
            # Update best
            if len(new_seq) > best_len:
                best_len = len(new_seq)
                best_seq = new_seq[:]
                
            # Recurse
            dfs(x + 1, new_seq, used_sums, new_suffix_sums)
            
            # Backtrack
            for ns in new_suffix_sums:
                used_sums[ns] = 0
                
            # Time check
            if time.time() > deadline:
                return

    # Initialize DFS from the greedy result?
    # Or start from scratch?
    # Starting from scratch allows finding better solutions that the greedy might miss.
    # However, greedy gives a good lower bound for pruning.
    # Let's start DFS from empty sequence, but use best_len from greedy for pruning.
    
    # We need to initialize used_sums and current_suffix_sums for the empty sequence.
    init_used = bytearray(MAX_SUM)
    init_suffix = []
    
    dfs(1, [], init_used, init_suffix)
    
    return best_seq
