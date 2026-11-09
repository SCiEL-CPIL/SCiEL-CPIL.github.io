import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Keep a safety margin below the 90s hard limit
    
    best_seq = []
    
    def dfs(seq, last_val, p_sums, diffs):
        nonlocal best_seq
        
        # Update best if current sequence is longer
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        # Time check
        if time.time() - start_time > time_limit:
            return
            
        # Pruning: Even if we pick every remaining number, can we beat the best?
        # Max possible length from here is len(seq) + (256 - last_val)
        if len(seq) + (256 - last_val) <= len(best_seq):
            return

        # Try adding the next value
        for next_val in range(last_val + 1, 257):
            new_p = p_sums[-1] + next_val
            
            # Check if new prefix sum creates a duplicate difference
            # We need to check |new_p - p| for all existing p in p_sums
            # If any such difference is already in 'diffs', it's a collision
            valid = True
            new_diffs = []
            for p in p_sums:
                d = abs(new_p - p)
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # If valid, recurse
            # We copy the lists/sets to maintain state for backtracking
            dfs(
                seq + [next_val], 
                next_val, 
                p_sums + [new_p], 
                diffs.union(new_diffs)
            )
            
            # Check time after each step to avoid long loops
            if time.time() - start_time > time_limit:
                return

    # Initial call
    # seq: []
    # last_val: 0 (so next_val starts from 1)
    # p_sums: [0] (P_0 = 0)
    # diffs: set() (no differences yet)
    dfs([], 0, [0], set())
    
    return best_seq
