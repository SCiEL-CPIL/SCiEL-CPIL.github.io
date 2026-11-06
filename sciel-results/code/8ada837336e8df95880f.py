"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    MAX_VAL = 256

    # Greedy baseline to establish a floor
    greedy_vals, greedy_sums = [], set()
    for x in range(1, MAX_VAL + 1):
        tails, total = [x], x
        for a in reversed(greedy_vals):
            total += a
            tails.append(total)
        if not greedy_sums.intersection(tails):
            greedy_vals.append(x)
            greedy_sums.update(tails)
    best_seq = greedy_vals
    best_len = len(best_seq)

    # Backtracking search with pruning
    # We build the sequence from left to right.
    # State: current sequence `seq`, set of all consecutive block sums `sums`
    # Optimization: Instead of recomputing all tails, we can incrementally update.
    # However, for simplicity and correctness within time limits, we use a
    # recursive DFS that tries to extend the sequence.
    
    # To speed up, we can order candidates. 
    # Usually smaller numbers are "safer" but might block larger ones later?
    # Actually, larger numbers create larger sums, potentially reducing collisions with existing small sums.
    # But we are strictly increasing.
    
    # Let's use a simple DFS.
    # seq: list of ints
    # sums: set of int
    # prefix_sums: list of prefix sums to quickly compute new block sums
    # If we append x to seq:
    # new prefix sum P = prefix_sums[-1] + x
    # new block sums are P - prefix_sums[i] for i in 0..len(seq)
    
    seq = []
    sums = set()
    prefix_sums = [0]
    
    def dfs(idx, current_seq, current_sums, current_prefixes):
        nonlocal best_seq, best_len
        nonlocal deadline_reached
        
        if time.time() > DEADLINE:
            deadline_reached = True
            return

        if deadline_reached:
            return

        # Update best if we have a longer sequence
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]

        # If we've used all possible values or reached max length, stop
        if len(current_seq) == MAX_VAL:
            return
        
        last_val = current_seq[-1] if current_seq else 0
        last_prefix = current_prefixes[-1]
        
        # Try next values greater than last_val
        # Heuristic: Try values that are likely to work.
        # Randomized restarts might help, but deterministic first.
        # We can try a range of next values.
        
        # To avoid deep recursion overhead, we can use a loop or limit depth.
        # Max depth is 256.
        
        # Candidate selection:
        # Start from last_val + 1
        # We can also try skipping some values if they fail immediately.
        
        # Optimization: Precompute potential collisions?
        # For a candidate x, new sums are {last_prefix + x - p for p in current_prefixes}
        # These must not be in current_sums.
        
        # Let's iterate candidates.
        # To improve search, we can use a heuristic ordering.
        # Often, taking the smallest valid candidate is good for packing,
        # but sometimes a larger one opens up space.
        # Let's try smallest first, but if we hit a dead end, backtrack.
        
        # To speed up, we can limit the branching factor or use a time check per node.
        
        # A simple improvement: if we are stuck, try a different strategy?
        # For now, standard DFS.
        
        # We can also use a "fail fast" check:
        # If the number of remaining values is less than (best_len - len(current_seq)), we can prune.
        # remaining = MAX_VAL - last_val
        # needed = best_len - len(current_seq)
        # if remaining < needed: return
        
        if MAX_VAL - last_val < best_len - len(current_seq):
            return

        for x in range(last_val + 1, MAX_VAL + 1):
            if time.time() > DEADLINE:
                deadline_reached = True
                return
            if deadline_reached:
                return

            # Check if x is valid
            new_prefix = last_prefix + x
            # New block sums ending at x
            valid = True
            new_sums_list = []
            for p in current_prefixes:
                s = new_prefix - p
                if s in current_sums:
                    valid = False
                    break
                new_sums_list.append(s)
            
            if not valid:
                continue

            # Recurse
            next_seq = current_seq + [x]
            next_sums = current_sums | set(new_sums_list)
            next_prefixes = current_prefixes + [new_prefix]
            
            dfs(idx + 1, next_seq, next_sums, next_prefixes)
            
            if deadline_reached:
                return

    deadline_reached = False
    dfs(0, seq, sums, prefix_sums)

    return best_seq
# EVOLVE-BLOCK-END
