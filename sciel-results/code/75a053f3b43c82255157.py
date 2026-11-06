import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Run Greedy to get initial best
    greedy_seq = []
    greedy_sums = set()
    greedy_suffix_sums = []
    
    for x in range(1, 257):
        valid = True
        # Check x
        if x in greedy_sums:
            valid = False
        else:
            # Check x + s
            for s in greedy_suffix_sums:
                if (x + s) in greedy_sums:
                    valid = False
                    break
        if valid:
            greedy_seq.append(x)
            greedy_sums.add(x)
            new_suffix = [x]
            for s in greedy_suffix_sums:
                new_suffix.append(x + s)
            greedy_suffix_sums = new_suffix
            greedy_sums.update(new_suffix)
            
    best_seq = greedy_seq.copy()
    best_len = len(best_seq)

    # 2. DFS to try to improve
    def dfs(seq, sums, suffix_sums):
        nonlocal best_seq
        nonlocal best_len
        
        current_len = len(seq)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
            
        # Time check
        if time.time() > deadline:
            return

        if not seq:
            last_val = 0
        else:
            last_val = seq[-1]
            
        # If we can't add any more elements, return
        if last_val == 256:
            return

        # Heuristic pruning
        if current_len + (256 - last_val) <= best_len:
            return

        for x in range(last_val + 1, 257):
            # Check time in loop
            if time.time() > deadline:
                return

            # Check collision
            valid = True
            if x in sums:
                valid = False
            else:
                for s in suffix_sums:
                    if (x + s) in sums:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Prepare next state
            new_sums = set(sums)
            new_sums.add(x)
            for s in suffix_sums:
                new_sums.add(x + s)
                
            new_suffix_sums = [x]
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
            
            seq.append(x)
            dfs(seq, new_sums, new_suffix_sums)
            seq.pop()

    # Start DFS from empty to explore all possibilities
    # Note: We could start DFS from the greedy sequence's prefix if we wanted to be smarter,
    # but starting from empty allows finding a completely different, potentially longer sequence.
    # However, starting from empty explores the same space as greedy initially.
    # To save time, we can start DFS only from states that diverge from greedy?
    # Simpler: Just run DFS from empty. The pruning will cut off branches that are worse than greedy.
    
    dfs([], set(), [])
    
    return best_seq
