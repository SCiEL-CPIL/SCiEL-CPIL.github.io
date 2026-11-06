import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    MAX_SUM = 20000
    used_sums = bytearray(MAX_SUM + 1)
    
    def is_valid(x, suffix_sums):
        if used_sums[x]:
            return False
        for s in suffix_sums:
            if used_sums[x + s]:
                return False
        return True

    def add_sums(x, suffix_sums):
        used_sums[x] = 1
        new_suffix = [x]
        for s in suffix_sums:
            ns = x + s
            new_suffix.append(ns)
            used_sums[ns] = 1
        return new_suffix

    # Greedy
    greedy_seq = []
    suffix_sums = []
    
    for x in range(1, 257):
        if is_valid(x, suffix_sums):
            greedy_seq.append(x)
            suffix_sums = add_sums(x, suffix_sums)
            
    best_seq = greedy_seq.copy()
    best_len = len(best_seq)
    
    # Reset used_sums for DFS
    # We need to reset it because DFS will modify it.
    # Since we are doing DFS from scratch, we need a clean state.
    # But we can't easily reset a global bytearray in a nested function without nonlocal or passing it.
    # Let's pass the mask as an argument or use a class/encapsulation.
    # Passing a bytearray and modifying it in-place requires backtracking (setting back to 0).
    
    def dfs(last_val, suffix_sums, seq):
        nonlocal best_seq
        nonlocal best_len
        
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
            
        if time.time() > deadline:
            return
            
        if last_val == 256:
            return
            
        if current_len + (256 - last_val) <= best_len:
            return

        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            if is_valid(x, suffix_sums):
                # Apply change
                new_suffix = add_sums(x, suffix_sums)
                seq.append(x)
                
                dfs(x, new_suffix, seq)
                
                # Backtrack: remove sums
                used_sums[x] = 0
                for s in new_suffix:
                    if s != x: # x is already cleared? No, x is in new_suffix[0]
                         pass 
                # Wait, add_sums sets used_sums[x] and used_sums[x+s].
                # We need to clear exactly what was set.
                used_sums[x] = 0
                for s in suffix_sums: # Original suffix sums
                    used_sums[x + s] = 0
                # Note: new_suffix contains x and x+s.
                # We cleared x and x+s.
                
                seq.pop()

    # We need to reset used_sums before starting DFS
    for i in range(MAX_SUM + 1):
        used_sums[i] = 0
        
    dfs(0, [], [])
    
    return best_seq
