import sys
import time

sys.setrecursionlimit(10000)

def construct():
    # 1. Generate a baseline using the greedy algorithm
    # Using bitset for sums for speed
    values = []
    # D represents suffix sums including 0. Initially [0]
    D = [0]
    diff_mask = 0
    
    for x in range(1, 257):
        # Calculate new suffix sums
        new_D = [x + d for d in D]
        
        # Check for conflicts
        valid = True
        for nd in new_D:
            if (diff_mask >> nd) & 1:
                valid = False
                break
        
        if valid:
            values.append(x)
            # Update mask
            for nd in new_D:
                diff_mask |= (1 << nd)
            # Update D
            D = new_D
                
    best_seq = values[:]
    best_len = len(values)
    
    # 2. DFS to find longer sequences
    start_time = time.time()
    time_limit = 55.0
    nodes = 0
    node_limit = 5000000 # Higher limit due to efficiency

    # We use in-place backtracking
    seq = []
    # D will be modified in place (pointer swap)
    current_D = [0]
    current_mask = 0

    def dfs(next_min_val):
        nonlocal nodes, best_seq, best_len, current_D, current_mask
        
        # Time check every 1000 nodes
        if nodes % 1000 == 0:
            if time.time() - start_time > time_limit:
                return
        if nodes > node_limit:
            return
        nodes += 1
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        if next_min_val > 256:
            return
            
        for x in range(next_min_val, 257):
            # Pruning: Max possible length if we pick x and all subsequent valid numbers
            # We can pick at most (256 - x) more numbers after x.
            # So max len is len(seq) + 1 + (256 - x)
            if len(seq) + 1 + (256 - x) <= best_len:
                break
            
            # Calculate new_D and check conflicts
            valid = True
            new_D = []
            new_mask_add = 0
            
            for d in current_D:
                nd = x + d
                new_D.append(nd)
                if (current_mask >> nd) & 1:
                    valid = False
                    break
                new_mask_add |= (1 << nd)
                
            if not valid:
                continue
            
            # Commit
            seq.append(x)
            old_mask = current_mask
            current_mask |= new_mask_add
            old_D = current_D
            current_D = new_D
            
            dfs(x + 1)
            
            # Backtrack
            seq.pop()
            current_mask = old_mask
            current_D = old_D
            
            if nodes % 1000 == 0 and time.time() - start_time > time_limit:
                return

    dfs(1)
    
    return best_seq
