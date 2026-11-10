import time

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_seq = []
    
    # 1. Greedy Baseline
    # Pick smallest valid x
    seq = []
    sums = set()
    suffix_sums = []
    last_val = 0
    
    while last_val < 256:
        found = False
        for x in range(last_val + 1, 257):
            if x in sums:
                continue
            valid = True
            new_sums = [x]
            for s in suffix_sums:
                ns = x + s
                new_sums.append(ns)
                if ns in sums:
                    valid = False
                    break
            if valid:
                seq.append(x)
                for ns in new_sums:
                    sums.add(ns)
                suffix_sums = [x] + [x + s for s in suffix_sums]
                last_val = x
                found = True
                break
        if not found:
            break
            
    if len(seq) > len(best_seq):
        best_seq = seq[:]
        
    # 2. DFS to improve
    # Use the greedy result as a starting point or just run full DFS
    # Full DFS might be too slow if it explores dead ends.
    # But the greedy result gives a strong lower bound for pruning.
    
    current_seq = []
    current_sums = set()
    suffix_sums = []
    
    def dfs_inplace():
        nonlocal best_seq
        nonlocal start_time
        nonlocal time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if len(current_seq) > len(best_seq):
            best_seq = current_seq[:]
            
        last_val = current_seq[-1] if current_seq else 0
        
        # Pruning: Max possible length
        if len(current_seq) + (256 - last_val) <= len(best_seq):
            return
            
        if last_val >= 256:
            return
            
        for x in range(last_val + 1, 257):
            if x in current_sums:
                continue
                
            valid = True
            new_sums = [x]
            for s in suffix_sums:
                ns = x + s
                new_sums.append(ns)
                if ns in current_sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Add
            current_seq.append(x)
            for ns in new_sums:
                current_sums.add(ns)
            
            old_suffix_sums = suffix_sums
            suffix_sums = [x] + [x + s for s in suffix_sums]
            
            dfs_inplace()
            
            # Undo
            current_seq.pop()
            for ns in new_sums:
                current_sums.remove(ns)
            suffix_sums = old_suffix_sums

    dfs_inplace()
    
    return best_seq
