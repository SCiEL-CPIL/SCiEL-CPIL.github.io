import time
import random

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
            if used_sums[x]:
                continue
                
            s = x
            valid = True
            for a in reversed(values):
                s += a
                if used_sums[s]:
                    valid = False
                    break
            
            if valid:
                values.append(x)
                s = x
                used_sums[x] = 1
                for a in reversed(values[:-1]):
                    s += a
                    used_sums[s] = 1
        return values, len(values)
    
    init_seq, init_len = greedy()
    best_seq = init_seq
    best_len = init_len
    
    # DFS with pruning
    def dfs(next_val, seq, used_sums, current_suffix_sums):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
            
        if len(seq) + (N - next_val + 1) <= best_len:
            return
        
        for x in range(next_val, N + 1):
            new_suffix_sums = [x]
            valid = True
            if used_sums[x]:
                valid = False
            else:
                for s in current_suffix_sums:
                    ns = x + s
                    new_suffix_sums.append(ns)
                    if used_sums[ns]:
                        valid = False
                        break
            
            if not valid:
                continue
                
            new_seq = seq + [x]
            for ns in new_suffix_sums:
                used_sums[ns] = 1
            
            if len(new_seq) > best_len:
                best_len = len(new_seq)
                best_seq = new_seq[:]
                
            dfs(x + 1, new_seq, used_sums, new_suffix_sums)
            
            for ns in new_suffix_sums:
                used_sums[ns] = 0
                
            if time.time() > deadline:
                return

    init_used = bytearray(MAX_SUM)
    init_suffix = []
    
    dfs(1, [], init_used, init_suffix)
    
    return best_seq
