def construct():
    import sys
    sys.setrecursionlimit(10000)
    
    best_len = 0
    best_seq = []
    memo = {}
    nodes_visited = 0
    MAX_NODES = 500000

    def baseline_greedy():
        values = []
        products = set()
        remaining = set(range(2, 129))
        
        for _ in range(127):
            best_next = None
            
            for x in sorted(remaining):
                if values and x <= values[-1]:
                    continue
                
                suffs = [1]
                for a in reversed(values):
                    suffs.append(suffs[-1] * a)
                
                new_prods = [x * s for s in suffs]
                new_prods_set = set(new_prods)
                
                if products.intersection(new_prods_set):
                    continue
                
                best_next = x
                break
                
            if best_next is None:
                break
            
            values.append(best_next)
            suffs = [1]
            for a in reversed(values):
                suffs.append(suffs[-1] * a)
            new_prods = [best_next * s for s in suffs]
            products.update(new_prods)
            remaining.discard(best_next)
            
        return values

    best_seq = baseline_greedy()
    best_len = len(best_seq)

    def dfs_path(last_val, suffs, prods, current_len, current_seq):
        nonlocal best_len, best_seq, nodes_visited
        
        nodes_visited += 1
        if nodes_visited > MAX_NODES:
            return
            
        if current_len + (128 - last_val) <= best_len:
            return
        
        state_key = (last_val, suffs, prods)
        
        if state_key in memo:
            if current_len <= memo[state_key]:
                return
        
        if state_key not in memo or current_len > memo[state_key]:
            memo[state_key] = current_len
            
        if current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]
            
        for next_val in range(last_val + 1, 129):
            new_prods_list = [next_val * s for s in suffs]
            new_prods_set = frozenset(new_prods_list)
            
            if prods.intersection(new_prods_set):
                continue
            
            new_prods = prods.union(new_prods_set)
            new_suffs = (1,) + tuple(new_prods_list)
            
            dfs_path(next_val, new_suffs, new_prods, current_len + 1, current_seq + [next_val])

    dfs_path(1, (1,), frozenset(), 0, [])
    
    return best_seq
