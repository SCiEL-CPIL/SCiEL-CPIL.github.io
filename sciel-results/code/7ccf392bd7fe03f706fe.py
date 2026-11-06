def construct():
    max_val = 256
    best_seq = []
    node_count = 0
    limit = 200000
    
    def dfs(seq, sums):
        nonlocal best_seq, node_count
        
        if node_count > limit:
            return
            
        node_count += 1
        
        if len(seq) > len(best_seq):
            best_seq = list(seq)
            
        if not seq:
            last = 0
        else:
            last = seq[-1]
            
        for x in range(last + 1, max_val + 1):
            if node_count > limit:
                return
                
            if x in sums:
                continue
                
            new_sums = []
            s = x
            new_sums.append(s)
            valid = True
            
            for val in reversed(seq):
                s += val
                if s in sums:
                    valid = False
                    break
                new_sums.append(s)
                
            if not valid:
                continue
                
            sums.update(new_sums)
            seq.append(x)
            
            dfs(seq, sums)
            
            seq.pop()
            for ns in new_sums:
                sums.discard(ns)
                
    dfs([], set())
    return best_seq
