import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    best_seq = []
    best_len = 0
    
    def dfs(current_seq, current_sums, last_val):
        nonlocal best_seq, best_len
        
        if time.time() > start_time + DEADLINE:
            return
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
            
        # Pruning: If even taking all remaining numbers can't beat best_len
        # Max elements we can add is (256 - last_val)
        # But we need distinct sums, so it's less.
        # This prune is weak but safe.
        if len(current_seq) + (256 - last_val) <= best_len:
            return

        for x in range(last_val + 1, 257):
            if time.time() > start_time + DEADLINE:
                return
            
            # Calculate new sums
            new_sums = []
            total = x
            new_sums.append(total)
            for i in range(len(current_seq) - 1, -1, -1):
                total += current_seq[i]
                new_sums.append(total)
                
            # Check collision
            valid = True
            for s in new_sums:
                if s in current_sums:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Recurse
            dfs(current_seq + [x], current_sums.union(new_sums), x)
            
    dfs([], set(), 0)
    
    return best_seq
