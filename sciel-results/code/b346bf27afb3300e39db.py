import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin before 90s limit
    
    best_seq = []
    
    def dfs(prefixes, last_gap, used_distances):
        nonlocal best_seq
        
        # Update best if current is longer
        current_len = len(prefixes) - 1
        if current_len > len(best_seq):
            best_seq = [prefixes[i+1] - prefixes[i] for i in range(current_len)]
        
        if time.time() > deadline:
            return
            
        # Pruning: check if we can fit at least one more mark
        # Min next gap is last_gap + 1
        if prefixes[-1] + last_gap + 1 > 256:
            return
        
        max_g = 256 - prefixes[-1]
        
        # Iterate gaps from smallest to largest
        # Smaller gaps allow more marks to fit, so they are more likely to lead to longer sequences
        for g in range(last_gap + 1, max_g + 1):
            new_p = prefixes[-1] + g
            new_distances = []
            valid = True
            for p in prefixes:
                d = new_p - p
                if d in used_distances:
                    valid = False
                    break
                new_distances.append(d)
            
            if valid:
                new_used = used_distances.union(new_distances)
                dfs(prefixes + [new_p], g, new_used)
                
                if time.time() > deadline:
                    return

    # Initial state
    dfs([0], 0, set())
    
    if not best_seq:
        return [1]
        
    return best_seq
# EVOLVE-BLOCK-END
