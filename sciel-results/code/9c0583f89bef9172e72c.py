import time

def construct():
    start_time = time.time()
    best_len = 0
    best_seq = []
    
    def backtrack(seq, used_sums, prefixes):
        nonlocal best_len, best_seq
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Time guard
        if time.time() - start_time > 55:
            return
            
        last_x = seq[-1] if seq else 0
        P_last = prefixes[-1]
        
        # Try candidates in ascending order
        for x in range(last_x + 1, 257):
            P_new = P_last + x
            
            # Check for collisions with existing sums
            valid = True
            new_diffs = []
            for p in prefixes:
                d = P_new - p
                if d in used_sums:
                    valid = False
                    break
                new_diffs.append(d)
                
            if valid:
                # Forward checking: verify that at least one valid extension exists
                # If no extension is possible, this branch is a dead end
                can_extend = False
                for y in range(x + 1, 257):
                    P_next = P_new + y
                    # Check if adding y would create a collision
                    extend_valid = True
                    for p in prefixes + [P_new]:
                        d = P_next - p
                        if d in used_sums or d in new_diffs:
                            extend_valid = False
                            break
                    if extend_valid:
                        can_extend = True
                        break
                
                # If we can't extend, only pursue this branch if it improves best_len
                if not can_extend and len(seq) + 1 <= best_len:
                    continue
                
                # Commit
                seq.append(x)
                prefixes.append(P_new)
                used_sums.update(new_diffs)
                
                # Recurse
                backtrack(seq, used_sums, prefixes)
                
                # Undo
                seq.pop()
                prefixes.pop()
                for d in new_diffs:
                    used_sums.discard(d)
                
    backtrack([], set(), [0])
    return best_seq
