import time

def construct():
    start_time = time.time()
    best_len = 0
    best_seq = []
    
    # To avoid recursion overhead and limit issues, we can use iterative deepening or just recursion with a high limit.
    # Given the problem size, recursion is fine.
    
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
        # This favors smaller prefix sums, which is generally better for fitting more elements
        for x in range(last_x + 1, 257):
            P_new = P_last + x
            
            # Check for collisions with existing sums
            # New sums are P_new - P_i for all i in prefixes
            valid = True
            new_diffs = []
            for p in prefixes:
                d = P_new - p
                if d in used_sums:
                    valid = False
                    break
                new_diffs.append(d)
                
            if valid:
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
                    
                # Optimization: If we found a solution, we don't need to try larger x in this branch 
                # if we are just looking for *a* solution? No, we want the *longest*.
                # But if we found a solution of length L, and we are at depth D, 
                # any further extension will be longer.
                # So we should continue searching for extensions.
                # The loop continues to try larger x if the current x didn't lead to a full solution?
                # Actually, the recursion handles the extension.
                # If the recursion returns, it means we exhausted that branch.
                # Then we try the next x.
                
    backtrack([], set(), [0])
    return best_seq
