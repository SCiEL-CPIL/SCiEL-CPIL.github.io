import time
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_len = 0
    best_seq = []
    
    # State
    prefix_sums = [0]
    diffs = set()
    seq = []
    
    def dfs(last_x, last_p):
        nonlocal best_len, best_seq, start_time, time_limit
        
        current_len = len(seq)
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
            
        if time.time() - start_time > time_limit:
            return
            
        # Pruning
        remaining = 256 - last_x
        if current_len + remaining <= best_len:
            return
            
        # Try next x
        for x in range(last_x + 1, 257):
            if time.time() - start_time > time_limit:
                return
                
            new_p = last_p + x
            new_diffs = set()
            valid = True
            
            for p in prefix_sums:
                d = new_p - p
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if not valid:
                continue
                
            # Commit
            seq.append(x)
            prefix_sums.append(new_p)
            diffs.update(new_diffs)
            
            dfs(x, new_p)
            
            # Backtrack
            seq.pop()
            prefix_sums.pop()
            diffs.difference_update(new_diffs)
            
            if time.time() - start_time > time_limit:
                return

    # Start with x_1
    # We can try x_1 from 1 to 256
    # But x_1=1 is most likely to yield longest sequence
    # To be safe, we can run a few iterations or just one deep search
    # Given the time limit, a single deep search starting with small x is best.
    
    # Initialize for x_1
    # We don't know x_1 yet.
    # Let's loop x_1
    for x1 in range(1, 257):
        if time.time() - start_time > time_limit:
            break
        
        # Reset state for this x1
        prefix_sums = [0, x1]
        diffs = {x1}
        seq = [x1]
        
        # Check if x1 is valid (always is for first element)
        
        dfs(x1, x1)
        
        # If we found a very long sequence, maybe stop?
        # Or continue to see if we can find longer with different x1?
        # Unlikely, small x1 is better.
        if best_len > 40: # Arbitrary threshold
             break

    return best_seq
