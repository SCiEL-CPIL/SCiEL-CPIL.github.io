import time
import sys

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    best_sequence = []
    
    # 1. Greedy Baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    best_sequence = values[:]
    best_len = len(best_sequence)
    
    # 2. Backtracking Search
    # State:
    # prefix_sums: list of prefix sums [S0, S1, ..., Sk]
    # used_diffs: set of all differences Sj - Si for i < j
    # current_seq: list of chosen integers
    
    prefix_sums = [0]
    used_diffs = set()
    current_seq = []
    
    def dfs():
        nonlocal best_sequence, best_len, start_time, timeout, prefix_sums, used_diffs, current_seq
        
        if time.time() - start_time > timeout:
            return
        
        k = len(current_seq)
        
        # Update best if we have a longer sequence
        if k > best_len:
            best_len = k
            best_sequence = current_seq[:]
        
        last_sum = prefix_sums[-1]
        
        # Try adding next element
        # We iterate x from 1 to 256
        for x in range(1, 257):
            if time.time() - start_time > timeout:
                return
            
            new_sum = last_sum + x
            
            # Check if new differences collide with existing ones
            # New differences are new_sum - prefix_sums[i] for i in 0..k
            valid = True
            new_diffs = []
            for i in range(k + 1):
                diff = new_sum - prefix_sums[i]
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if valid:
                # Commit
                prefix_sums.append(new_sum)
                current_seq.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                # Recurse
                dfs()
                
                # Backtrack
                prefix_sums.pop()
                current_seq.pop()
                for d in new_diffs:
                    used_diffs.remove(d)

    dfs()
    
    return best_sequence
