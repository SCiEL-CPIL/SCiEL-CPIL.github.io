import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for the evaluator's overhead
    
    best_seq = []
    best_len = 0
    
    # State:
    # seq: list of chosen numbers
    # sums: set of all consecutive block sums so far
    # last_val: last number in seq
    # suffix_sums: list of sums of consecutive blocks ending at the last element of seq
    #             e.g., if seq=[2,3], suffix_sums=[3, 5]
    
    def dfs(seq, sums, last_val, suffix_sums):
        nonlocal best_seq, best_len
        
        # Check time
        if time.time() - start_time > time_limit:
            return
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()
            
        # Try to extend
        # Max possible length remaining: if we pick 256, 255... no, must be increasing.
        # Max remaining steps is roughly (256 - last_val) but that's not a tight bound for pruning depth.
        
        for x in range(last_val + 1, 257):
            # Calculate new suffix sums
            # New sums ending at x: x, x + suffix_sums[-1], x + suffix_sums[-2] ...
            # Actually, suffix_sums is [sum_ending_at_last, sum_ending_at_last-1, ...]
            # Let's define suffix_sums properly.
            # If seq = [a1, ..., ak], suffix_sums = [ak, ak-1+ak, ..., a1+...+ak]
            # If we append x, new suffix_sums = [x, x+ak, x+ak-1+ak, ...]
            # So new_suffix_sums = [x] + [x + s for s in suffix_sums]
            
            new_suffix_sums = [x]
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                
            # Check for conflicts
            # The new sums are exactly new_suffix_sums.
            # We need to check if any of these are in the existing `sums` set.
            # Note: `sums` contains all sums from previous steps.
            # The new sums are distinct from each other because seq is increasing and positive?
            # Yes, if $a_i > 0$, then suffix sums are strictly increasing in length, hence distinct values.
            # So we just need to check intersection with `sums`.
            
            conflict = False
            for ns in new_suffix_sums:
                if ns in sums:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Valid move
            new_sums = sums.union(new_suffix_sums)
            # Recurse
            dfs(seq + [x], new_sums, x, new_suffix_sums)
            
            # If we found a solution with length >= some target, we could stop?
            # No, we want to maximize. But if we hit the time limit, we stop.
            # Also, if best_len reaches a theoretical max, we can stop.
            # Theoretical max for N=256 is likely around 15-17.
            # If best_len > 15, we are doing well.

    # Initial call
    dfs([], set(), 0, [])
    
    return best_seq

# Run construct to test? No, just return the function.
# The evaluator calls construct().
