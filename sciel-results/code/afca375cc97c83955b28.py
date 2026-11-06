import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_seq = []
    
    # Use non-local or class to track best
    # We will use a simple recursive function
    
    def dfs(seq, sums, last_val):
        nonlocal best_seq, deadline
        
        # Check time
        if time.time() > deadline:
            return
            
        # Update best if current is longer
        if len(seq) > len(best_seq):
            best_seq = list(seq)
            
        # Pruning: Can we possibly beat best_seq?
        # Max possible length is len(seq) + (256 - last_val)
        # If this is <= len(best_seq), we can't improve.
        if len(seq) + (256 - last_val) <= len(best_seq):
            return
            
        # Try next values
        # Heuristic: Try smallest first to keep sums small
        for x in range(last_val + 1, 257):
            # Calculate new sums
            # New sums are:
            # 1. x
            # 2. x + seq[-1]
            # 3. x + seq[-1] + seq[-2]
            # ...
            # 4. x + sum(seq)
            
            # We need to check if any of these are in `sums`
            # Or if any of these are equal to each other? 
            # No, sums within the new block are distinct because seq is strictly increasing positive ints.
            # So we only need to check against existing `sums`.
            
            # Optimization: Pre-compute suffix sums of seq?
            # Or just iterate.
            
            current_sum = 0
            valid = True
            # Iterate from end of seq to start
            # The new sums are x + suffix_sum[i]
            # where suffix_sum[i] is sum of seq[i:]
            # Note: x itself is x + 0
            
            # Let's just build the list of new sums
            new_sums = []
            s = 0
            for val in reversed(seq):
                s += val
                new_sums.append(x + s)
            new_sums.append(x) # The block [x]
            
            # Check for intersection
            # Since new_sums are distinct among themselves, we just check against `sums`
            for ns in new_sums:
                if ns in sums:
                    valid = False
                    break
            
            if valid:
                # Update sums
                # We can create a new set: sums | set(new_sums)
                new_sums_set = sums.union(new_sums)
                
                # Recurse
                dfs(seq + [x], new_sums_set, x)
                
                # Backtracking is implicit in the recursion returning
                # We don't need to modify `sums` in place if we pass a new set
                # But creating a new set is O(N). N is number of sums ~ L^2.
                # L is likely < 20. So N < 400. This is fast.

    # Initial call
    # We can start with an empty sequence
    dfs([], set(), 0)
    
    return best_seq
