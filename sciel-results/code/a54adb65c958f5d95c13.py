import time
import sys

def construct():
    start_time = time.time()
    # 50s safety margin to ensure we return before 60s limit
    deadline = start_time + 50
    
    best_seq = []
    
    # 1. Greedy Baseline
    # Try to build a sequence greedily using the smallest possible next element.
    # This ensures we use the smallest numbers, leaving the most "room" for count,
    # though it might not maximize the sum range.
    curr_seq = []
    curr_sums = set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(curr_seq):
            total += a
            tails.append(total)
        if not curr_sums.intersection(tails):
            curr_seq.append(x)
            curr_sums.update(tails)
    best_seq = curr_seq[:]
    
    # 2. DFS with Pruning
    # We search for longer sequences by backtracking.
    # We iterate candidates in ascending order to keep 'last_val' low,
    # which maximizes the number of available integers for future steps.
    
    def dfs(seq, sums, last_val):
        nonlocal best_seq
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > len(best_seq):
            best_seq = seq[:]
            
        # Pruning: Check if we can possibly beat the current best length.
        # The maximum number of elements we can still add is (256 - last_val).
        # If current length + max possible additions <= best length, stop.
        if n + (256 - last_val) <= len(best_seq):
            return
            
        # Try adding next elements
        # Ascending order: 1, 2, 3...
        # This prioritizes sequences with smaller elements, which allows for
        # more elements to be added later (since we are bounded by 256).
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Calculate new consecutive sums ending at x
            # These are x, x + seq[-1], x + seq[-1] + seq[-2], ...
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
                
            # Check for conflicts with existing sums
            conflict = False
            for s in new_sums:
                if s in sums:
                    conflict = True
                    break
                    
            if not conflict:
                # Update state
                sums.update(new_sums)
                seq.append(x)
                
                dfs(seq, sums, x)
                
                # Backtrack
                seq.pop()
                sums.difference_update(new_sums)

    # Start DFS from empty sequence
    # We can start from the greedy result to prune more effectively?
    # No, starting from empty allows finding sequences that are lexicographically smaller
    # or structurally different. But the greedy result is a good lower bound.
    # We can also initialize the DFS with the greedy sequence to start deeper?
    # No, DFS needs to explore branches.
    # However, we can set best_seq to the greedy result, which we already did.
    
    dfs([], set(), 0)
    
    return best_seq
