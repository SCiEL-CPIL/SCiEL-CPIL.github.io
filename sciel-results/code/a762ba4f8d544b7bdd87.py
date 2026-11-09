import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    best_seq = []
    best_len = 0
    
    # 1. Greedy initialization to get a good lower bound quickly
    greedy_seq, greedy_sums, greedy_suffix = [], set(), []
    for x in range(1, 257):
        new_sums = [x]
        for s in greedy_suffix:
            new_sums.append(x + s)
        if not greedy_sums.intersection(new_sums):
            greedy_seq.append(x)
            greedy_sums.update(new_sums)
            greedy_suffix = new_sums
    best_seq = greedy_seq
    best_len = len(greedy_seq)
    
    # 2. Backtracking search to improve
    def dfs(seq, sums, suffix_sums, last_val):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
        
        current_len = len(seq)
        
        # Pruning: Can we beat best_len?
        # Max possible length is current_len + (256 - last_val)
        if current_len + (256 - last_val) <= best_len:
            return
            
        if current_len > best_len:
            best_len = current_len
            best_seq = list(seq)
            
        # Try next numbers
        # Heuristic: Try smaller numbers first to keep sums small and allow more elements
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            # Generate new suffix sums
            new_sums = [x]
            for s in suffix_sums:
                new_sums.append(x + s)
            
            # Check for conflicts
            conflict = False
            for ns in new_sums:
                if ns in sums:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # No conflict, proceed
            seq.append(x)
            sums.update(new_sums)
            
            dfs(seq, sums, new_sums, x)
            
            # Backtrack
            seq.pop()
            for ns in new_sums:
                sums.discard(ns)
            
            # If we found a sequence significantly better, we might want to restart?
            # No, DFS continues.
            
    # Start DFS from empty sequence
    # We can also try starting from 1, 2, etc. but empty covers all.
    dfs([], set(), [], 0)
    
    return best_seq
