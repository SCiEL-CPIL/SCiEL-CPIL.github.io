import time
import sys

# Increase recursion limit if necessary, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # 1. Greedy Solution
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_seq = values[:]
    best_len = len(values)
    
    # Helper to compute sums and suffixes for a sequence
    def compute_state(seq):
        s_set = set()
        suffixes = []
        total = 0
        # We need all consecutive block sums.
        # For a sequence a1...ak, sums are sum(ai...aj).
        # We can compute this by iterating start i and end j.
        # Or more efficiently using the greedy method's logic but for the whole seq.
        # Actually, let's just re-calculate all sums.
        # k is small (< 20), so O(k^2) is fine.
        k = len(seq)
        for i in range(k):
            curr_sum = 0
            for j in range(i, k):
                curr_sum += seq[j]
                s_set.add(curr_sum)
        
        # Compute suffixes: sum of last 1, last 2, ... last k elements
        suffixes = [0] * k
        curr_sum = 0
        for i in range(k - 1, -1, -1):
            curr_sum += seq[i]
            suffixes[k - 1 - i] = curr_sum # suffixes[0] is last 1, suffixes[k-1] is last k
        
        return s_set, suffixes

    # DFS function
    # seq: list
    # sums: set of all consecutive block sums of seq
    # suffixes: list where suffixes[i] is sum of last i+1 elements of seq
    def dfs_extend(seq, sums, suffixes, last_val):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
            
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
            
        # Try to extend
        # Optimization: If we are close to deadline, be more conservative?
        # Just return if time is up.
        
        for x in range(last_val + 1, 257):
            # Check for conflicts
            # New sums are x, x + suffixes[0], x + suffixes[1], ...
            # suffixes[0] is last 1 element, suffixes[1] is last 2, etc.
            
            # Quick check: x itself
            if x in sums:
                continue
            
            conflict = False
            new_suffixes = [x]
            # Check x + suffixes[i]
            for s in suffixes:
                ns = x + s
                if ns in sums:
                    conflict = True
                    break
                new_suffixes.append(ns)
            
            if conflict:
                continue
            
            # If no conflict, prepare new state
            # new_sums = sums.union(new_suffixes)
            # We don't need to create the full set if we just pass it down,
            # but we need it for the next checks.
            # Creating a new set is O(|sums|). |sums| is O(k^2). k is small.
            new_sums = sums.union(new_suffixes)
            
            # Recurse
            dfs_extend(seq + [x], new_sums, new_suffixes, x)
            
            # Check time after each branch?
            if time.time() > deadline:
                return

    # 2. Repair Strategy
    # Try removing elements from the greedy sequence and re-extending
    
    # First, try extending the greedy sequence itself (in case greedy stopped early due to time or just end of range)
    # Greedy already tried all x in 1..256. So it's maximal for that path.
    # But maybe a different path from the same prefix is longer?
    # The greedy path is fixed. We can't change the prefix without removing elements.
    
    # Strategy: 
    # 1. Try to extend the full greedy sequence? No, it's stuck.
    # 2. Remove the last element, try to extend.
    # 3. Remove the last two elements, try to extend.
    # ...
    # 4. Remove element at index i, try to extend.
    
    # To be efficient, we should prioritize removing elements that are "likely" to be the bottleneck.
    # Removing the last element is the cheapest and most likely to help.
    
    # Let's iterate through possible "cut points"
    # Cut at len(G)-1, len(G)-2, etc.
    # Also, maybe cut earlier?
    
    # We will do a loop:
    # for cut_len in range(len(best_seq), 0, -1):
    #     prefix = best_seq[:cut_len]
    #     ...
    
    # However, simply removing the last element of the greedy sequence and re-running DFS
    # might just re-find the greedy sequence or a slightly different one.
    # The DFS explores *all* valid extensions from the prefix.
    # So if there is a longer sequence starting with `best_seq[:cut_len]`, DFS will find it.
    
    # We need to be careful about time.
    # Each DFS run might take some time.
    
    # Let's start with cutting the last element.
    if len(best_seq) > 1:
        prefix = best_seq[:-1]
        sums, suffixes = compute_state(prefix)
        last_val = prefix[-1]
        dfs_extend(prefix, sums, suffixes, last_val)
        
        if time.time() > deadline:
            return best_seq
    
    # Try cutting the last two elements
    if len(best_seq) > 2:
        prefix = best_seq[:-2]
        sums, suffixes = compute_state(prefix)
        last_val = prefix[-1]
        dfs_extend(prefix, sums, suffixes, last_val)
        
        if time.time() > deadline:
            return best_seq
            
    # Try cutting earlier
    # We can try a few more cut points
    # But DFS from a shorter prefix has a larger search space.
    # So we should prioritize longer prefixes.
    
    # Let's try a few more
    for cut_len in range(len(best_seq) - 3, max(0, len(best_seq) - 6), -1):
        if time.time() > deadline:
            break
        prefix = best_seq[:cut_len]
        sums, suffixes = compute_state(prefix)
        last_val = prefix[-1]
        dfs_extend(prefix, sums, suffixes, last_val)
        
    # If we still have time, we can try random restarts or different prefixes
    # But the repair strategy from the greedy sequence is a good start.
    
    # Another idea: 
    # The greedy sequence might be [1, 2, 4, 5, 8, ...]
    # What if we start from scratch but with a different heuristic?
    # E.g., try larger numbers first?
    
    # Let's add a simple DFS from scratch if we have time, trying to beat best_len.
    # But DFS from scratch is expensive.
    
    # Let's stick to the repair strategy.
    
    return best_seq

# Test locally if possible, but we just output the code.
