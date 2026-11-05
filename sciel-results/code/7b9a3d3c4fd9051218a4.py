"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def _check_valid_fast(seq, existing_sums):
    """
    Check if appending the last element of seq to a sequence with existing_sums is valid.
    Returns (is_valid, new_sums_set)
    """
    if not seq:
        return True, existing_sums
    
    x = seq[-1]
    n = len(seq) - 1
    # New sums are sums of suffixes of seq ending at x
    # Suffix sums: x, seq[-2]+x, seq[-3]+seq[-2]+x, ...
    
    # We need to check if any of these new sums are in existing_sums
    # And also if any new sums collide with each other (they won't if seq is strictly increasing and positive? 
    # No, suffix sums are strictly increasing if elements are positive. 
    # x < seq[-2]+x < ... 
    # So new sums are distinct among themselves.
    # We only need to check against existing_sums.
    
    s = 0
    new_sums = set()
    for i in range(len(seq) - 1, -1, -1):
        s += seq[i]
        if s in existing_sums:
            return False, None
        new_sums.add(s)
        
    # Union
    new_sums.update(existing_sums)
    return True, new_sums

def _dfs_backtrack(seq, sums, last_val, target_len, best_seq, time_limit):
    """
    Depth First Search to find a sequence of length target_len.
    """
    if time.time() > time_limit:
        return False
    
    if len(seq) == target_len:
        best_seq.extend(seq)
        return True
        
    # Pruning: If we can't reach target_len even with max values?
    # Max possible length is bounded by 256 - last_val, but also by sum constraints.
    # Simple bound: if len(seq) + (256 - last_val) < target_len: return False
    
    # Try candidates from last_val + 1 to 256
    # Heuristic: Try smaller values first? Or larger?
    # Smaller values keep sums smaller, potentially allowing more distinct sums later?
    # Actually, smaller values pack tighter, which might cause collisions.
    # Larger values spread sums out.
    # Let's try increasing order (greedy-like) but with backtracking.
    
    for x in range(last_val + 1, 257):
        # Quick check: if x is too small, might cause collision with existing small sums?
        # Just check validity.
        
        # We pass seq + [x]
        # To avoid copying list, we can append/pop
        seq.append(x)
        valid, new_sums = _check_valid_fast(seq, sums)
        
        if valid:
            if _dfs_backtrack(seq, new_sums, x, target_len, best_seq, time_limit):
                return True
        
        seq.pop()
        
    return False

def _find_max_length_by_search(start_len, time_limit_per_attempt=5.0):
    """
    Try to find a sequence of length start_len, start_len+1, etc.
    """
    best_seq = []
    current_len = start_len
    
    while True:
        # Reset best_seq for this attempt
        best_seq = []
        seq = []
        sums = set()
        last_val = 0
        
        # Start time for this specific target length
        t_start = time.time()
        
        # Use a global time limit for the whole search phase
        found = _dfs_backtrack(seq, sums, last_val, current_len, best_seq, time.time() + time_limit_per_attempt)
        
        if found:
            current_len += 1
            # Continue to next length
            if time.time() - t_start > 40: # Global time limit safety
                break
        else:
            # Failed to find sequence of current_len
            break
            
    return best_seq

def construct():
    # 1. Get a baseline from Greedy (fast)
    # Re-implementing a fast greedy to get a good start
    values = []
    sums = set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    baseline_len = len(values)
    best_seq = values
    
    # 2. Try to improve using Backtracking Search
    # Start from baseline_len + 1
    # We have ~40 seconds left.
    # We will try to find a sequence of length baseline_len + 1, then +2, etc.
    
    # To make DFS faster, we can use the baseline sequence as a guide?
    # Or just run DFS from scratch for the target length.
    # DFS from scratch might be slow if branching is high.
    # However, the constraint is tight.
    
    # Let's try to find a sequence of length baseline_len + 1
    target_len = baseline_len + 1
    t_start = time.time()
    time_limit = 45.0 # Leave 15s for safety
    
    while target_len <= 20: # Max possible is likely < 20
        seq = []
        sums = set()
        last_val = 0
        best_seq_for_len = []
        
        # Use a tighter time limit per attempt
        attempt_limit = time.time() + 5.0
        
        found = _dfs_backtrack(seq, sums, last_val, target_len, best_seq_for_len, attempt_limit)
        
        if found:
            if len(best_seq_for_len) > len(best_seq):
                best_seq = best_seq_for_len
            target_len += 1
        else:
            # If we failed to find a sequence of this length, we stop.
            # It's possible a longer sequence exists but a shorter one doesn't?
            # No, if a sequence of length L exists, a sequence of length L-1 exists (prefix).
            # So if we fail to find L, we can stop.
            break
            
        if time.time() > time_limit:
            break
            
    return best_seq
# EVOLVE-BLOCK-END
