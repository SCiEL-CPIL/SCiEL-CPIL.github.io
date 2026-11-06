"""Structural redesign: multi-start greedy with lookahead + deadline guard."""
import time
import random
from itertools import combinations

# We want strictly increasing sequence in [1,256] where all nonempty consecutive-block sums are distinct.
# Maximize length.

# Helper: given a partial sequence (list), compute the set of all consecutive-block sums.
# For efficiency, we maintain the set incrementally.

def _all_block_sums(seq):
    """Compute all nonempty consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _check_valid(seq):
    """Check that seq is strictly increasing, all elements in [1,256], and all consecutive block sums distinct."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    if any(x < 1 or x > 256 for x in seq):
        return False
    sums = _all_block_sums(seq)
    # Number of consecutive blocks is n(n+1)/2
    n = len(seq)
    expected = n * (n + 1) // 2
    if len(sums) != expected:
        return False
    return True

def _greedy_build(start_candidates, used_sums, max_time=None, start_time=None):
    """
    Greedy build: try to extend the sequence by picking the smallest (or given order) valid next element.
    start_candidates: list of candidate values to try in order for the first element (or a fixed start).
    Actually, let's make this more flexible.
    """
    pass

def _greedy_with_lookahead(seq, sums_set, remaining_candidates, max_time=None, start_time=None):
    """
    Given current seq and its block sums, try to extend by one element.
    For each candidate x > seq[-1] in remaining_candidates:
        Check if adding x keeps all block sums distinct.
        If multiple work, prefer the one that leaves the most future options (lookahead).
    Return (new_seq, new_sums, next_candidates) or None if no extension possible.
    """
    if max_time is not None and start_time is not None:
        if time.time() - start_time > max_time:
            return None
    
    if not seq:
        last = 0
    else:
        last = seq[-1]
    
    n = len(seq)
    # When we add x, the new block sums are:
    #   x itself, and for each i in 0..n-1: sum(seq[i:]) + x ... wait no.
    # Actually, new consecutive blocks ending at x:
    #   x, seq[n-1]+x, seq[n-2]+seq[n-1]+x, ..., sum(seq)+x
    # Plus all old block sums.
    
    # Precompute suffix sums of current seq for quick check
    # suffix_sums[k] = sum(seq[k:]) for k in 0..n
    # When adding x, new sums are: x, suffix_sums[n-1]+x, suffix_sums[n-2]+x, ..., suffix_sums[0]+x
    # i.e., for k in 0..n: suffix_sums[k] + x where suffix_sums[n] = 0, so suffix_sums[n]+x = x
    
    if n == 0:
        suffix_sums = [0]
    else:
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
    
    valid_next = []
    for x in remaining_candidates:
        if x <= last:
            continue
        # Check new sums
        conflict = False
        for k in range(n + 1):
            new_sum = suffix_sums[k] + x
            if new_sum in sums_set:
                conflict = True
                break
        if not conflict:
            valid_next.append(x)
    
    if not valid_next:
        return None
    
    # Lookahead: for each valid x, count how many further extensions are possible (1-step lookahead)
    # To keep it fast, just pick the smallest valid x (greedy) OR do a limited lookahead.
    # Let's do: try the smallest valid x first (greedy), but if we want lookahead, we can score.
    # For now, pure greedy: pick smallest valid x.
    best_x = valid_next[0]
    
    # Update seq and sums
    new_seq = seq + [best_x]
    new_sums = set(sums_set)
    for k in range(n + 1):
        new_sums.add(suffix_sums[k] + best_x)
    
    # Update remaining candidates
    new_remaining = [x for x in remaining_candidates if x > best_x]
    
    return (new_seq, new_sums, new_remaining)

def _build_greedy(start_value, max_time=55.0):
    """
    Build a sequence starting with start_value using greedy smallest-first extension.
    """
    start_time = time.time()
    seq = [start_value]
    sums_set = {start_value}
    remaining = list(range(start_value + 1, 257))
    
    while True:
        if time.time() - start_time > max_time:
            break
        result = _greedy_with_lookahead(seq, sums_set, remaining, max_time, start_time)
        if result is None:
            break
        seq, sums_set, remaining = result
    
    return seq

def _build_greedy_multi_start(max_time=55.0):
    """
    Try multiple starting values and return the best sequence.
    """
    start_time = time.time()
    best_seq = []
    best_len = 0
    
    # Try different starting values
    # Start from 1, 2, 3, ... up to some range
    # Also try larger starts
    start_values = list(range(1, 50)) + list(range(50, 257, 10))
    random.shuffle(start_values)
    # Ensure 1 is tried first or early
    if 1 in start_values:
        start_values.remove(1)
        start_values.insert(0, 1)
    
    for sv in start_values:
        if time.time() - start_time > max_time - 2.0:
            break
        seq = _build_greedy(sv, max_time=max_time - (time.time() - start_time) - 1.0)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    return best_seq

def _dfs_build(max_depth, max_time=55.0):
    """
    DFS with backtracking to find longer sequences.
    This is a structural change: use backtracking search instead of pure greedy.
    """
    start_time = time.time()
    best_seq = []
    
    def dfs(seq, sums_set, remaining, depth):
        nonlocal best_seq, start_time
        if time.time() - start_time > max_time - 2.0:
            return
        if len(seq) > len(best_seq):
            best_seq = seq[:]
        if len(seq) >= max_depth:
            return
        if not remaining:
            return
        
        n = len(seq)
        # Compute suffix sums
        if n == 0:
            suffix_sums = [0]
        else:
            suffix_sums = [0] * (n + 1)
            s = 0
            for k in range(n - 1, -1, -1):
                s += seq[k]
                suffix_sums[k] = s
            suffix_sums[n] = 0
        
        last = seq[-1] if seq else 0
        
        # Try each candidate
        for i, x in enumerate(remaining):
            if x <= last:
                continue
            # Check conflict
            conflict = False
            for k in range(n + 1):
                new_sum = suffix_sums[k] + x
                if new_sum in sums_set:
                    conflict = True
                    break
            if conflict:
                continue
            
            new_seq = seq + [x]
            new_sums = sums_set | {suffix_sums[k] + x for k in range(n + 1)}
            new_remaining = remaining[i+1:]
            
            dfs(new_seq, new_sums, new_remaining, depth + 1)
            
            if time.time() - start_time > max_time - 2.0:
                return
    
    # Start with empty sequence
    dfs([], set(), list(range(1, 257)), 0)
    
    return best_seq

def construct():
    """
    Main entry point. Returns a strictly increasing sequence in [1,256] with distinct consecutive-block sums.
    Uses multi-start greedy with deadline guard.
    """
    total_start = time.time()
    deadline = total_start + 55.0  # Safety margin before 60s
    
    # First, try the original greedy (smallest-first from 1) as a baseline
    baseline = _build_greedy(1, max_time=10.0)
    
    # Then try multi-start greedy
    multi_start = _build_greedy_multi_start(max_time=40.0)
    
    # Choose the best
    if len(multi_start) > len(baseline):
        result = multi_start
    else:
        result = baseline
    
    # Verify validity
    if not _check_valid(result):
        # Fallback to a known valid small sequence
        result = [1, 2, 5]
    
    return result
