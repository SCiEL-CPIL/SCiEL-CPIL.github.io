"""Improved program: backtracking DFS with in-place state for max-length increasing sequence with distinct consecutive-block sums."""
import time
import random
import sys

def _check_valid(seq):
    """Check if a strictly increasing sequence has all distinct consecutive-block sums."""
    n = len(seq)
    if n == 0:
        return True
    for i in range(n - 1):
        if seq[i + 1] <= seq[i]:
            return False
    for x in seq:
        if x < 1 or x > 256:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build_fast(order, deadline=None):
    """Fast greedy using a set for O(1) lookup of conflicts."""
    values = []
    sums = set()
    
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        total = x
        conflict = False
        for a in reversed(values):
            if total in sums:
                conflict = True
                break
            total += a
        if total in sums:
            conflict = True
        
        if conflict:
            continue
        
        total = x
        sums.add(total)
        for a in reversed(values):
            total += a
            sums.add(total)
        
        values.append(x)
    
    return values

def _backtracking_dfs(start_val, end_val, current_seq, current_sums, suffix_sums, deadline, best_ref, max_depth_hint=20):
    """
    Backtracking DFS with in-place state modification.
    
    State:
    - current_seq: list of chosen values (strictly increasing)
    - current_sums: set of all consecutive block sums
    - suffix_sums: list where suffix_sums[i] = sum of current_seq[i:] (suffix sums)
    
    We try to extend by choosing the next value from [start_val+1, end_val].
    """
    if time.time() > deadline:
        return
    
    n = len(current_seq)
    if n > len(best_ref[0]):
        best_ref[0] = list(current_seq)
    
    # If we've reached max depth hint, stop going deeper (but still record)
    if n >= max_depth_hint:
        return
    
    # Try each candidate value in order
    # Heuristic: try values that are "spaced out" from the last value
    last_val = current_seq[-1] if current_seq else 0
    min_next = last_val + 1
    
    for x in range(min_next, end_val + 1):
        if time.time() > deadline:
            return
        
        # Compute new suffix sums if we add x
        # New suffix sums: x, x + current_seq[-1], x + current_seq[-1] + current_seq[-2], ...
        new_suffix = [x]
        total = x
        for i in range(n - 1, -1, -1):
            total += current_seq[i]
            new_suffix.append(total)
        
        # Check if any new suffix sum conflicts with existing sums
        conflict = False
        for s in new_suffix:
            if s in current_sums:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Also check: do new suffix sums conflict with each other? No, they're all different since seq is increasing and positive.
        
        # Commit: add x to state
        current_seq.append(x)
        for s in new_suffix:
            current_sums.add(s)
        
        # Recurse
        _backtracking_dfs(x, end_val, current_seq, current_sums, new_suffix, deadline, best_ref, max_depth_hint)
        
        # Undo
        current_seq.pop()
        for s in new_suffix:
            current_sums.discard(s)
        
        # Early termination: if we've found a sequence of max_depth_hint, we can stop
        # (since we're searching for max length, and we've hit the target)
        if len(best_ref[0]) >= max_depth_hint:
            return

def _dfs_with_random_order(start_candidates, end_val, deadline, best_ref, max_depth_hint=20):
    """
    Run DFS starting from a shuffled set of initial candidates.
    """
    current_seq = []
    current_sums = set()
    
    # Try each starting value
    for x in start_candidates:
        if time.time() > deadline:
            return
        
        # Check if x alone is valid (trivially yes)
        current_seq.append(x)
        current_sums.add(x)
        
        _backtracking_dfs(x, end_val, current_seq, current_sums, [x], deadline, best_ref, max_depth_hint)
        
        current_seq.pop()
        current_sums.discard(x)
        
        if len(best_ref[0]) >= max_depth_hint:
            return

def construct():
    """
    Return a strictly increasing sequence of integers in [1,256] where all 
    nonempty consecutive block sums are distinct. Maximize sequence length.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_seq = []
    
    # Phase 1: Quick greedy to get a baseline
    seq = _greedy_build_fast(range(1, 257), deadline)
    best_seq = seq
    baseline_len = len(seq)
    
    # Phase 2: Backtracking DFS to improve
    # Use the greedy result to set a reasonable max_depth_hint
    # The greedy gives us a lower bound; try to exceed it
    target_len = baseline_len + 3  # Try to get 3 more than greedy
    
    # Strategy: Start DFS from various starting values
    # Focus on starting values that are small to allow room for many elements
    start_candidates = list(range(1, 30))
    random.shuffle(start_candidates)
    
    best_ref = [best_seq]
    
    # Run DFS with increasing depth hints
    for depth_hint in [baseline_len + 1, baseline_len + 2, target_len]:
        if time.time() > deadline - 5.0:
            break
        
        _dfs_with_random_order(start_candidates, 256, deadline, best_ref, depth_hint)
        
        if len(best_ref[0]) >= depth_hint:
            # We hit the target, try to go even deeper
            continue
    
    # Phase 3: Try DFS starting from different ranges
    if time.time() < deadline - 10.0:
        # Try starting from slightly larger values with more spread
        start_candidates2 = list(range(5, 50))
        random.shuffle(start_candidates2)
        
        for depth_hint in [len(best_ref[0]) + 1, len(best_ref[0]) + 2]:
            if time.time() > deadline - 5.0:
                break
            _dfs_with_random_order(start_candidates2, 256, deadline, best_ref, depth_hint)
    
    # Phase 4: If time remains, try more random restarts
    if time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            start_candidates3 = list(range(1, 60))
            random.shuffle(start_candidates3)
            depth_hint = len(best_ref[0]) + 1
            _dfs_with_random_order(start_candidates3, 256, deadline, best_ref, depth_hint)
    
    best_seq = best_ref[0]
    
    # Final validation
    if not _check_valid(best_seq):
        # Fallback to the simple greedy which is guaranteed valid
        best_seq = _greedy_build_fast(range(1, 257))
    
    return best_seq

if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
