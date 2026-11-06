"""
Strategy: Deterministic incremental backtracking with feasibility pruning.
Unlike the current stochastic local search, this builds the sequence from scratch
using depth-first search where each step tries candidate values in a smart order
(biased toward values that leave maximum slack for future insertions).
We maintain the set of all consecutive block sums incrementally for O(n) per candidate.
"""
import time
import sys
from functools import lru_cache

_DEADLINE = 85.0

def _compute_block_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    sums = []
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.append(s)
    return sums

def _check_validity(seq):
    """Check if sequence is strictly increasing, in [1,256], with distinct block sums."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    for v in seq:
        if v < 1 or v > 256:
            return False
    sums = _compute_block_sums(seq)
    return len(set(sums)) == len(sums)

def _greedy_build():
    """Original greedy baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _try_extend(seq, sums_set, prefix_sum, max_len_target, deadline):
    """
    Try to extend the sequence to reach max_len_target using backtracking.
    seq: current sequence
    sums_set: set of all consecutive block sums
    prefix_sum: sum of all elements so far (for quick computation)
    Returns the extended sequence or None.
    """
    start_time = time.time()
    
    # If we already reached target, return
    if len(seq) >= max_len_target:
        return list(seq)
    
    # Generate candidate values to try
    # We want strictly increasing, so next value must be > seq[-1]
    last_val = seq[-1] if seq else 0
    candidates = list(range(last_val + 1, 257))
    
    # Sort candidates by a heuristic: prefer values that are "spaced out"
    # to leave room for future insertions. We estimate slack as the gap to previous.
    # Actually, let's try multiple orderings:
    # 1. Ascending (greedy-like)
    # 2. Descending (try large values first to see if we can jump)
    # 3. Middle-out
    
    # For backtracking efficiency, try ascending first (most likely to work)
    for val in candidates:
        if time.time() > deadline:
            return None
        
        # Check if adding val creates any duplicate block sums
        # New block sums created by adding val at the end:
        # - val itself
        # - seq[-1] + val
        # - seq[-2] + seq[-1] + val
        # ...
        # - sum(seq) + val
        
        new_sums = []
        running = val
        for i in range(len(seq) - 1, -1, -1):
            new_sums.append(running)
            running += seq[i]
        
        # Check for collisions
        collision = False
        for s in new_sums:
            if s in sums_set:
                collision = True
                break
        
        if collision:
            continue
        
        # No collision, try this value
        new_seq = seq + [val]
        new_sums_set = sums_set | set(new_sums)
        new_prefix_sum = prefix_sum + val
        
        # Recurse
        result = _try_extend(new_seq, new_sums_set, new_prefix_sum, max_len_target, deadline)
        if result is not None:
            return result
    
    return None

def _backtrack_search(target_len, deadline):
    """
    Try to build a sequence of length target_len using backtracking.
    Uses multiple strategies for candidate ordering.
    """
    # Strategy 1: Ascending order (greedy-like)
    result = _try_extend([], set(), 0, target_len, deadline)
    if result is not None:
        return result
    
    # If simple backtracking fails, try a different approach:
    # Start with a smaller valid sequence and try to extend it
    greedy = _greedy_build()
    if len(greedy) >= target_len:
        return greedy[:target_len]
    
    # Try extending from the greedy solution
    if len(greedy) > 0:
        sums_set = set(_compute_block_sums(greedy))
        prefix_sum = sum(greedy)
        # Try to extend greedy to target_len
        extended = _try_extend(greedy, sums_set, prefix_sum, target_len, deadline)
        if extended is not None:
            return extended
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Start with greedy baseline
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # Try to extend beyond greedy using backtracking
    # The greedy gives us a valid starting point; try to add more elements
    # by searching for values that can be inserted at various positions
    
    # First, try simple extension (append)
    for target_len in range(best_len + 1, min(best_len + 10, 257)):
        if time.time() > deadline - 5:
            break
        
        # Try to build a sequence of this length
        result = _backtrack_search(target_len, deadline - 5)
        if result is not None and len(result) > best_len:
            # Verify
            if _check_validity(result):
                best_seq = result
                best_len = len(best_seq)
    
    # Try a more sophisticated approach: 
    # Use the greedy sequence as a base and try to insert additional elements
    # at various positions
    
    if time.time() < deadline - 10:
        # Try inserting elements into the greedy sequence
        base = _greedy_build()
        base_sums = set(_compute_block_sums(base))
        
        # Try to insert one element at a time at various positions
        improved = list(base)
        improved_sums = set(base_sums)
        
        # Try multiple rounds of insertion
        for round_num in range(5):
            if time.time() > deadline - 5:
                break
            
            inserted = False
            # Try each position
            for pos in range(len(improved) + 1):
                if time.time() > deadline - 5:
                    break
                
                low = 1 if pos == 0 else improved[pos - 1] + 1
                high = 256 if pos == len(improved) else improved[pos] - 1
                
                if low > high:
                    continue
                
                # Try values in this range
                for val in range(low, high + 1):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute new block sums if we insert val at pos
                    # This is more complex; let's just check the full sequence
                    test_seq = improved[:pos] + [val] + improved[pos:]
                    
                    # Quick check: strictly increasing
                    valid_inc = True
                    for i in range(1, len(test_seq)):
                        if test_seq[i] <= test_seq[i-1]:
                            valid_inc = False
                            break
                    if not valid_inc:
                        continue
                    
                    # Check block sums
                    sums = _compute_block_sums(test_seq)
                    if len(set(sums)) == len(sums):
                        improved = test_seq
                        improved_sums = set(sums)
                        inserted = True
                        break
                
                if inserted:
                    break
            
            if not inserted:
                break
        
        if len(improved) > best_len and _check_validity(improved):
            best_seq = improved
            best_len = len(best_seq)
    
    # Final safety check
    if not _check_validity(best_seq):
        best_seq = _greedy_build()
    
    return best_seq
