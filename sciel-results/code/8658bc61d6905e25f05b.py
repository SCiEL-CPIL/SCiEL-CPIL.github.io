"""Strategy: Deterministic backtracking with greedy heuristics to maximize sequence length."""
import time
import sys

sys.setrecursionlimit(10000)

_DEADLINE = 85.0

def _is_valid_prefix(seq, used_sums):
    """
    Check if adding the last element of seq maintains validity.
    seq is assumed to be strictly increasing and values in [1, 256].
    used_sums is the set of all block sums from seq[:-1].
    Returns (is_valid, new_used_sums)
    """
    if not seq:
        return True, used_sums
    
    last_val = seq[-1]
    prev_seq = seq[:-1]
    
    # Calculate new sums ending at last_val
    new_sums = []
    current_sum = 0
    for i in range(len(seq) - 1, -1, -1):
        current_sum += seq[i]
        new_sums.append(current_sum)
        
    # Check for collisions with existing sums
    for s in new_sums:
        if s in used_sums:
            return False, used_sums
            
    # Update used_sums
    new_used_sums = set(used_sums)
    new_used_sums.update(new_sums)
    
    return True, new_used_sums

def _backtrack_search(current_seq, current_sums, max_val, target_len):
    """
    Recursively try to extend current_seq to reach target_len.
    current_seq: list of ints
    current_sums: set of all block sums
    max_val: max allowed value (256)
    target_len: desired final length
    Returns: list of ints if found, else None
    """
    if len(current_seq) == target_len:
        return current_seq[:]
    
    if time.time() > _DEADLINE:
        return None
    
    last_val = current_seq[-1] if current_seq else 0
    
    # Try to extend. We want to keep numbers small to allow more elements.
    # So we iterate from last_val + 1 upwards.
    # Optimization: We can prune if even using the smallest possible next values
    # we can't reach the target length? Hard to bound tightly without knowing gaps.
    # Simple bound: if (max_val - last_val) < (target_len - len(current_seq)), we can't fit enough distinct integers.
    if (max_val - last_val) < (target_len - len(current_seq)):
        return None

    start_val = last_val + 1
    # We must pick distinct integers.
    for next_val in range(start_val, max_val + 1):
        # Quick check: if we pick next_val, can we fit the remaining elements?
        # Remaining elements needed: target_len - (len(current_seq) + 1)
        # Available values: max_val - next_val
        if (max_val - next_val) < (target_len - len(current_seq) - 1):
            continue
            
        candidate_seq = current_seq + [next_val]
        
        # Check validity incrementally
        is_valid, new_sums = _is_valid_prefix(candidate_seq, current_sums)
        
        if is_valid:
            result = _backtrack_search(candidate_seq, new_sums, max_val, target_len)
            if result is not None:
                return result
            if time.time() > _DEADLINE:
                return None

    return None

def _greedy_build():
    """
    Greedy baseline: pick the smallest available number that doesn't create a duplicate block sum.
    This usually yields a reasonably long sequence quickly.
    """
    values = []
    sums = set()
    for x in range(1, 257):
        # Calculate new sums if we add x
        new_sums = []
        current_sum = 0
        # Sums ending at x
        # x, x + values[-1], x + values[-1] + values[-2], ...
        # Wait, the block sums are consecutive.
        # If we append x to values, the new block sums are:
        # x
        # x + values[-1]
        # x + values[-1] + values[-2]
        # ...
        # x + sum(values)
        
        # Let's compute them
        rev_values = list(reversed(values))
        s = 0
        for i in range(len(rev_values)):
            s += rev_values[i]
            new_sums.append(x + s)
        new_sums.append(x) # The block [x] itself? 
        # Actually, the loop above covers [x, v_{n-1}], [x, v_{n-1}, v_{n-2}]...
        # The block [x] is just x.
        # My loop:
        # i=0: s = v_{n-1}. sum = x + v_{n-1}. Block [v_{n-1}, x]? No, sequence is increasing.
        # Block sums are sums of seq[i:j].
        # If we append x to seq, the new blocks are those ending at x.
        # seq[k..n-1] for k=0..n-1.
        # Let seq be a_1, ..., a_m. New element a_{m+1}=x.
        # New sums: 
        # a_{m+1}
        # a_m + a_{m+1}
        # a_{m-1} + a_m + a_{m+1}
        # ...
        # sum(a_1..a_m) + a_{m+1}
        
        # Let's re-calculate correctly.
        new_sums = [x]
        s = 0
        for i in range(len(values) - 1, -1, -1):
            s += values[i]
            new_sums.append(s + x)
            
        # Check intersection
        if not sums.intersection(new_sums):
            values.append(x)
            sums.update(new_sums)
            
    return values

def construct():
    start_time = time.time()
    
    # 1. Get a good baseline using greedy
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # 2. Try to improve using backtracking
    # We want to find a sequence of length L > best_len.
    # We can try to construct a sequence of length L using backtracking.
    # To make backtracking efficient, we can use the greedy sequence as a guide or start.
    # However, backtracking from scratch might be slow if the space is large.
    # A better approach: Use backtracking to verify if a longer sequence exists,
    # and if so, find one.
    
    # Let's try lengths from best_len + 1 upwards, but limited by time.
    # Since max value is 256, max possible length is 256 (1,2,...,256).
    # But constraints are tight. Greedy usually gets ~10-15? Let's check.
    
    # Let's try to find a sequence of length best_len + 1, then +2, etc.
    # We will use a timeout for each backtracking attempt.
    
    # Optimization: We can prune the search space by noting that if a sequence of length L exists,
    # it's likely composed of smaller numbers.
    
    # Let's try a hybrid:
    # Start with the greedy sequence.
    # Try to extend it? No, greedy is maximal for its path.
    # We need a different path.
    
    # Let's run backtracking for target_len = best_len + 1.
    # If it finds a solution, update best_len and try best_len + 1.
    
    # To speed up backtracking, we can order the next values.
    # Smaller values are generally better for packing more elements.
    
    # We'll implement a recursive search that tries to build a sequence of length L.
    # It will start with an empty sequence and try values 1, 2, 3...
    # This is essentially the same as _backtrack_search but with a specific target.
    
    # To avoid re-exploring the same dead ends, we could use memoization,
    # but the state is (last_val, used_sums) which is huge.
    # So simple backtracking with pruning is the way.
    
    # We'll add a global timeout check inside the recursion.
    
    # Let's try to find a sequence of length best_len + 1.
    # If found, set best_seq to it and increment target.
    
    # To make it faster, we can restrict the search space.
    # E.g., if we are looking for length 15, we don't need to consider values > 200 in the early steps?
    # Actually, we do, but smaller is preferred.
    
    # Let's run the backtracking search.
    # We will try to find a sequence of length L starting from scratch.
    
    # To prevent the recursion from being too deep or slow, we limit the depth
    # and use the deadline.
    
    # Let's try to improve best_len.
    current_target = best_len + 1
    
    # We will try to find a valid sequence of length current_target.
    # We use backtracking.
    # To make it efficient, we can use a heuristic: 
    # When choosing the next number, prefer smaller ones.
    
    # The _backtrack_search function above does this (iterates from last_val+1 upwards).
    
    # However, starting from scratch for each target length is redundant.
    # We can just run one search that tries to maximize length.
    # But maximizing length is hard.
    
    # Alternative: 
    # Run backtracking for target_len = best_len + 1.
    # If successful, best_len += 1.
    # Repeat until failure or timeout.
    
    # This is safe because if a sequence of length L+1 exists, the search will find it
    # (assuming it explores the space sufficiently).
    
    # Let's implement a helper that tries to find a sequence of a specific length.
    # It returns the sequence or None.
    
    def find_sequence_of_length(L):
        # Use backtracking to find a sequence of length L.
        # Start with empty sequence.
        # To speed up, we can use a greedy prefix?
        # No, just backtrack from start.
        
        # We can add a heuristic: if L is large, the values must be small.
        # Max sum of L smallest distinct integers is L(L+1)/2.
        # This doesn't help much with the distinct block sum constraint.
        
        # Let's just run the backtracking.
        # We pass an empty list and empty set.
        # We need to be careful with recursion depth and time.
        
        # To avoid deep recursion overhead, we could use an iterative stack, 
        # but Python recursion is fine for depth < 256.
        
        # We'll use the _backtrack_search function.
        # It returns a sequence or None.
        
        # One optimization: if the time remaining is very small, abort.
        
        result = _backtrack_search([], set(), 256, L)
        return result
    
    # Try to find longer sequences
    while time.time() < _DEADLINE - 5:  # Leave 5s buffer
        target_len = best_len + 1
        # If target_len is too large, we might not find it quickly.
        # But let's try.
        
        # To avoid getting stuck in a long search for a non-existent length,
        # we can impose a smaller time limit per search?
        # The global deadline handles this.
        
        found = find_sequence_of_length(target_len)
        
        if found:
            best_seq = found
            best_len = len(best_seq)
        else:
            # If we didn't find a sequence of length best_len + 1, 
            # it might be because the search space was exhausted or time ran out.
            # If time ran out, we break.
            # If space was exhausted (unlikely for small L), we break.
            # How to distinguish? 
            # _backtrack_search returns None if no solution found OR if timeout.
            # We can't easily distinguish.
            # But if we just found a sequence of length L, and failed for L+1, 
            # it's likely L is the max for this search strategy.
            # We can break to save time for final validation.
            break
            
    # Final safety check
    if not best_seq or len(best_seq) == 0:
        best_seq = _greedy_build()
        
    # Verify validity of the returned sequence
    # (The backtracking guarantees validity, but greedy does too. 
    # Just in case of any bug, we re-verify.)
    
    # Quick verification
    def _verify(seq):
        if not seq:
            return False
        if seq[0] < 1 or seq[-1] > 256:
            return False
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    if not _verify(best_seq):
        # Fallback to greedy if something went wrong
        best_seq = _greedy_build()
        if not _verify(best_seq):
            # Absolute fallback: [1]
            best_seq = [1]

    return best_seq

# EVOLVE-BLOCK-END
