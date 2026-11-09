import time
import sys
import functools

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    if n == 0:
        return True
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _compute_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += seq[j]
            sums.add(total)
    return sums

def _valid_extension(seq, x, cur_sums):
    """Check if x can be appended to seq without creating duplicate sums."""
    tails = [x]
    total = x
    for a in reversed(seq):
        total += a
        tails.append(total)
    return not cur_sums.intersection(tails)

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element. Used for initial seed."""
    current = list(values)
    if not current:
        cur_sums = set()
    else:
        cur_sums = _compute_sums(current)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        if _valid_extension(current, x, cur_sums):
            tails = [x]
            total = x
            for a in reversed(current):
                total += a
                tails.append(total)
            current.append(x)
            cur_sums.update(tails)
    return current

def _upper_bound_length(seq, cur_sums, max_val=256):
    """
    Estimate the maximum possible length of a sequence extending `seq`.
    
    Heuristic 1: Value space limit.
    If current last value is L, we can pick at most (max_val - L) more values.
    So upper bound is len(seq) + (max_val - L).
    
    Heuristic 2: Sum space limit.
    The maximum possible sum is sum(seq) + sum of largest possible extension.
    Actually, a tighter bound comes from the fact that we need distinct sums.
    The number of distinct sums is n(n+1)/2.
    The maximum sum is bounded by max_val * n.
    This is loose.
    
    Let's use a tighter structural bound.
    For a sequence of length k, the sums range from min_val to max_sum.
    The number of distinct sums is k(k+1)/2.
    The range of sums is at most (max_val * k) - min_val.
    This doesn't directly limit k easily without solving a quadratic.
    
    However, a very strong bound for this specific "increasing" constraint:
    If we pick the smallest possible values, we maximize the count.
    The greedy approach picks the smallest valid values.
    If the greedy approach from `seq` yields length G, then any other branch
    from `seq` cannot yield a length > G? NO, that's false. Picking a larger
    value now might allow a much larger value later, or different combinations.
    But generally, smaller values leave more "sum budget" for distinctness.
    
    Let's use a simple but effective bound:
    Max additional elements is limited by the available integers.
    Additional elements <= max_val - seq[-1].
    
    Also, we can bound by the number of unused sums?
    Total possible sums in [1, max_val * (len(seq) + max_val - seq[-1])] is huge.
    
    Let's stick to:
    1. Value range bound: len(seq) + (max_val - seq[-1])
    2. If this bound <= current_best, prune.
    """
    if not seq:
        return max_val
    return len(seq) + (max_val - seq[-1])

def _dfs_csp(seq, cur_sums, max_val, deadline, best_len, best_seq, dead_ends):
    """
    DFS with Branch and Bound.
    
    Args:
    seq: Current sequence
    cur_sums: Set of all block sums for seq
    max_val: Maximum value allowed (256)
    deadline: Time limit
    best_len: Current best length found
    best_seq: Current best sequence
    dead_ends: Set of prefixes (tuples) that have been explored and found to be dead ends (or not improving).
               To keep memory low, we only store prefixes of length <= some threshold or use a hash.
               Actually, storing the whole tuple is fine for small depths.
    """
    # Check time
    if time.time() > deadline:
        return best_len, best_seq
    
    # Check dead end cache
    # We only cache if we are deep enough to make it useful, or if we are sure it's a dead end.
    # For simplicity, we cache states where we failed to extend further or reached a local max.
    # Key: (tuple(seq), frozenset(cur_sums)) is too big.
    # Just (tuple(seq)) might be ambiguous if sums differ? No, sums are determined by seq.
    # So key = tuple(seq).
    
    key = tuple(seq)
    if key in dead_ends:
        return best_len, best_seq
    
    # Bound check
    ub = _upper_bound_length(seq, cur_sums, max_val)
    if ub <= best_len:
        dead_ends.add(key)
        return best_len, best_seq
    
    # Try to extend
    last_val = seq[-1] if seq else 0
    extended = False
    
    # Sort candidates? Smaller values generally allow more extensions.
    # We iterate from smallest to largest to find a good solution quickly (depth-first bias towards valid small steps).
    for x in range(last_val + 1, max_val + 1):
        if time.time() > deadline:
            return best_len, best_seq
            
        if _valid_extension(seq, x, cur_sums):
            # Create new state
            new_seq = seq + [x]
            # Update sums efficiently
            new_sums = set(cur_sums)
            tails = [x]
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
            new_sums.update(tails)
            
            extended = True
            
            # Recursive call
            best_len, best_seq = _dfs_csp(new_seq, new_sums, max_val, deadline, best_len, best_seq, dead_ends)
            
            # If we found a solution that beats the upper bound of the parent (unlikely but possible if bound was loose), 
            # or if we just want to explore more, we continue.
            # Since we want MAX length, and we are doing DFS, we should keep exploring other x values
            # unless we hit the absolute theoretical max for this branch.
            
            # Optimization: If best_len == ub (the upper bound of the *parent* was ub, but now we are deeper),
            # we can't easily prune here without recalculating.
            # However, if we found a sequence of length == max_val (impossible) or very close, we can stop.
            # A better pruning: If best_len >= _upper_bound_length(new_seq, new_sums, max_val), 
            # then no extension of new_seq can beat best_len. But we already returned from the recursion.
            # The recursion itself would have pruned further.
            
            # We continue to try larger x to see if a different path yields a longer sequence.
    
    if not extended:
        # Dead end
        dead_ends.add(key)
        
    return best_len, best_seq

def construct():
    start_time = time.time()
    # Safety margin: 60s limit, stop at 50s to allow for cleanup/return
    deadline = start_time + 50
    
    max_val = 256
    
    # 1. Get a good initial lower bound using Greedy
    # Greedy is fast and gives a solid baseline.
    greedy_seq = _greedy_forward([], max_val)
    best_len = len(greedy_seq)
    best_seq = greedy_seq
    best_sums = _compute_sums(greedy_seq)
    
    # 2. Run DFS with Branch and Bound starting from the empty sequence
    # We pass the greedy result as the initial best to prune early.
    dead_ends = set()
    
    # To make DFS effective, we should start from scratch but use the greedy bound.
    # However, DFS from scratch might be slow if it explores bad branches first.
    # We can use a "beam" of starting points or just rely on the pruning.
    # Given the bound is simple, let's try DFS from empty.
    
    # Optimization: If greedy found a long sequence, the bound `len + (256 - last)` is quite tight.
    # For example, if greedy ends at 200 with length 10, bound is 10 + 56 = 66.
    # If greedy length is 10, and we explore a branch that ends at 100 with length 5, bound is 5 + 156 = 161.
    # The DFS will explore until it proves it can't beat 10.
    
    # We will run DFS from the empty set.
    # To handle time, we break out if time runs out.
    
    # Note: _dfs_csp uses global time check.
    
    try:
        best_len, best_seq = _dfs_csp([], set(), max_val, deadline, best_len, best_seq, dead_ends)
    except RecursionError:
        pass # Fallback to best found so far
    
    # 3. If time remains, try to improve from the best sequence found by DFS
    # Sometimes DFS finds a local max due to pruning or ordering.
    # We can try a local search (ILS) starting from the best DFS result.
    
    if time.time() < deadline - 5 and best_seq:
        # Simple local search: try to swap/replace elements in the best_seq to find a longer one?
        # Actually, the DFS should have covered this if the bound was tight.
        # Let's try a different strategy: Restart DFS with a different ordering or seed?
        # Or just run another DFS from a slightly perturbed greedy sequence?
        
        # Let's try to extend the best_seq found by DFS using a randomized greedy approach
        # to see if we can push it further (in case DFS pruned too aggressively due to time).
        # But DFS prunes based on bound, not just time. If it returned, it's either a dead end or time out.
        
        # If time out, best_seq is the best found so far.
        # We can try to "shake" it: remove last element, try a different next one.
        
        current = list(best_seq)
        cur_sums = _compute_sums(current)
        
        import random
        rng = random.Random(99)
        
        for _ in range(20): # Limited iterations
            if time.time() > deadline:
                break
            if not current:
                break
            # Remove last element
            last = current.pop()
            cur_sums = _compute_sums(current)
            
            # Try a random valid extension different from 'last'
            last_val = current[-1] if current else 0
            candidates = []
            for x in range(last_val + 1, max_val + 1):
                if x != last and _valid_extension(current, x, cur_sums):
                    candidates.append(x)
            
            if not candidates:
                current.append(last)
                cur_sums = _compute_sums(current)
                continue
            
            # Pick a random candidate
            x = rng.choice(candidates)
            current.append(x)
            cur_sums = _compute_sums(current)
            
            # Try to extend greedily from here
            ext_seq = _greedy_forward(current, max_val)
            if len(ext_seq) > best_len:
                best_len = len(ext_seq)
                best_seq = ext_seq
    
    # Final validation
    if not _is_valid_full(best_seq):
        # Fallback to greedy
        best_seq = _greedy_forward([], max_val)
        
    return best_seq

# EVOLVE-BLOCK-END
