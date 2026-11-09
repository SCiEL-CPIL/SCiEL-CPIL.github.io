import time
import random
import sys

START_TIME = time.time()
DEADLINE = 90.0

def elapsed():
    return time.time() - START_TIME

def backtrack_search(s, bitsets, candidates, target_size, deadline_margin=5.0):
    """Backtracking search with forward checking to find a set of target_size.
    
    Args:
        s: current list of selected integers
        bitsets: current cardinality-indexed bitsets
        candidates: list of remaining candidate integers to try
        target_size: desired final size
        deadline_margin: safety margin in seconds
    
    Returns:
        A valid set of size >= target_size, or None if not found within time.
    """
    n = len(s)
    if n >= target_size:
        return s[:]
    
    if elapsed() > DEADLINE - deadline_margin:
        return None
    
    # Try adding each candidate
    for i, val in enumerate(candidates):
        if elapsed() > DEADLINE - deadline_margin:
            return None
        
        # Forward check: can we add val without creating cross-cardinality collisions?
        new_bitsets = try_add(s, bitsets, val)
        if new_bitsets is not None:
            # Recursively try to complete the set
            remaining = candidates[i+1:]
            result = backtrack_search(s + [val], new_bitsets, remaining, target_size, deadline_margin)
            if result is not None:
                return result
    
    return None

def backtrack_from_seed(seed_set, target_size, deadline_margin=5.0):
    """Start backtracking from a seed set and try to reach target_size.
    
    Args:
        seed_set: initial valid set
        target_size: desired final size
        deadline_margin: safety margin in seconds
    
    Returns:
        A valid set of size >= target_size, or the seed_set if no improvement found.
    """
    seed_set = list(seed_set)
    bitsets = build_bitsets(seed_set)
    used = set(seed_set)
    
    # Generate candidates not in seed_set, sorted by "promisingness"
    # Larger values tend to be more separable
    candidates = [v for v in range(1, 257) if v not in used]
    candidates.sort(reverse=True)  # Try larger values first for better separation
    
    result = backtrack_search(seed_set, bitsets, candidates, target_size, deadline_margin)
    if result is not None:
        return result
    return seed_set

def build_bitsets(s):
    """Build cardinality-indexed bitset of subset sums.
    Returns list of ints where bitset[k] has bit j set iff sum j is achievable with k elements.
    """
    n = len(s)
    bitsets = [0] * (n + 1)
    bitsets[0] = 1  # bit 0 set: empty sum
    for val in s:
        # Update in reverse to avoid using same element twice
        for k in range(n, 0, -1):
            bitsets[k] |= (bitsets[k - 1] << val)
    return bitsets

def is_valid_bitset(bitsets):
    """Check that no two cardinality levels share a sum."""
    n = len(bitsets) - 1
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if bitsets[i] & bitsets[j]:
                return False
    return True

def try_add(s, bitsets, val):
    """Check if adding val to s keeps it valid. Returns new bitsets if valid, else None."""
    n = len(s)
    n_new = n + 1
    new_bitsets = [0] * (n_new + 1)
    new_bitsets[0] = bitsets[0]
    for k in range(n_new, 0, -1):
        if k - 1 <= n:
            new_bitsets[k] |= (bitsets[k - 1] << val)
        if k <= n:
            new_bitsets[k] |= bitsets[k]
    
    # Check validity
    for i in range(n_new + 1):
        for j in range(i + 1, n_new + 1):
            if new_bitsets[i] & new_bitsets[j]:
                return None
    return new_bitsets

def try_swap(s, bitsets, remove_val, add_val):
    """Check if swapping remove_val with add_val in s keeps it valid."""
    # Remove remove_val
    # This is effectively building bitsets for s without remove_val
    # Then adding add_val
    s_rem = [x for x in s if x != remove_val]
    rem_bitsets = build_bitsets(s_rem)
    
    return try_add(s_rem, rem_bitsets, add_val)

def local_search(s, deadline_margin=5.0):
    """Iterative local search to maximize set size."""
    s = list(s)
    n = len(s)
    bitsets = build_bitsets(s)
    used = set(s)
    
    improved = True
    while improved:
        if elapsed() > DEADLINE - deadline_margin:
            break
        improved = False
        
        # 1. Try adding
        for val in range(1, 257):
            if elapsed() > DEADLINE - deadline_margin:
                break
            if val in used:
                continue
            new_bitsets = try_add(s, bitsets, val)
            if new_bitsets is not None:
                s.append(val)
                used.add(val)
                bitsets = new_bitsets
                n += 1
                improved = True
                break # Restart loop with new size
        
        # 2. Try swapping if no addition found
        if not improved:
            # Try all pairs (remove, add)
            for i in range(n):
                if elapsed() > DEADLINE - deadline_margin:
                    break
                remove_val = s[i]
                rem_bitsets = build_bitsets([x for j, x in enumerate(s) if j != i])
                
                for val in range(1, 257):
                    if elapsed() > DEADLINE - deadline_margin:
                        break
                    if val in used:
                        continue
                    
                    new_bitsets = try_add([x for j, x in enumerate(s) if j != i], rem_bitsets, val)
                    if new_bitsets is not None:
                        # Perform swap
                        s[i] = val
                        used.remove(remove_val)
                        used.add(val)
                        bitsets = new_bitsets
                        improved = True
                        break # Restart loop
                if improved:
                    break
    return s

def greedy_construct(order, deadline_margin=5.0):
    """Greedily build a valid set by adding elements in the given order."""
    s = []
    bitsets = [1]
    for val in order:
        if elapsed() > DEADLINE - deadline_margin:
            break
        new_bitsets = try_add(s, bitsets, val)
        if new_bitsets is not None:
            s.append(val)
            bitsets = new_bitsets
    return s

def construct():
    if elapsed() > DEADLINE - 10.0:
        return list(range(226, 257)) # Fallback size 31
    
    # Initial set from greedy
    candidates = list(range(1, 257))
    
    # Try a few orders
    best = []
    best_size = 0
    
    # Descending
    s = greedy_construct(sorted(candidates, reverse=True), deadline_margin=10.0)
    if len(s) > best_size:
        best = s
        best_size = len(s)
        
    # Ascending
    s = greedy_construct(sorted(candidates), deadline_margin=10.0)
    if len(s) > best_size:
        best = s
        best_size = len(s)
        
    # Random orders
    random.seed(42)
    for _ in range(20):
        if elapsed() > DEADLINE - 20.0:
            break
        random.shuffle(candidates)
        s = greedy_construct(candidates, deadline_margin=10.0)
        if len(s) > best_size:
            best = s
            best_size = len(s)
            
    # Local search
    if elapsed() < DEADLINE - 15.0:
        improved = local_search(best, deadline_margin=10.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Backtracking search to try to exceed current best
    if elapsed() < DEADLINE - 15.0 and best_size < 60:
        # Try to reach best_size + 1, then +2, etc.
        for target in range(best_size + 1, min(best_size + 5, 61)):
            if elapsed() > DEADLINE - 15.0:
                break
            result = backtrack_from_seed(best, target, deadline_margin=12.0)
            if result is not None and len(result) > best_size:
                best = result
                best_size = len(best)
                # Update bitsets for further backtracking
                # Continue trying to add more
    
    # Final validation
    if not is_valid_bitset(build_bitsets(best)):
        return list(range(226, 257))
        
    return [int(x) for x in best]
