import time
import random
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def _can_append(seq, x, sums_set):
    """
    Check if x can be appended to seq.
    Returns (True, new_sums_set) if valid, else (False, None).
    """
    if not seq:
        if x < 1 or x > 256:
            return False, None
        if x in sums_set: # sums_set is empty initially, so this is fine
            return False, None
        return True, {x}
    
    if x <= seq[-1]:
        return False, None
    if x > 256:
        return False, None
        
    # Check x itself
    if x in sums_set:
        return False, None
        
    # Check sums ending at x
    # These are sum(seq[i:]) + x
    # We compute them in reverse order
    suffix = 0
    new_sums = [x] # The sum of just x
    for i in range(len(seq) - 1, -1, -1):
        suffix += seq[i]
        ns = suffix + x
        if ns in sums_set:
            return False, None
        new_sums.append(ns)
        
    # new_sums are distinct from each other (proven earlier)
    # and distinct from sums_set (checked)
    
    # Create new set
    # To be efficient, we can update the set
    # But for simplicity and safety, create a new set or update in place carefully?
    # Since we are backtracking, it's safer to return a new set or copy.
    # Given small N, copying is fine.
    new_sums_set = set(sums_set)
    new_sums_set.update(new_sums)
    
    return True, new_sums_set

def _get_initial_sums(seq):
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _is_valid(seq):
    if not seq:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(len(seq) - 1):
        if seq[i + 1] <= seq[i]:
            return False
    sums = _get_initial_sums(seq)
    if len(sums) != len(seq) * (len(seq) + 1) // 2:
        return False
    return True

def _dfs_extend(seq, sums_set, deadline, best_seq, best_len_holder):
    """
    Depth-first search to extend seq.
    best_len_holder is a list [current_best_len] to allow modification in nested scope.
    """
    if time.time() > deadline:
        return
        
    current_len = len(seq)
    if current_len > best_len_holder[0]:
        best_len_holder[0] = current_len
        best_seq[:] = seq
        
    # If we have reached a reasonable length, we might stop deep diving?
    # No, we want max length.
    
    last = seq[-1] if seq else 0
    # Generate candidates
    # To avoid trying too many small numbers that might block future large numbers,
    # we can bias towards larger numbers? Or just random.
    # Let's try random sampling of candidates.
    candidates = list(range(last + 1, 257))
    if not candidates:
        return
        
    random.shuffle(candidates)
    
    # Limit branching factor per node to avoid explosion?
    # If we try all, it's exhaustive.
    # Let's try all, but the time limit will cut it off.
    # To make it faster, we can prioritize candidates that are "far" from existing sums?
    # For now, random order is a good unbiased heuristic.
    
    for x in candidates:
        if time.time() > deadline:
            return
        ok, new_sums = _can_append(seq, x, sums_set)
        if ok:
            seq.append(x)
            _dfs_extend(seq, new_sums, deadline, best_seq, best_len_holder)
            seq.pop()

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial known good sequence
    # Powers of 2 up to 128, plus 256
    # 1, 2, 4, 8, 16, 32, 64, 128, 256
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if not _is_valid(best_seq):
        # Fallback to simple valid seq
        best_seq = [1, 3, 7, 15]
        
    best_len_holder = [len(best_seq)]
    
    # Run multiple restarts
    # Each restart tries to extend the current best or a perturbed version
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0:
        # Perturb the current best sequence to escape local optima
        # Remove a random element
        seq_copy = list(best_seq)
        
        # With some probability, restart from a smaller core
        if rng.random() < 0.2 and len(seq_copy) > 5:
            # Keep first half
            seq_copy = seq_copy[:len(seq_copy)//2]
        
        # Remove a random element from the rest
        if len(seq_copy) > 1:
            idx = rng.randint(0, len(seq_copy) - 1)
            seq_copy.pop(idx)
        
        sums_set = _get_initial_sums(seq_copy)
        
        # Run DFS to extend
        _dfs_extend(seq_copy, sums_set, deadline, best_seq, best_len_holder)
        
        # If we found a new best, we can continue.
        # The best_seq is updated in place by _dfs_extend if it finds something longer.
        
    return best_seq

if __name__ == "__main__":
    # Quick self-test
    seq = construct()
    print(f"Length: {len(seq)}")
    print(f"Seq: {seq}")
    assert _is_valid(seq), "Invalid sequence generated"
