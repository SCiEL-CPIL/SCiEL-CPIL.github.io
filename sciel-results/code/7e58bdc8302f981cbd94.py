import time
import random

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
    n = len(seq)
    expected = n * (n + 1) // 2
    if len(sums) != expected:
        return False
    return True

def _greedy_build(start_value, max_time=5.0, strategy="smallest"):
    """
    Build a sequence starting with start_value using a greedy strategy.
    strategy: "smallest", "largest", "random"
    """
    start_time = time.time()
    seq = [start_value]
    sums_set = {start_value}
    last = start_value
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        n = len(seq)
        # Compute suffix sums for conflict check
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        valid_next = []
        for x in range(last + 1, 257):
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            if not conflict:
                valid_next.append(x)
        
        if not valid_next:
            break
        
        if strategy == "smallest":
            best_x = valid_next[0]
        elif strategy == "largest":
            best_x = valid_next[-1]
        else: # random
            best_x = random.choice(valid_next)
        
        seq.append(best_x)
        # Update sums
        for k in range(n + 1):
            sums_set.add(suffix_sums[k] + best_x)
        last = best_x
        
    return seq

def construct():
    """
    Main entry point. Returns a strictly increasing sequence in [1,256] with distinct consecutive-block sums.
    Uses a combination of greedy initialization and DFS with backtracking and pruning.
    """
    total_start = time.time()
    deadline = total_start + 55.0  # Safety margin
    
    best_seq = []
    best_len = 0
    
    # 1. Initial Greedy Passes to get a good lower bound
    # Try starting from 1, 2, 3... with different strategies
    for start_val in range(1, 20):
        if time.time() > deadline - 5:
            break
        for strategy in ["smallest", "largest", "random"]:
            seq = _greedy_build(start_val, max_time=1.0, strategy=strategy)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
    
    # 2. DFS with Backtracking to improve
    # We use a global best_len for pruning
    def dfs(seq, sums_set, last_val, depth):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        # Pruning: Even if we add all remaining possible numbers, can we beat best_len?
        # Max additional elements is 256 - last_val
        if len(seq) + (256 - last_val) <= best_len:
            return
            
        # If we are already very long, we might want to stop expanding?
        # No, we want to maximize length.
        
        n = len(seq)
        # Compute suffix sums
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        # Generate candidates
        # To explore different branches, we can randomize the order of candidates
        # But for pruning efficiency, smaller x might be better? Or larger?
        # Let's try both: one pass with ascending, one with descending?
        # Or just randomize.
        
        candidates = list(range(last_val + 1, 257))
        
        # Heuristic: Try candidates that are "compatible" with many future choices?
        # Too expensive.
        
        # Shuffle to escape local optima
        random.shuffle(candidates)
        
        for x in candidates:
            # Check validity
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Recurse
            new_seq = seq + [x]
            new_sums = sums_set | {suffix_sums[k] + x for k in range(n + 1)}
            
            dfs(new_seq, new_sums, x, depth + 1)
            
            # Early exit if time is up
            if time.time() > deadline:
                return
            
            # If we found a solution that reached the theoretical max or close to it, we can stop?
            # Theoretical max is not known, but if len(seq) + (256-x) <= best_len, the recursion would have pruned.
            
    # Start DFS from the best greedy solution found?
    # Or from scratch?
    # Starting from scratch allows finding a completely different structure.
    # But starting from a good seed is faster.
    # Let's start DFS from an empty sequence, but the pruning will be weak initially.
    # Better: Start DFS from the best_seq found in step 1?
    # No, that only explores extensions of that specific sequence.
    # We want to find a *different* sequence that might be longer.
    
    # Let's run DFS from scratch, but with a time limit.
    # To make it effective, we need the initial best_len to be high.
    
    if time.time() < deadline - 2:
        dfs([], set(), 0, 0)
        
    # Verify result
    if not _check_valid(best_seq):
        # Fallback
        best_seq = [1, 2, 5]
        
    return best_seq
