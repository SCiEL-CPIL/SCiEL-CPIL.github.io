import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _dfs_csp(max_val=256, deadline=None, initial_seq=None, initial_sums=None):
    """
    DFS with Constraint Propagation (Forward Checking).
    
    State:
    - seq: list of chosen integers
    - sums: set of all consecutive block sums so far
    - last: last element in seq
    
    Heuristic:
    - Try candidates in increasing order (smallest first). This tends to keep
      the sequence values small, leaving more room for more elements later.
    - If a branch fails, backtrack.
    
    Optimization:
    - Pre-compute potential tails for each candidate x given the current seq.
    - Check intersection with sums immediately.
    """
    if deadline is None:
        deadline = time.time() + 50
        
    best_seq = []
    best_len = 0
    
    # Initialize state
    if initial_seq is None:
        seq = []
        sums = set()
        last = 0
    else:
        seq = list(initial_seq)
        sums = set(initial_sums)
        last = seq[-1] if seq else 0
        
    # Update best
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = list(seq)
        
    # Recursion stack to avoid Python recursion limit issues
    # Stack items: (seq, sums, last)
    # Since we want to maximize length, we do a depth-first search.
    # However, pure DFS might get stuck in a local optimum if it doesn't backtrack enough.
    # We will implement a standard backtracking DFS.
    
    # To manage time, we use a global counter or time check.
    # Python recursion depth is limited to ~1000 by default, which is fine for length < 100.
    # But let's use an iterative approach or increase recursion limit?
    # Max length is likely < 20. Recursion is safe.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(seq, sums, last):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() > deadline:
            return
            
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
        # If we can't add more (last >= max_val), return
        if last >= max_val:
            return
            
        # Try candidates from last+1 to max_val
        # Heuristic: Smaller candidates are generally better for maximizing length
        # because they leave more "sum space" and "value space" for future elements.
        
        # Optimization: We can skip candidates that are clearly worse?
        # No, we must try them to ensure optimality (or near-optimality).
        
        for x in range(last + 1, max_val + 1):
            # Check validity
            tails = [x]
            total = x
            valid = True
            # Compute tails and check collision
            for a in reversed(seq):
                total += a
                tails.append(total)
                if total in sums:
                    valid = False
                    break
            if x in sums:
                valid = False
            
            if valid:
                # Create new state
                new_sums = set(sums)
                new_sums.update(tails)
                new_seq = seq + [x]
                
                # Recurse
                dfs(new_seq, new_sums, x)
                
                # If we found a very long sequence, we might stop early?
                # No, we want the global max. But if time is tight, we might prune.
                # For now, full backtracking.
                
        # Optional: If we found a solution of length L, and we are at depth L-1, 
        # and we've tried the smallest candidates, maybe we can prune larger ones?
        # Not strictly correct without proof. Keep it simple.

    dfs(seq, sums, last)
    
    return best_seq, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    best = []
    best_len = 0
    
    # 1. Get a good initial lower bound using Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
        
    # 2. Run DFS CSP
    # The DFS explores the space of sequences.
    # The "smallest first" heuristic is crucial.
    # We run it from the start.
    
    found_seq, found_len = _dfs_csp(max_val=256, deadline=deadline)
    if found_len > best_len:
        best = found_seq
        best_len = found_len
        
    # 3. If time remains, try randomized restarts or different heuristics?
    # DFS with "smallest first" is deterministic.
    # We can try "randomized DFS" to escape local optima if the first run didn't find the global max.
    # However, for this specific problem (increasing, distinct block sums),
    # the "smallest valid" path is often very strong because it minimizes the growth of sums.
    
    # Let's add a randomized variant if time allows.
    if time.time() < deadline - 5:
        rng = random.Random(42)
        
        def _dfs_randomized(max_val=256, deadline=None, initial_seq=None, initial_sums=None):
            """DFS with randomized candidate ordering."""
            if deadline is None:
                deadline = time.time() + 50
                
            best_seq = []
            best_len = 0
            
            if initial_seq is None:
                seq = []
                sums = set()
                last = 0
            else:
                seq = list(initial_seq)
                sums = set(initial_sums)
                last = seq[-1] if seq else 0
                
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = list(seq)
                
            if last >= max_val:
                return best_seq, best_len
                
            import sys
            sys.setrecursionlimit(10000)
            
            def dfs(seq, sums, last):
                nonlocal best_seq, best_len
                
                if time.time() > deadline:
                    return
                    
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = list(seq)
                    
                if last >= max_val:
                    return
                    
                # Generate valid candidates
                candidates = []
                for x in range(last + 1, max_val + 1):
                    tails = [x]
                    total = x
                    valid = True
                    for a in reversed(seq):
                        total += a
                        tails.append(total)
                        if total in sums:
                            valid = False
                            break
                    if x in sums:
                        valid = False
                    if valid:
                        candidates.append((x, tails))
                        
                # Shuffle candidates to break determinism
                rng.shuffle(candidates)
                
                # Sort by x to keep some structure? Or pure random?
                # Pure random might explore bad branches.
                # Let's sort by x but with a random tie-breaker or just randomize the order of checking.
                # Actually, for this problem, smaller x is usually better.
                # Randomizing might be counter-productive.
                # Let's try a "weighted" random: prefer smaller x but allow larger.
                # Or just run the deterministic one multiple times? No, it's deterministic.
                
                # Let's stick to the deterministic "smallest first" as it is theoretically stronger for this packing problem.
                # The randomization is more useful for finding *different* valid sequences if we were optimizing something else.
                # For maximizing length, the "tightest" packing (smallest values) is usually optimal.
                
                # So, skip randomized DFS if it's not clearly better.
                # Instead, let's try a beam search with a very specific heuristic if DFS is too slow?
                # DFS is exponential. Beam is polynomial but heuristic.
                # The previous program used Beam. This one uses DFS.
                # If DFS is too slow, it will timeout and return the best found so far (greedy).
                # To ensure we don't timeout, we rely on the pruning.
                
                for x, tails in candidates:
                    new_sums = set(sums)
                    new_sums.update(tails)
                    new_seq = seq + [x]
                    dfs(new_seq, new_sums, x)
                    
            dfs(seq, sums, last)
            return best_seq, best_len

        # Run a few randomized searches? 
        # Actually, the deterministic DFS with "smallest first" is the most likely to find the longest sequence 
        # because it minimizes the sum growth. Randomizing the order of x likely leads to larger sums faster, 
        # reducing the maximum possible length. 
        # So we stick to the deterministic DFS.
        
        # However, if the deterministic DFS gets stuck in a local optimum (which is unlikely with full backtracking 
        # unless it times out), we might want to try a different start.
        # But full backtracking from empty is exhaustive (within time limits).
        
    # Final validation
    if not _is_valid_full(best):
        # Fallback
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
