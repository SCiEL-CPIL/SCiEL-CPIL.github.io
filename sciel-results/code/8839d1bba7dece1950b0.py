"""Snake-in-the-box in Q_9: Greedy seed + MCV Backtracking Extension."""
import time
import random
import numpy as np

N = 9
FULL = (1 << N) - 1
# Safety margin: evaluator kills at 150s. We stop at 140s to be safe.
DEADLINE = 140.0

# Precompute single-bit masks
MASKS = [1 << b for b in range(N)]

# Precompute bit counts for all 512 values
BC = np.zeros(1 << N, dtype=np.uint8)
for v in range(1 << N):
    BC[v] = v.bit_count()

def _is_valid_extend(v, path, used):
    """
    Check if v can be appended to path.
    v must be unused.
    v must differ by >= 2 bits from all vertices in path except the last one.
    """
    if v in used:
        return False
    
    # Vectorized check against path[:-1]
    if len(path) >= 2:
        # Convert path[:-1] to numpy array for fast XOR and bit count lookup
        # We can't easily vectorize the bit_count of XOR directly without 
        # precomputing a full 512x512 table or using numpy's ability to handle 
        # integer operations. 
        # Let's use a precomputed table approach or just numpy array ops.
        # Since N=9, path length is at most 512. 
        # Efficient way: 
        # diffs = [ (v ^ p).bit_count() for p in path[:-1] ]
        # if any is 1, invalid.
        
        # Using NumPy:
        # Create array of path[:-1]
        arr = np.array(path[:-1], dtype=np.uint16)
        # XOR v with all
        xors = (v ^ arr).astype(np.intp)
        # Get bit counts from precomputed BC
        counts = BC[xors]
        if np.any(counts == 1):
            return False
            
    return True

def _count_future_options(v, path, used):
    """
    Count how many valid neighbors v has that are not in used.
    Used for MCV heuristic.
    """
    cnt = 0
    # v's neighbors
    for m in MASKS:
        w = v ^ m
        if w in used:
            continue
        # Check if w is valid extension from v, given current path
        # Note: w is neighbor of v. 
        # w must differ >= 2 from path[:-1] (excluding v, which is new last)
        # Actually, the condition is:
        # 1. w not in used
        # 2. w differs >= 2 from all p in path[:-1]
        # 3. w differs 1 from v (guaranteed by construction)
        
        if len(path) >= 1:
             # We check against path[:-1] because v is the new last element.
             # The constraint is on non-consecutive pairs.
             # (v, w) are consecutive.
             # (p, w) must have dist >= 2 for p in path[:-1].
             # (p, v) is already valid.
             
             if len(path) >= 2:
                 arr = np.array(path[:-1], dtype=np.uint16)
                 xors = (w ^ arr).astype(np.intp)
                 counts = BC[xors]
                 if np.any(counts == 1):
                     continue
             # If path length is 1, path[:-1] is empty, so no conflict check needed for w vs older nodes.
             
        cnt += 1
    return cnt

def _greedy_extend(path, used, bit_order):
    """
    Greedy extension. Returns path and used.
    """
    while True:
        cur = path[-1]
        found = False
        for b in bit_order:
            v = cur ^ MASKS[b]
            if _is_valid_extend(v, path, used):
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path, used

def _mcv_backtrack(path, used, deadline):
    """
    DFS with MCV heuristic.
    Returns the best path found.
    """
    best_path = list(path)
    best_len = len(path)
    
    # To avoid excessive recursion depth issues or stack overflow, 
    # we use a recursive function but Python's default limit is 1000.
    # Max path length is 512, so depth is fine.
    
    def dfs(current_path, current_used):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return
        
        cur = current_path[-1]
        
        # Generate valid candidates
        cands = []
        for m in MASKS:
            v = cur ^ MASKS[0] # placeholder, will overwrite
            v = cur ^ m
            if _is_valid_extend(v, current_path, current_used):
                cands.append(v)
                
        if not cands:
            if len(current_path) > best_len:
                best_len = len(current_path)
                best_path = list(current_path)
            return
            
        # MCV Heuristic: Sort candidates by number of future options (ascending)
        # This forces the search to try the most constrained moves first.
        # This often leads to dead ends quickly, which is good for backtracking.
        
        # To speed up sorting, we compute the count for each candidate.
        # This is expensive, so we only do it if the number of candidates is small?
        # Or always? N=9, max 9 candidates. Counting options for 9 candidates is cheap.
        
        scored_cands = []
        for v in cands:
            # Count future options for v
            # We pass current_path and current_used. 
            # Note: when we extend to v, the "path" for the next step's conflict check 
            # will be current_path + [v]. 
            # So for v's neighbors w, we check w against current_path[:-1].
            # This is exactly what _count_future_options does if we pass current_path.
            # Wait, _count_future_options(v, path, used) checks v's neighbors against path[:-1].
            # If we append v to path, the new path is P' = P + [v].
            # The next node w must be valid against P'[ :-1 ] = P.
            # So we should count options for v against P.
            # My _count_future_options(v, path, used) uses path[:-1].
            # If I call _count_future_options(v, current_path, current_used), 
            # it checks v's neighbors against current_path[:-1].
            # But if v is added, the "previous" nodes are current_path.
            # So the conflict check for the NEXT node w (neighbor of v) is against current_path.
            # So I need to count options for v assuming v is the new last.
            # The "history" for conflict checking w is current_path.
            # So I should call a function that counts valid neighbors of v 
            # given that the history is current_path.
            
            # Let's refine _count_future_options to take the "history" explicitly or adjust logic.
            # Current _count_future_options(v, path, used) checks neighbors of v against path[:-1].
            # If I want to know how many options v has *after* being added to current_path,
            # the history for the *next* step will be current_path.
            # So the neighbors w of v must be valid against current_path.
            # So I should compute: count of w in neighbors(v) such that w not in used AND w valid against current_path.
            
            cnt = 0
            for m in MASKS:
                w = v ^ m
                if w in current_used:
                    continue
                # Check w against current_path
                if len(current_path) >= 1:
                    # w must differ >= 2 from all nodes in current_path
                    # Note: v is not in current_path yet.
                    # But w is neighbor of v.
                    # If current_path is empty, no check.
                    if len(current_path) > 0:
                        arr = np.array(current_path, dtype=np.uint16)
                        xors = (w ^ arr).astype(np.intp)
                        counts = BC[xors]
                        if np.any(counts == 1):
                            continue
                cnt += 1
            scored_cands.append((cnt, v))
            
        # Sort by count (ascending)
        scored_cands.sort(key=lambda x: x[0])
        
        for cnt, v in scored_cands:
            if time.time() > deadline:
                return
            
            new_path = current_path + [v]
            new_used = current_used | {v}
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = list(new_path)
                
            dfs(new_path, new_used)
            
            if time.time() > deadline:
                return

    dfs(path, used)
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_path = []
    best_len = 0
    
    # 1. Generate a good initial greedy path
    # Try a few deterministic bit orders
    bit_orders = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        [0, 2, 4, 6, 8, 1, 3, 5, 7],
        [8, 6, 4, 2, 0, 7, 5, 3, 1],
    ]
    
    initial_paths = []
    for order in bit_orders:
        if time.time() > deadline - 10: # Leave time for backtracking
            break
        path, used = _greedy_extend([0], {0}, order)
        initial_paths.append((len(path), path, used))
        
    # Sort initial paths by length
    initial_paths.sort(key=lambda x: x[0], reverse=True)
    
    if initial_paths:
        best_len = initial_paths[0][0]
        best_path = initial_paths[0][1]
    
    # 2. Apply MCV Backtracking to the best initial paths
    # We might want to try backtracking on the top few initial paths if time allows
    for length, path, used in initial_paths[:2]: # Top 2
        if time.time() > deadline - 5:
            break
        
        # We use a slightly earlier deadline for the DFS to ensure we return in time
        # The DFS itself checks time, but we want to stop spawning new DFS if time is up.
        # Let's use the main deadline.
        
        extended_path = _mcv_backtrack(path, used, deadline)
        
        if len(extended_path) > best_len:
            best_len = len(extended_path)
            best_path = extended_path
            
        # If we found a very long path, maybe stop early?
        # In Q9, max is unknown but >190. 
        # If we hit 200+, it's good.
        # But let's try to maximize.
        
    # 3. Randomized restarts with backtracking if time remains
    # This helps escape local optima of the greedy start.
    rng = random.Random(42)
    for i in range(10):
        if time.time() > deadline - 5:
            break
            
        order = list(range(N))
        rng.shuffle(order)
        
        path, used = _greedy_extend([0], {0}, order)
        
        # Only backtrack if this greedy path is promising or if we have lots of time
        # Backtracking is expensive.
        if time.time() < deadline - 10:
            extended_path = _mcv_backtrack(path, used, deadline)
            if len(extended_path) > best_len:
                best_len = len(extended_path)
                best_path = extended_path
        else:
            if len(path) > best_len:
                best_len = len(path)
                best_path = path

    # Final validation
    if not _validate(best_path):
        # Fallback to simple greedy if something went wrong
        path, used = _greedy_extend([0], {0}, list(range(N)))
        best_path = path

    return best_path

def _validate(path):
    """Sanity check the induced-path property."""
    if len(path) < 2:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print("length:", len(res), "edges:", len(res) - 1, "valid:", _validate(res), "time:", time.time() - t0)
