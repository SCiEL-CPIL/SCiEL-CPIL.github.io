# EVOLVE-BLOCK-START

def _is_sidon_and_cover(a, required_mask):
    """
    Check if sorted list 'a' is a Sidon set.
    Returns (is_sidon, diffs_mask, max_elem).
    diffs_mask contains bits for all positive differences.
    """
    n = len(a)
    diffs_mask = 0
    for i in range(n):
        ai = a[i]
        for j in range(i + 1, n):
            d = a[j] - ai
            bit = 1 << d
            if diffs_mask & bit:
                return False, diffs_mask, a[-1]
            diffs_mask |= bit
    return True, diffs_mask, a[-1]

def _covers_all(diffs_mask, required_mask):
    return (diffs_mask & required_mask) == required_mask

def _backtrack_minimal(deadline, best_diam_ref):
    """
    Backtracking to find minimal diameter Sidon set covering 1..30.
    best_diam_ref is a list [current_best_diam] to allow updating.
    """
    required_mask = (1 << 31) - 2 # bits 1..30
    best_set = [None]
    
    def rec(a, diffs_mask, current_max):
        if time.time() > deadline:
            return
        
        # Pruning: if we already cover everything, update best
        if _covers_all(diffs_mask, required_mask):
            if current_max < best_diam_ref[0]:
                best_diam_ref[0] = current_max
                best_set[0] = list(a)
            return
        
        # Pruning: if current_max >= best known, stop
        if current_max >= best_diam_ref[0]:
            return
        
        n = len(a)
        if n >= 12: # Max size constraint is 32, but usually smaller is better for Sidon
            return

        # Determine next element range
        start = a[-1] + 1 if a else 1
        # Upper bound: we can't exceed best_diam - 1
        upper = min(best_diam_ref[0] - 1, 100)
        
        for x in range(start, upper + 1):
            if time.time() > deadline:
                return
            
            # Check if x creates a conflict
            new_diffs_mask = 0
            conflict = False
            for v in a:
                d = x - v
                bit = 1 << d
                if (diffs_mask & bit) or (new_diffs_mask & bit):
                    conflict = True
                    break
                new_diffs_mask |= bit
            
            if conflict:
                continue
            
            # Recurse
            a.append(x)
            rec(a, diffs_mask | new_diffs_mask, x)
            a.pop()

    # Initial call
    a = [0]
    rec(a, 0, 0)
    
    if best_set[0]:
        return best_set[0]
    return None

def _greedy_construct(deadline):
    """
    Greedy construction: pick next element that maximizes coverage of missing diffs.
    """
    required_mask = (1 << 31) - 2
    a = [0]
    diffs_mask = 0
    current_max = 0
    
    while not _covers_all(diffs_mask, required_mask):
        if time.time() > deadline:
            break
        
        best_x = None
        best_new_mask = 0
        best_score = -1
        
        # Search for next element
        for x in range(1, 100):
            if x in a:
                continue
            
            new_diffs_mask = 0
            conflict = False
            for v in a:
                d = x - v
                if d <= 0: continue
                bit = 1 << d
                if (diffs_mask & bit) or (new_diffs_mask & bit):
                    conflict = True
                    break
                new_diffs_mask |= bit
            
            if conflict:
                continue
            
            # Score: number of new required diffs covered
            new_required = new_diffs_mask & required_mask
            newly_covered = new_required & ~diffs_mask
            score = bin(newly_covered).count('1')
            
            # Tie-breaker: smaller x is better (keeps diameter low)
            if score > best_score or (score == best_score and (best_x is None or x < best_x)):
                best_score = score
                best_x = x
                best_new_mask = new_diffs_mask
        
        if best_x is None:
            # Fallback: just pick the smallest valid x
            for x in range(1, 100):
                if x in a:
                    continue
                new_diffs_mask = 0
                conflict = False
                for v in a:
                    d = x - v
                    if d <= 0: continue
                    bit = 1 << d
                    if (diffs_mask & bit) or (new_diffs_mask & bit):
                        conflict = True
                        break
                    new_diffs_mask |= bit
                if not conflict:
                    best_x = x
                    best_new_mask = new_diffs_mask
                    break
        
        if best_x is None:
            return None
        
        a.append(best_x)
        diffs_mask |= best_new_mask
        current_max = best_x
        
        if _covers_all(diffs_mask, required_mask):
            break
            
    return a

def _local_search(a, deadline):
    """
    Local search to minimize diameter.
    """
    required_mask = (1 << 31) - 2
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        n = len(a)
        
        # Try to reduce each element
        for i in range(n):
            if time.time() > deadline:
                break
            old_val = a[i]
            if i == 0 and old_val == 0:
                continue # Keep 0 fixed
            
            # Try smaller values
            for new_val in range(old_val - 1, 0, -1):
                if new_val in a:
                    continue
                
                trial = a[:i] + [new_val] + a[i+1:]
                trial.sort()
                
                ok, mask, _ = _is_sidon_and_cover(trial, required_mask)
                if not ok:
                    continue
                
                if _covers_all(mask, required_mask):
                    if trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = a[-1]
                        improved = True
                        break # Restart loop with new a
        
        # Random perturbation
        if not improved:
            if rng.random() < 0.1:
                idx = rng.randint(1, n - 1)
                new_val = rng.randint(1, best_diam)
                if new_val not in a:
                    trial = a[:idx] + [new_val] + a[idx+1:]
                    trial.sort()
                    ok, mask, _ = _is_sidon_and_cover(trial, required_mask)
                    if ok and _covers_all(mask, required_mask):
                        if trial[-1] < best_diam:
                            a = trial
                            best = list(a)
                            best_diam = a[-1]
                            improved = True
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    required_mask = (1 << 31) - 2
    
    # Phase 1: Greedy construction to get a valid set quickly
    best_set = _greedy_construct(deadline - 50)
    best_diam = best_set[-1] if best_set else float('inf')
    
    # Phase 2: Backtracking to find a better (smaller diameter) set
    best_diam_ref = [best_diam]
    better_set = _backtrack_minimal(deadline - 10, best_diam_ref)
    
    if better_set:
        best_set = better_set
        best_diam = best_diam_ref[0]
    
    # Phase 3: Local search to polish
    if best_set:
        polished = _local_search(best_set, deadline - 2)
        if polished:
            best_set = polished
            best_diam = best_set[-1]
    
    if best_set is None:
        # Fallback: return a known valid set if search failed
        best_set = [0, 1, 4, 10, 12, 28, 32, 45, 63]
        best_diam = 63
        
    return best_set

# EVOLVE-BLOCK-END
