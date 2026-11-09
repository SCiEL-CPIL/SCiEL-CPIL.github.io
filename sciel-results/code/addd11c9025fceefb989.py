import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Fast greedy to get an initial valid solution and upper bound
    def greedy_construct():
        S = [0]
        covered = set()
        max_val = 0
        
        for _ in range(32):
            if covered.issuperset(set(range(1, 31))):
                break
            missing = set(range(1, 31)) - covered
            if not missing:
                break
                
            best_x = None
            best_score = float('inf')
            
            # Try candidates from max_val + 1 upwards
            # We don't want to go too high in greedy, keep it tight
            for x in range(max_val + 1, max_val + 100):
                # Check Sidon property
                valid = True
                new_diffs = []
                for y in S:
                    d = x - y
                    if d in covered:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                # Score: prioritize covering missing diffs
                new_covered = covered.union(new_diffs)
                new_missing = set(range(1, 31)) - new_covered
                # Prefer x that covers more missing
                score = len(new_missing)
                if score < best_score:
                    best_score = score
                    best_x = x
                    if best_score == 0:
                        break
            
            if best_x is None:
                # Fallback: find any valid x
                x = max_val + 1
                while True:
                    valid = True
                    for y in S:
                        if x - y in covered:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                best_x = x
            
            S.append(best_x)
            S.sort()
            max_val = max(S)
            # Update covered
            for i in range(len(S)):
                for j in range(i):
                    covered.add(S[i] - S[j])
        
        return S, max(S)

    # Get initial solution
    init_S, init_max = greedy_construct()
    if not init_S or not set(range(1, 31)).issubset(set(y-x for i, x in enumerate(init_S) for y in init_S[i+1:])):
         # If greedy failed (unlikely), return a simple valid set if possible or exit
         # But greedy should work.
         pass

    global_best_max = init_max
    global_best_set = init_S
    
    # DFS to improve
    # State: S (sorted list), covered (set), max_elem (int)
    # We will use recursion with a limit on depth and time
    
    def dfs(S, covered, max_elem):
        nonlocal global_best_max, global_best_set
        nonlocal start_time, deadline
        
        if time.time() > deadline:
            return
            
        # Check if goal met
        if covered.issuperset(set(range(1, 31))):
            if max_elem < global_best_max:
                global_best_max = max_elem
                global_best_set = S[:]
            return
        
        # Prune if we can't improve
        if max_elem >= global_best_max:
            return
            
        # Prune if set is too large
        if len(S) >= 32:
            return
            
        # Generate candidates
        # We want to try x > max_elem
        # To minimize max, try small x first
        # But we must ensure we can still cover missing diffs.
        # If the largest missing diff is d_max, we need b-a=d_max.
        # This requires b >= d_max.
        # If max_elem + (32 - len(S)) * something < d_max? 
        # Simple bound: if max_elem < max(covered) is not useful.
        # If max_elem < max(missing), we must add elements.
        
        missing = set(range(1, 31)) - covered
        if not missing:
            return
            
        # Optimization: If we have very few elements left, we can bound the search
        # For now, just iterate x
        
        # Limit the range of x to search to avoid infinite loops or slow progress
        # If we don't find a valid x in a reasonable range, we might be stuck
        # But with Sidon sets, valid x are sparse.
        
        # Heuristic: Try x that cover missing diffs
        # Instead of all x, we can generate candidates based on missing diffs
        # If d is missing, we need x-y=d => x = y+d.
        # So for each missing d and each y in S, x = y+d is a candidate.
        # Also, we can just try x = max_elem + 1, max_elem + 2...
        
        # Combining both:
        candidates = set()
        # 1. Candidates from missing diffs
        for d in missing:
            for y in S:
                x = y + d
                if x > max_elem:
                    candidates.add(x)
                # x = y - d is < y <= max_elem, so not a new max
                # But wait, we are adding to the set, so x must be > max(S) to be a new element?
                # No, the set is sorted. We are adding a new element.
                # If we add x < max_elem, it would be inserted in the middle.
                # But we are building the set by adding the *next* element?
                # No, we are building a set. The order of addition doesn't matter for the final set.
                # However, to maintain the "max_elem" bound effectively, we usually add the largest element last?
                # Or we just build the set and track the max.
                # If we add x < max_elem, the max doesn't change.
                # If we add x > max_elem, the max increases.
                # To minimize the final max, we should prefer adding small elements.
                # But if we add small elements, we might not cover large differences.
                
                # Let's stick to adding elements in increasing order of value?
                # No, that's not required.
                # But if we allow x < max_elem, the search space is huge.
                # Standard approach: Build the set by adding elements in sorted order?
                # If we assume the set is sorted $a_0 < a_1 < \dots < a_k$, and we are choosing $a_k$,
                # then $a_k$ must be $> a_{k-1}$.
                # This reduces the search space significantly.
                # Let's enforce that we only add elements $x > \max(S)$.
                # This means we are constructing the set in increasing order.
                # Is it possible that the optimal set requires adding a smaller element later?
                # No, the set is a set. The final set is sorted.
                # The order of construction is just an artifact.
                # If we construct $a_0, a_1, \dots, a_k$ such that $a_i < a_{i+1}$,
                # we are enumerating all possible sorted sets.
                # So yes, we can restrict to $x > \max(S)$.
        
        # So, candidates must be $> \max_elem$.
        # We generate candidates from missing diffs: $x = y + d$.
        # If $y+d > \max_elem$, it's a candidate.
        # We also try $x = \max_elem + 1, \max_elem + 2, \dots$ up to some limit?
        # If we only use $y+d$, we might miss a valid $x$ that doesn't directly cover a missing diff
        # but is necessary to allow future elements to cover diffs?
        # Actually, if $x$ doesn't cover any missing diff, it adds only "wasted" differences.
        # Since we have a limit of 32 elements, and we need to cover 30 diffs,
        # each element adds at most $|S|$ diffs.
        # If an element adds 0 new missing diffs, it's a bad move unless we are forced.
        # But we are not forced to use all 32. We can stop early.
        # So, skipping candidates that don't cover any missing diff is a strong heuristic.
        
        valid_candidates = []
        for x in candidates:
            if x <= max_elem:
                continue
            if x >= global_best_max:
                continue
            
            # Check Sidon property
            valid = True
            new_diffs = []
            for y in S:
                d = x - y
                if d in covered:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check if it covers any missing diff
            new_covered = covered.union(new_diffs)
            if not (new_covered - covered).intersection(missing):
                # Doesn't cover any new missing diff
                # We might still take it if we are stuck, but for now, let's skip
                # to focus on progress.
                # However, if we skip all, we might fail.
                # Let's collect "progress" candidates and "neutral" candidates.
                pass # Skip for now to keep search tree small
                
            valid_candidates.append((x, new_diffs, new_covered))
            
        # Sort candidates by x (smallest first) to minimize max
        valid_candidates.sort(key=lambda t: t[0])
        
        # If no candidates cover missing diffs, we might need to try neutral ones?
        # Or try a broader range of x?
        # For now, if valid_candidates is empty, we backtrack.
        
        for x, new_diffs, new_covered in valid_candidates:
            S.append(x)
            dfs(S, new_covered, x)
            S.pop()
            
            if time.time() > deadline:
                return

    # Run DFS
    # To avoid deep recursion limits, we can use iterative DFS or increase recursion limit
    import sys
    sys.setrecursionlimit(10000)
    
    dfs(init_S, set(y-x for i, x in enumerate(init_S) for y in init_S[i+1:]), max(init_S))
    
    return global_best_set
