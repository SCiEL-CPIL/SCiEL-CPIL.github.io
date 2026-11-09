import time
import sys

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    MAX_SIZE = 32
    
    best_max = float('inf')
    best_set = None
    
    def is_valid_extension(S, x, used_diffs):
        """Check if adding x to S maintains Sidon property."""
        for s in S:
            d = x - s
            if d in used_diffs:
                return False, None
            # We don't need to add to used_diffs here, just check
        # Also check if x is already in S (implied by x > max(S) in caller)
        return True, None

    def dfs(S, used_diffs, covered_mask, depth):
        nonlocal best_max, best_set
        
        # Time check
        if time.time() - start_time > deadline:
            return False # Signal timeout
        
        # Check if solution found
        if covered_mask == full_mask:
            current_max = max(S)
            if current_max < best_max:
                best_max = current_max
                best_set = list(S)
                # We don't return True immediately because we might find a smaller max
                # But if we are doing DFS in order of increasing max, the first one is best?
                # No, we generate candidates not strictly sorted by final max.
                # So we continue to search for better.
                # However, if we prune based on best_max, we can stop this branch.
            return True # Found a valid solution, no need to extend further in this branch
        
        if len(S) >= MAX_SIZE:
            return False
        
        # Pruning: If current max is already >= best_max, no point continuing this branch
        # Note: S is always sorted, so max(S) is S[-1]
        if S[-1] >= best_max:
            return False
            
        # Pruning: Capacity check
        missing = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        if not missing:
            return True # Should have been caught above
            
        remaining_slots = MAX_SIZE - len(S)
        # Max new differences we can create
        # Each new element creates (len(S) + i) differences with previous elements
        # Sum_{i=0}^{remaining_slots-1} (len(S) + i)
        max_possible_new_diffs = remaining_slots * len(S) + (remaining_slots * (remaining_slots - 1)) // 2
        
        if len(missing) > max_possible_new_diffs:
            return False
            
        # Generate candidates
        # We want to pick x > S[-1]
        # Candidates are derived from missing differences
        candidates = set()
        last = S[-1]
        
        for d in missing:
            for s in S:
                x = s + d
                if x > last:
                    candidates.add(x)
                    
        # Also, to ensure we don't miss "bridging" elements that don't immediately cover a missing diff
        # but are small, we might add last + 1, last + 2, etc?
        # Given the density of 1..30, the "from missing" strategy is usually sufficient.
        # However, if we are stuck, we might need a small step.
        # Let's add a few small steps just in case.
        for step in range(1, 5):
            candidates.add(last + step)
            
        # Filter candidates
        valid_candidates = []
        for x in candidates:
            if x > 10**6: # Hard limit from problem statement, though best_max will be much smaller
                continue
                
            # Check Sidon property
            valid = True
            new_diffs = []
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            # Calculate new covered mask
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            # Heuristic: Prefer candidates that cover more missing diffs
            new_missing_count = 30 - bin(new_mask).count('1')
            # We want to minimize new_missing_count, and secondarily minimize x
            valid_candidates.append((new_missing_count, x, new_mask, new_diffs))
            
        # Sort candidates: Fewest missing diffs first, then smallest x
        valid_candidates.sort(key=lambda item: (item[0], item[1]))
        
        found_solution = False
        for _, x, new_mask, new_diffs in valid_candidates:
            new_S = S + (x,)
            new_used_diffs = used_diffs.union(new_diffs)
            
            # Recurse
            res = dfs(new_S, new_used_diffs, new_mask, depth + 1)
            if res:
                found_solution = True
                # We continue looping to find potentially better solutions (smaller max)
                # unless we hit the time limit or best_max is very small
                if time.time() - start_time > deadline:
                    return True
                    
        return found_solution

    # Initial state
    initial_S = (0,)
    initial_used_diffs = set()
    initial_mask = 0
    
    dfs(initial_S, initial_used_diffs, initial_mask, 0)
    
    if best_set:
        return best_set
    else:
        # Fallback: Return a known valid set if search fails (e.g. due to time)
        # This should rarely happen with the pruning above.
        # A simple greedy construction as fallback
        S = [0]
        covered = set()
        used = set()
        while not covered.issuperset(target_diffs) and len(S) < MAX_SIZE:
            missing = target_diffs - covered
            best_x = None
            best_score = float('inf')
            last = max(S)
            # Search for a good x
            for d in missing:
                for s in S:
                    x = s + d
                    if x <= last:
                        continue
                    # Check validity
                    valid = True
                    new_diffs = []
                    for ss in S:
                        dd = x - ss
                        if dd in used:
                            valid = False
                            break
                        new_diffs.append(dd)
                    if not valid:
                        continue
                    new_cov = covered.union([dd for dd in new_diffs if dd in target_diffs])
                    score = len(target_diffs - new_cov)
                    if score < best_score:
                        best_score = score
                        best_x = x
                        best_new_diffs = new_diffs
                        best_new_cov = new_cov
            if best_x is None:
                # No good candidate, try smallest valid x
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for ss in S:
                        dd = x - ss
                        if dd in used:
                            valid = False
                            break
                        new_diffs.append(dd)
                    if valid:
                        best_x = x
                        best_new_diffs = new_diffs
                        best_new_cov = covered.union([dd for dd in new_diffs if dd in target_diffs])
                        break
                    x += 1
            
            S.append(best_x)
            S.sort()
            covered = best_new_cov
            used = used.union(best_new_diffs)
            
        return S

