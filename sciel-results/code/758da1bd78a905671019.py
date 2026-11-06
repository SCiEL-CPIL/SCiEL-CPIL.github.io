import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit
    
    target_diffs = set(range(1, 31))
    best_max = float('inf')
    best_set = None
    
    def dfs(current_set, used_diffs, current_max):
        nonlocal best_max, best_set
        
        # Check time
        if time.time() > deadline:
            return
            
        # Check if we have covered all target differences
        covered = used_diffs & target_diffs
        if len(covered) == 30:
            if current_max < best_max:
                best_max = current_max
                best_set = list(current_set)
            return
            
        # Pruning: if current_max is already >= best_max, no need to continue
        if current_max >= best_max:
            return
            
        # Heuristic: if we have too many elements, it might be hard to keep max small?
        # Max size for Sidon set with max element M is approx sqrt(M). 
        # If M is small, size is small.
        # Let's just let the search proceed, but limit depth if necessary.
        
        # Generate candidates for the next element x
        candidates = set()
        
        # 1. Smallest possible next element
        candidates.add(current_max + 1)
        
        # 2. Elements that would create a missing difference
        missing = target_diffs - used_diffs
        for s in current_set:
            for d in missing:
                x = s + d
                # x must be > current_max
                if x > current_max:
                    candidates.add(x)
                    
        # Sort candidates to try smaller ones first (to minimize max)
        for x in sorted(candidates):
            # Check if adding x is valid
            new_diffs = []
            valid = True
            for s in current_set:
                diff = x - s
                if diff <= 0:
                    continue # Should not happen if x > current_max
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
                
            if not valid:
                continue
                
            # Also check for duplicate differences among the new ones? 
            # Since s are distinct and x is fixed, x-s are distinct.
            # So we only need to check against used_diffs.
            
            new_used_diffs = used_diffs | set(new_diffs)
            new_set = current_set + [x]
            
            dfs(new_set, new_used_diffs, x)
            
            # Early exit if we found a perfect solution (diameter 30 is impossible for Sidon? 
            # Min diameter for covering 1..30 is likely > 30. 
            # If best_max == 30, we can stop, but it's unlikely.)
            if best_max == 30: 
                return

    # Start DFS with {0}
    # Initial set: [0]
    # Used diffs: empty
    # Current max: 0
    dfs([0], set(), 0)
    
    if best_set is None:
        # Fallback to a known valid set if search fails or times out without finding one
        # This set is a Sidon set covering 1..30?
        # Let's construct a safe fallback using the greedy method from the seed, 
        # but optimized to be fast.
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best_set
