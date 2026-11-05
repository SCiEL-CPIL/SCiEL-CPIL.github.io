import time

def construct():
    target_diffs = set(range(1, 31))
    start_time = time.time()
    
    # Try to find optimal solution via DFS for increasing diameters
    # Lower bound: k=9 requires at least 36 diffs.
    # Min diameter for Sidon set of size 9 is approx 50.
    # Start from 40 to be safe.
    
    for D in range(40, 200):
        if time.time() - start_time > 100:
            break
        
        res = find_solution_dfs(D, target_diffs)
        if res is not None:
            return res
        
    # Fallback to greedy if DFS fails or times out
    return greedy_construct()

def find_solution_dfs(D, target_diffs):
    current = [0]
    used = set()
    missing = set(target_diffs)
    start_time = time.time()
    
    def backtrack(current, used, missing):
        if time.time() - start_time > 90:
            return None
        
        if not missing:
            return current[:]
        
        if len(current) >= 32:
            return None
        
        # Pruning: if we can't cover missing diffs with remaining slots
        # Max new diffs from m elements is m*len(current) + m*(m-1)/2
        # This is a weak bound, skip for now to keep code simple
        
        # Try adding x in 1..D
        # Heuristic: Try x that covers more missing diffs first?
        # Or just iterate 1..D. Since we want min diameter, and we are in fixed D loop,
        # finding ANY solution is enough. The first one found will have max <= D.
        # Since we iterate D upwards, the first valid D gives min diameter.
        
        for x in range(1, D + 1):
            if x in current:
                continue
            
            valid = True
            new_diffs = set()
            for y in current:
                d = abs(x - y)
                if d in used:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Must make progress
            if not (new_diffs & missing):
                continue
            
            current.append(x)
            used.update(new_diffs)
            missing.difference_update(new_diffs)
            
            res = backtrack(current, used, missing)
            if res:
                return res
            
            current.pop()
            used.difference_update(new_diffs)
            for d in new_diffs:
                if d in target_diffs:
                    missing.add(d)
        
        return None
    
    return backtrack(current, used, missing)

def greedy_construct():
    a = [0]
    target = set(range(1, 31))
    missing = set(target)
    used = set()
    
    while missing and len(a) < 32:
        best_x = None
        best_score = -1
        
        # Search for x that maximizes coverage of missing diffs
        # Limit search range to keep it fast
        search_limit = max(100, 2 * max(a) + 50)
        
        for x in range(1, search_limit + 1):
            if x in a:
                continue
            
            valid = True
            new_diffs = set()
            for y in a:
                d = abs(x - y)
                if d in used:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            score = len(new_diffs & missing)
            if score > best_score:
                best_score = score
                best_x = x
                if best_score == len(missing):
                    break
        
        if best_x is None:
            break
        
        a.append(best_x)
        a.sort()
        
        # Update state
        for y in a[:-1]:
            d = abs(best_x - y)
            used.add(d)
            if d in missing:
                missing.remove(d)
        
    if not missing:
        return a
    
    # If still missing, force add pairs
    for d in sorted(list(missing)):
        found = False
        max_a = max(a)
        # Try to find x such that x and x+d are valid
        for x in range(max_a + 1, max_a + 500):
            if x in a:
                continue
            
            # Check x
            ok1 = True
            diffs1 = set()
            for y in a:
                dd = abs(x - y)
                if dd in used:
                    ok1 = False
                    break
                diffs1.add(dd)
            
            if not ok1:
                continue
            
            # Check x + d
            y2 = x + d
            if y2 in a:
                continue
            
            ok2 = True
            diffs2 = set()
            # Check against a and x
            for y in a:
                dd = abs(y2 - y)
                if dd in used:
                    ok2 = False
                    break
                diffs2.add(dd)
            
            if ok2:
                dd = d
                if dd in used:
                    ok2 = False
                else:
                    diffs2.add(dd)
            
            if ok2:
                a.extend([x, y2])
                a.sort()
                used.update(diffs1)
                used.update(diffs2)
                missing.difference_update(diffs1)
                missing.difference_update(diffs2)
                found = True
                break
        
        if not found:
            # Last resort: append far away
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = a + [y, y + d]
                # Verify Sidon
                diffs = []
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        diffs.append(trial[j] - trial[i])
                if len(set(diffs)) == len(diffs):
                    a = trial
                    break
                y += 1
    
    return a

