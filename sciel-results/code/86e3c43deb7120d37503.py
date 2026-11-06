import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    initial = [1, 2, 5, 11]
    initial_set = set(initial)
    
    used_diffs = set()
    for i in range(len(initial)):
        for j in range(i):
            used_diffs.add(abs(initial[i] - initial[j]))
            
    best_set = list(initial)
    best_size = len(initial)
    
    all_candidates = [x for x in range(1, 285) if x not in initial_set]
    
    def is_valid(x, current_set, current_diffs):
        for y in current_set:
            d = abs(x - y)
            if d in current_diffs:
                return False
        return True

    def get_candidates(current_set, current_diffs):
        cands = []
        for x in all_candidates:
            if x in current_set:
                continue
            if is_valid(x, current_set, current_diffs):
                cands.append(x)
        return cands

    def min_dist_to_set(x, current_set):
        min_d = float('inf')
        for y in current_set:
            d = abs(x - y)
            if d < min_d:
                min_d = d
        return min_d

    def heuristic_score_lcv(x, current_set, current_diffs, cands):
        blocked = 0
        new_diffs = set(current_diffs)
        for y in current_set:
            new_diffs.add(abs(x - y))
        
        for z in cands:
            if z == x or z in current_set:
                continue
            valid = True
            for y in current_set:
                if abs(z - y) in new_diffs:
                    valid = False
                    break
            if not valid:
                blocked += 1
        return blocked

    def backtrack(current_set, current_diffs, depth):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            
        if best_size >= 20:
            return
            
        cands = get_candidates(current_set, current_diffs)
        if not cands:
            return
            
        if len(cands) <= 25:
            # Use LCV
            scores = []
            for x in cands:
                s = heuristic_score_lcv(x, current_set, current_diffs, cands)
                scores.append((s, x))
            scores.sort()
            cands = [x for _, x in scores]
        else:
            # Use Max Min-Distance heuristic
            # Sort by min distance to existing set, descending
            cands.sort(key=lambda x: min_dist_to_set(x, current_set), reverse=True)
            
        for x in cands:
            if time.time() > deadline:
                return
                
            new_set = set(current_set)
            new_set.add(x)
            
            new_diffs = set(current_diffs)
            for y in current_set:
                new_diffs.add(abs(x - y))
                
            backtrack(new_set, new_diffs, depth + 1)
            
            if best_size >= 20:
                return

    backtrack(set(initial), used_diffs, 0)
    
    return best_set
