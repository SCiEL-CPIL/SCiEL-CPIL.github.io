import time
import random
import sys

N = 9

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    best_path = [0]
    best_len = 1
    sys.setrecursionlimit(10000)
    
    def count_valid_next(v, path, used_set):
        # Count how many valid moves are available from v, given current path and used_set
        # Note: v is the NEW vertex. path is the path BEFORE v.
        # So the full path would be path + [v].
        # The constraint for a next vertex w (neighbor of v) is:
        # 1. w not in used_set (and w != v, which is true for neighbors)
        # 2. dist(w, path[i]) >= 2 for all i in range(len(path))
        #    AND dist(w, v) == 1 (guaranteed by neighbor)
        #    Note: The induced path condition requires dist(w, path[i]) >= 2 for all i < len(path+1)-1 = len(path).
        #    So we check against all existing path vertices.
        
        count = 0
        for b in range(N):
            w = v ^ (1 << b)
            if w in used_set:
                continue
            # Check induced constraint against all vertices in current path
            valid = True
            for p in path:
                if (w ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                count += 1
        return count

    def dfs(path, used_set, depth):
        nonlocal best_path, best_len, deadline
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            
        cur = path[-1]
        
        # Generate candidates
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            # Check if v is valid to append to current path
            valid = True
            for p in path[:-1]: # Check against all but the last one (dist to last is 1)
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                # Calculate heuristic score
                # We need to check if v is valid, then count its next moves.
                # To count next moves, we temporarily consider v as part of path.
                # count_valid_next(v, path, used_set) does this.
                score = count_valid_next(v, path, used_set)
                candidates.append((score, v))
                
        if not candidates:
            return
            
        # Sort by score (Warnsdorff: fewest exits first)
        candidates.sort(key=lambda x: x[0])
        
        # Add some randomness to escape local minima?
        # Sometimes pure Warnsdorff gets stuck.
        # Let's shuffle candidates with the same score, or just pick the first one.
        # For restarts, we can randomize the tie-breaking or the initial order.
        
        # To allow different paths in different restarts, we can randomize the bit order 
        # or add noise to the score.
        # Let's just use the sorted order. The restarts with different random seeds 
        # might not help if the heuristic is deterministic.
        # We can randomize the order of bits in the candidate generation?
        # Or randomize the tie-breaking.
        
        # Let's try: if multiple candidates have the same score, pick one randomly.
        # Or, simply shuffle the candidates list before sorting? No, that defeats the purpose.
        # Let's add a small random noise to the score to break ties randomly.
        for i in range(len(candidates)):
            candidates[i] = (candidates[i][0] + random.random() * 0.1, candidates[i][1])
        candidates.sort(key=lambda x: x[0])
        
        for score, v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            used_set.add(v)
            
            dfs(path, used_set, depth + 1)
            
            path.pop()
            used_set.remove(v)

    # Main loop
    while time.time() < deadline:
        random.seed() # Re-seed for each restart? Or just let it run.
        # Actually, random state persists. We just run DFS.
        # Since the heuristic has random noise, each run will be different.
        dfs([0], {0}, 0)

    return best_path
