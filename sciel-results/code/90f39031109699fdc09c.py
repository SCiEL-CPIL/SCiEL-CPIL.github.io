import time
import random
import sys

N = 9
MAX_TIME = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    best_path = []
    
    # Precompute neighbors for all vertices? No, 2^9 = 512 vertices.
    # We can precompute adjacency.
    # But we need to check induced path property dynamically.
    
    def is_valid_extension(path, used, new_v):
        # new_v is not in used
        # Check if new_v is adjacent to any p_i where i < len(path) - 2
        # path[-1] is the current vertex, new_v is adjacent to it.
        # We need to ensure new_v is NOT adjacent to path[i] for i < len(path) - 2.
        # Note: path[i] and path[i+1] are adjacent.
        # The condition is: for all i such that |i - (len(path))| > 1, dist(new_v, path[i]) >= 2.
        # Since new_v is adjacent to path[-1], we only need to check i < len(path) - 2.
        for i in range(len(path) - 2):
            if (new_v ^ path[i]).bit_count() == 1:
                return False
        return True

    def backtrack(path, used, depth):
        nonlocal best_path
        if time.time() - start_time > MAX_TIME:
            return False
        
        if len(path) > len(best_path):
            best_path = path.copy()
        
        cur = path[-1]
        candidates = []
        
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if is_valid_extension(path, used, v):
                # Heuristic: count how many valid neighbors v has
                # This requires checking v's neighbors, which is a bit expensive if done for all candidates.
                # Let's just collect candidates first.
                candidates.append(v)
        
        if not candidates:
            return False
        
        # Apply heuristic: sort candidates by number of their valid unused neighbors
        # This is a 1-step lookahead.
        def score_candidate(v):
            count = 0
            for b in range(N):
                nv = v ^ (1 << b)
                if nv in used:
                    continue
                # Check if nv is valid extension from v (considering path + [v])
                # The path would be path + [v]. The new vertex is nv.
                # nv is adjacent to v (last of new path).
                # We need to check if nv is adjacent to path[i] for i < len(path) - 1.
                # Note: len(path + [v]) = len(path) + 1.
                # So we check i < (len(path) + 1) - 2 = len(path) - 1.
                # This includes path[-1] (which is cur). nv is adjacent to cur? No, nv is neighbor of v.
                # Wait, if nv is neighbor of v, and v is neighbor of cur, nv might be cur or another neighbor.
                # If nv == cur, it's in used, so skipped.
                # So we check i < len(path) - 1.
                valid = True
                for i in range(len(path) - 1):
                    if (nv ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    count += 1
            return count
        
        # Sort by score ascending (Most Constrained First)
        candidates.sort(key=score_candidate)
        
        # Randomize order slightly to avoid deterministic traps?
        # Or try a few random orders?
        # For now, deterministic MRV.
        
        for v in candidates:
            path.append(v)
            used.add(v)
            
            if backtrack(path, used, depth + 1):
                # If we found a path longer than best, we can continue to see if we can go even longer
                # But if we hit time limit, we stop.
                pass
            
            path.pop()
            used.remove(v)
            
            if time.time() - start_time > MAX_TIME:
                return False
        
        return False

    # Initial path
    path = [0]
    used = {0}
    
    # We might want to try multiple starting points or randomizations if the first one fails to be long.
    # But 0 is symmetric.
    
    # To improve exploration, we can randomize the bit order in the candidate generation or sorting.
    # Let's do a few iterations with different random seeds for the tie-breaking in sorting or candidate selection.
    
    best_score = 0
    best_result = [0]
    
    # Run backtracking for a few seconds or until time is up.
    # Since backtracking is exhaustive (within time limit), we just run it once with MRV.
    # If it's too slow, we might need to limit depth or use beam search.
    # For N=9, MRV backtracking should be able to find long snakes.
    
    backtrack(path, used, 0)
    
    if len(best_path) > len(best_result):
        best_result = best_path
        
    return best_result

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
