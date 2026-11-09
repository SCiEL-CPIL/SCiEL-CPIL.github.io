import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute neighbor sets for fast lookup
    neighbor_sets = [set(n) for n in neighbors]
    
    best_path = [0, 1]
    best_len = 2
    
    def can_extend(used, end, candidate):
        """Check if candidate can be added to path ending at end."""
        if used[candidate]:
            return False
        # candidate must not be adjacent to any used vertex except end
        for u in neighbors[candidate]:
            if u == end:
                continue
            if used[u]:
                return False
        return True
    
    def get_valid_extensions(used, end):
        """Get all valid candidates to extend from end."""
        result = []
        for v in neighbors[end]:
            if can_extend(used, end, v):
                result.append(v)
        return result
    
    def extend_with_backtrack(path, used, max_depth=50, time_limit=None):
        """
        Try to extend path by up to max_depth steps using DFS with backtracking.
        Returns the longest path found.
        """
        if time_limit is not None and time.time() > time_limit:
            return path
        
        best = list(path)
        best_used = list(used)
        
        def dfs(current_path, current_used, depth, max_d):
            nonlocal best, best_used
            if time_limit is not None and time.time() > time_limit:
                return
            
            if len(current_path) > len(best):
                best = list(current_path)
                best_used = list(current_used)
            
            if depth >= max_d:
                return
            
            end = current_path[-1]
            candidates = get_valid_extensions(current_used, end)
            
            # Sort by degree (Warnsdorff)
            candidates.sort(key=lambda v: sum(1 for u in neighbors[v] if not current_used[u]))
            
            for cand in candidates:
                current_used[cand] = True
                current_path.append(cand)
                dfs(current_path, current_used, depth + 1, max_d)
                current_path.pop()
                current_used[cand] = False
                
                if time_limit is not None and time.time() > time_limit:
                    return
        
        dfs(path, used, 0, max_depth)
        return best
    
    def build_structured_initial():
        """
        Build a longer initial snake using a structured approach.
        Use Gray-code-like ordering to build a longer path.
        """
        # Start with a simple path and try to extend
        path = [0]
        used = [False] * MAX_VERTS
        used[0] = True
        
        # Greedy extension with backtracking
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 5:
                break
            end = path[-1]
            candidates = get_valid_extensions(used, end)
            if not candidates:
                break
            
            # Try each candidate, pick one that allows further extension
            best_cand = None
            best_score = -1
            for cand in candidates:
                # Count how many further extensions are possible
                used[cand] = True
                next_cands = get_valid_extensions(used, cand)
                score = len(next_cands)
                used[cand] = False
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_cand is None:
                break
            path.append(best_cand)
            used[best_cand] = True
        
        return path
    
    def local_search_improve(path, time_budget=10.0):
        """
        Try to improve path by removing a vertex and re-extending.
        """
        if not path or len(path) < 2:
            return path
        
        best = list(path)
        best_len = len(best)
        deadline = start_time + time_budget
        
        # Try removing vertices from the middle and re-extending both ends
        for remove_idx in range(1, len(path) - 1):
            if time.time() > deadline:
                break
            
            # Remove vertex at remove_idx
            removed = path[remove_idx]
            new_path = path[:remove_idx] + path[remove_idx+1:]
            
            # Check if the new path is still valid (adjacent vertices still differ by 1 bit)
            valid = True
            for i in range(len(new_path) - 1):
                diff = new_path[i] ^ new_path[i+1]
                if diff & (diff - 1) != 0:  # Not a power of 2
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Rebuild used set
            used = [False] * MAX_VERTS
            for v in new_path:
                used[v] = True
            
            # Try to extend both ends
            improved = extend_with_backtrack(new_path, used, max_depth=30, time_limit=deadline)
            
            if len(improved) > best_len:
                best = improved
                best_len = len(improved)
        
        return best
    
    # Phase 1: Build a good initial snake
    initial = build_structured_initial()
    if len(initial) > best_len:
        best_path = initial
        best_len = len(initial)
    
    # Phase 2: Extend with backtracking
    used = [False] * MAX_VERTS
    for v in best_path:
        used[v] = True
    
    extended = extend_with_backtrack(best_path, used, max_depth=100, time_limit=start_time + 20)
    if len(extended) > best_len:
        best_path = extended
        best_len = len(extended)
    
    # Phase 3: Local search improvements
    while time.time() - start_time < DEADLINE - 10:
        improved = local_search_improve(best_path, time_budget=min(10, DEADLINE - 5 - (time.time() - start_time)))
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
        
        # Try extending from the improved path
        used = [False] * MAX_VERTS
        for v in best_path:
            used[v] = True
        extended = extend_with_backtrack(best_path, used, max_depth=50, time_limit=start_time + DEADLINE - 5)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(extended)
    
    # Final extension attempt
    if time.time() - start_time < DEADLINE - 2:
        used = [False] * MAX_VERTS
        for v in best_path:
            used[v] = True
        extended = extend_with_backtrack(best_path, used, max_depth=100, time_limit=DEADLINE - 1)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(extended)
    
    return best_path
