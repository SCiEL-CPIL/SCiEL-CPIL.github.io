import time
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    target_diffs = set(range(1, 31))
    best_diameter = float('inf')
    best_set = None
    
    # State for backtracking
    # S: sorted list of elements
    # used_diffs: set of all differences generated so far
    # missing: set of target differences not yet covered
    
    def dfs(S, used_diffs, missing, current_max):
        nonlocal best_diameter, best_set
        
        if time.time() - start_time > deadline:
            return
            
        if not missing:
            # Found a valid set
            if current_max < best_diameter:
                best_diameter = current_max
                best_set = S[:]
            return
        
        # Pruning: If we have reached the size limit (32) and still have missing diffs
        if len(S) >= 32:
            return
            
        # Pruning: If current_max is already >= best_diameter, no point continuing
        # Note: We might still find a better set with smaller max if we haven't hit the limit?
        # No, if current_max >= best_diameter, any superset will have max >= current_max.
        # So we can prune if current_max >= best_diameter.
        if current_max >= best_diameter:
            return

        # Generate candidates for the next element
        # We want to cover missing differences.
        # Candidate x must be > max(S).
        # If x is added, new diffs are {x - s for s in S}.
        # We need at least one of these to be in 'missing' to make progress?
        # Not strictly, but it's a good heuristic. However, sometimes we need to add an element
        # that doesn't immediately cover a missing diff to enable future coverage or avoid conflicts.
        # But for minimization, covering diffs early is key.
        
        candidates = set()
        max_s = max(S)
        
        # Candidates from missing diffs
        for s in S:
            for d in missing:
                x = s + d
                if x > max_s:
                    candidates.add(x)
        
        # Also consider small integers just above max_s to avoid getting stuck
        # if direct sums cause conflicts.
        # But this might bloat the search. Let's stick to s+d first.
        # If no candidates, we might be stuck.
        
        # Sort candidates to try smaller values first (minimizes diameter growth)
        sorted_candidates = sorted(candidates)
        
        for x in sorted_candidates:
            if time.time() - start_time > deadline:
                return
                
            # Check Sidon condition for x
            valid = True
            new_diffs = []
            for s in S:
                diff = x - s
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
                
            # Check if x is within reasonable bounds
            # If x is huge, it might not help minimize diameter, but we check best_diameter
            # The pruning `current_max >= best_diameter` handles this if we pass x.
            
            # Update state
            new_used_diffs = set(used_diffs)
            new_missing = set(missing)
            for d in new_diffs:
                new_used_diffs.add(d)
                if d in new_missing:
                    new_missing.remove(d)
            
            new_S = S + [x]
            
            # Recurse
            dfs(new_S, new_used_diffs, new_missing, x)
            
            # Backtrack
            # (State is passed by value/reference carefully, sets are immutable in logic here via copying)
            
    # Initial call
    initial_S = [0]
    initial_used = set()
    initial_missing = set(range(1, 31))
    
    dfs(initial_S, initial_used, initial_missing, 0)
    
    # If no solution found (unlikely, but possible if time runs out or logic error),
    # provide a fallback valid set.
    # A known small Sidon set covering 1..30 can be constructed greedily if needed.
    if best_set is None:
        # Fallback: Greedy construction
        S = [0]
        used = set()
        missing = set(range(1, 31))
        while missing and len(S) < 32:
            found = False
            # Try to find x that covers most missing diffs without conflict
            best_x = None
            best_cov = -1
            # Search space for x: 1 to 200 (heuristic)
            for x in range(max(S) + 1, max(S) + 100):
                if x in S:
                    continue
                valid = True
                new_cov = 0
                for s in S:
                    d = x - s
                    if d in used:
                        valid = False
                        break
                    if d in missing:
                        new_cov += 1
                if valid and new_cov > best_cov:
                    best_cov = new_cov
                    best_x = x
            
            if best_x is None:
                # If no good x found in range, try wider
                for x in range(max(S) + 1, max(S) + 1000):
                    valid = True
                    for s in S:
                        d = x - s
                        if d in used:
                            valid = False
                            break
                    if valid:
                        best_x = x
                        break
            
            if best_x is None:
                break
                
            S.append(best_x)
            for s in S[:-1]:
                used.add(best_x - s)
            for d in used:
                if d in missing:
                    missing.remove(d)
        
        best_set = S
        best_diameter = max(S)
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    print(max(res))
