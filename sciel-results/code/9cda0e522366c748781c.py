import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
    
    best_diameter = float('inf')
    best_set = None
    
    # Memoization: (frozenset of S, used_diffs_mask) -> min max_seen so far
    # Actually, let's use: (tuple(S), used_diffs_mask) -> True if we've proven this state can't beat current best
    # Better: use a visited set of (tuple(S), used_mask) to avoid re-exploring same states
    
    visited = set()
    
    def dfs(S, used_diffs_mask, current_max):
        nonlocal best_diameter, best_set
        
        if time.time() - start_time > deadline:
            return False
            
        missing_mask = target_mask & ~used_diffs_mask
        
        if missing_mask == 0:
            # Found a valid set
            if current_max < best_diameter:
                best_diameter = current_max
                best_set = S[:]
            return True
        
        # Pruning: If we have reached the size limit (32) and still have missing diffs
        if len(S) >= 32:
            return False
            
        # Pruning: If current_max is already >= best_diameter, no point continuing
        if current_max >= best_diameter:
            return False
            
        # Memoization check
        state_key = (tuple(S), used_diffs_mask)
        if state_key in visited:
            return False
        visited.add(state_key)
        
        # Generate candidates for the next element
        # We want to cover missing differences.
        # Candidate x must be > max(S).
        # If x is added, new diffs are {x - s for s in S}.
        # We need at least one of these to be in 'missing' to make progress.
        
        candidates = set()
        max_s = max(S)
        
        # Candidates from missing diffs: for each s in S and each missing d, x = s + d
        missing_diffs = []
        for d in range(1, 31):
            if missing_mask & (1 << d):
                missing_diffs.append(d)
        
        for s in S:
            for d in missing_diffs:
                x = s + d
                if x > max_s:
                    candidates.add(x)
        
        # Sort candidates to try smaller values first (minimizes diameter growth)
        sorted_candidates = sorted(candidates)
        
        for x in sorted_candidates:
            if time.time() - start_time > deadline:
                return False
                
            # Check Sidon condition for x
            valid = True
            new_diffs_mask = 0
            for s in S:
                diff = x - s
                if used_diffs_mask & (1 << diff):
                    valid = False
                    break
                new_diffs_mask |= (1 << diff)
            
            if not valid:
                continue
            
            # Update state
            new_S = S + [x]
            new_used_mask = used_diffs_mask | new_diffs_mask
            
            # Recurse
            dfs(new_S, new_used_mask, x)
            
            # If we found a better solution, we can potentially prune more
            # But we continue to see if we can do even better
            
    # Initial call
    initial_S = (0,)
    initial_used_mask = 0
    
    dfs(initial_S, initial_used_mask, 0)
    
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
