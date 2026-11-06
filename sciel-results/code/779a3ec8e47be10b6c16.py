import sys
import time

def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30, minimizing max element.
    Uses a backtracking search with candidate generation from missing differences.
    """
    target_diffs = set(range(1, 31))
    
    # We want to minimize the largest element.
    # We iterate over possible set sizes N.
    # Minimum N: N(N-1)/2 >= 30 -> N >= 8.
    # However, Sidon sets have structural constraints.
    # Let's try N from 9 to 15.
    
    best_solution = None
    best_max_val = float('inf')
    
    start_time = time.time()
    max_time = 100  # seconds, leaving buffer for 120s limit

    for N in range(9, 16):
        if time.time() - start_time > max_time:
            break
            
        # DFS to find a Sidon set of size N covering 1..30
        # We want to minimize the max element, so we prefer smaller candidates.
        
        def solve_N(N):
            # S: list of elements, sorted
            # D: set of differences
            # Missing: set of target diffs not in D
            
            # Initial state: S=[0], D=set(), Missing=target_diffs
            # We need to add N-1 more elements.
            
            # To ensure we find the minimal max element, we should probably
            # search for solutions with max element < current best.
            # But since we don't have a bound yet, we just find *a* solution.
            # Sorting candidates by value helps find compact ones.
            
            def dfs(S, D, Missing, depth):
                nonlocal best_solution, best_max_val, start_time, max_time
                
                if time.time() - start_time > max_time:
                    return None
                
                if len(S) == N:
                    if not Missing:
                        # Found a solution
                        max_val = max(S)
                        if max_val < best_max_val:
                            best_max_val = max_val
                            best_solution = list(S)
                        return True
                    else:
                        return False
                
                # Pruning:
                # Max possible new differences from remaining slots
                slots_left = N - len(S)
                # Each new element adds at most len(S) diffs with existing, and 1 with previous new?
                # Actually, if we add k elements, they form k*(k-1)/2 diffs among themselves
                # and k*len(S) diffs with existing.
                # Total new diffs <= slots_left * len(S) + slots_left * (slots_left - 1) // 2
                # This is an upper bound on how many missing diffs we can cover.
                max_new_diffs = slots_left * len(S) + slots_left * (slots_left - 1) // 2
                if len(Missing) > max_new_diffs:
                    return False
                
                # Generate candidates
                # Candidates come from:
                # 1. x = y + d for y in S, d in Missing
                # 2. x = y - d for y in S, d in Missing (if x >= 0)
                
                candidates = set()
                for y in S:
                    for d in Missing:
                        x1 = y + d
                        candidates.add(x1)
                        x2 = y - d
                        if x2 >= 0:
                            candidates.add(x2)
                
                # Filter candidates
                # x must not be in S
                # x must be such that adding it doesn't violate Sidon
                # Sidon check: for all y in S, (x - y) not in D and (y - x) not in D
                # Since we add x, new diffs are {x-y for y in S} (if x > y) or {y-x for y in S} (if x < y).
                # Actually, diff is abs(x-y).
                # We just need to check if abs(x-y) is already in D.
                
                valid_candidates = []
                for x in candidates:
                    if x in S:
                        continue
                    
                    # Check Sidon property
                    valid = True
                    for y in S:
                        diff = abs(x - y)
                        if diff in D:
                            valid = False
                            break
                    
                    if valid:
                        # Also check if x is "too large" to be useful?
                        # We want to minimize max(S).
                        # If best_solution exists, we can prune x > best_max_val
                        if best_solution is not None and x > best_max_val:
                            continue
                            
                        valid_candidates.append(x)
                
                # Sort candidates to prefer smaller values (to keep max small)
                valid_candidates.sort()
                
                # Try each candidate
                for x in valid_candidates:
                    new_S = S + [x]
                    new_S.sort()
                    
                    new_D = set(D)
                    new_Missing = set(Missing)
                    
                    for y in S:
                        diff = abs(x - y)
                        new_D.add(diff)
                        if diff in new_Missing:
                            new_Missing.remove(diff)
                    
                    # Recurse
                    # Note: We pass new_Missing to avoid recalculating
                    if dfs(new_S, new_D, new_Missing, depth + 1):
                        # We found a solution. 
                        # Do we stop? 
                        # We want the GLOBAL minimum max element.
                        # If we are iterating N from small to large, and within N we sort candidates,
                        # the first solution found for a given N might not be the absolute minimum max element for that N.
                        # However, for small N, the search space is small.
                        # To be safe, we could continue searching for better solutions for this N?
                        # That might be too slow.
                        # Let's assume the first solution found with sorted candidates is good enough,
                        # or we can do a limited search.
                        # Given the time limit, let's just return the first one found for this N.
                        # If we find a solution for N=9, it's likely better than N=10.
                        return True
                        
                return False
            
            return dfs([0], set(), set(target_diffs), 1)
        
        # Run the solver for this N
        # We need to reset best_solution/best_max_val? No, we want the global best.
        # But we should only update if we find a solution.
        # The dfs function updates best_solution.
        
        # To make the search more effective, we can try to prune with a heuristic upper bound.
        # But let's just run it.
        
        # Optimization: If we already found a solution with max_val < some threshold, 
        # we might not need to check larger N if we are confident.
        # But smaller N usually allows smaller max_val? Not necessarily, but likely.
        
        solve_N(N)
        
        # If we found a solution with N=9, it's likely optimal or near optimal.
        # We can break if best_max_val is very small (e.g., < 50).
        if best_solution is not None and best_max_val < 40:
            break

    if best_solution is None:
        # Fallback: Greedy construction
        # This should rarely be needed
        a = [0]
        def get_diffs(s):
            return set(abs(y - x) for i, x in enumerate(s) for y in s[i + 1:])
        
        for d in range(1, 31):
            if d in get_diffs(a):
                continue
            found = False
            # Try to add x such that x-y=d or y-x=d
            for y in a:
                x1 = y + d
                x2 = y - d
                for x in [x1, x2]:
                    if x < 0 or x in a:
                        continue
                    trial = a + [x]
                    diffs = get_diffs(trial)
                    if len(diffs) == len(trial) * (len(trial) - 1) // 2:
                        a = trial
                        found = True
                        break
                if found:
                    break
            if not found:
                # Add a large element to cover d
                # We need x such that x-y=d for some y, and no other conflicts.
                # Just pick a large x.
                x = 2 * max(a) + d + 1
                while True:
                    trial = a + [x]
                    diffs = get_diffs(trial)
                    if len(diffs) == len(trial) * (len(trial) - 1) // 2:
                        a = trial
                        break
                    x += 1
        best_solution = a
        best_max_val = max(a)

    return best_solution
