import time
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    best_solution = None
    best_max_val = float('inf')
    
    # Try different set sizes k
    # k=8 is impossible (28 diffs < 30)
    # Start with k=9
    for k in range(9, 13): 
        if time.time() > deadline:
            break
            
        # Lower bound for M
        # We need 30 diffs, so M >= 30.
        # We have k(k-1)/2 diffs, all <= M, so M >= k(k-1)/2.
        min_m = max(30, k * (k - 1) // 2)
        
        # Search M from min_m upwards
        # Limit M to some reasonable max to avoid infinite loop if no solution
        # But we expect a solution soon.
        for m in range(min_m, 200): 
            if time.time() > deadline:
                break
                
            # DFS to find a set of size k in [0, m] covering target_diffs
            # S starts with [0]
            # We need to pick k-1 more elements.
            
            found = _dfs(
                S=[0],
                diffs=set(),
                needed=target_diffs.copy(),
                k=k,
                m=m,
                deadline=deadline
            )
            
            if found:
                max_val = max(found)
                if max_val < best_max_val:
                    best_max_val = max_val
                    best_solution = found
                # If we found a solution for this k, can a larger k give a smaller M?
                # Unlikely, as larger k requires larger M to fit diffs.
                # But let's check a few.
                # Actually, if k=9 gives M=40, k=10 might give M=39?
                # Diffs for k=10 is 45. M>=45. So M will be larger.
                # So k=9 is likely optimal for minimal M.
                # We can break out of k loop if we are confident.
                # Let's just return the first valid one found for k=9 if it's small.
                # Or continue to ensure we get the global min.
                # Given time constraints, finding one good one is priority.
                pass
                
        # If we found a solution for k=9, it's likely the best.
        if best_solution and k == 9:
             # Check if we can do better with k=10?
             # Min M for k=10 is 45.
             # If best_max_val for k=9 is < 45, k=10 cannot beat it.
             if best_max_val < 45:
                 break

    if best_solution:
        return best_solution
    
    # Fallback: Greedy (like seed) if search failed
    return _greedy()

def _dfs(S, diffs, needed, k, m, deadline):
    if time.time() > deadline:
        return None
        
    if not needed:
        return S
        
    if len(S) == k:
        return None
        
    last = S[-1]
    
    # Pruning:
    # The largest needed difference is max(needed).
    # The maximum difference we can generate with current S and max element m is m.
    # If max(needed) > m, we can't cover it? 
    # Wait, if we pick x <= m, x-s <= m.
    # So if max(needed) > m, it's impossible.
    if needed and max(needed) > m:
        return None
        
    # Pruning:
    # Remaining elements: r = k - len(S)
    # Max new differences we can generate:
    # Each new element x adds len(S) diffs.
    # Total new diffs = sum_{i=0}^{r-1} (len(S) + i)
    # If len(needed) > total_new_diffs, impossible.
    r = k - len(S)
    total_new = 0
    curr_len = len(S)
    for i in range(r):
        total_new += curr_len + i
    if len(needed) > total_new:
        return None
        
    # Try next elements
    # Optimization: 
    # We want to cover 'needed'.
    # For each x in range(last+1, m+1):
    #    Check if x-s are in diffs.
    #    If valid, recurse.
    
    # To speed up, we can prioritize x that cover many needed diffs.
    # But simple iteration is okay for small m.
    
    for x in range(last + 1, m + 1):
        # Check distinctness
        valid = True
        new_diffs = []
        for s in S:
            d = x - s
            if d in diffs:
                valid = False
                break
            new_diffs.append(d)
            
        if not valid:
            continue
            
        # Check if new_diffs are within bounds? 
        # They are <= x <= m.
        
        # Update needed
        # Intersect new_diffs with needed
        # new_needed = needed - set(new_diffs)
        
        # Optimization: Check if we can still cover the rest
        # If max(new_needed) > m, prune (already checked at top level, but state changes)
        
        new_diffs_set = set(new_diffs)
        new_needed = needed - new_diffs_set
        
        # If we covered all, success
        # Recurse
        result = _dfs(
            S + [x],
            diffs | new_diffs_set,
            new_needed,
            k,
            m,
            deadline
        )
        if result:
            return result
            
    return None

def _greedy():
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
