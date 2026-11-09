import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # We need a Sidon set A of size k (start with 9)
    # such that 1..30 are in D(A).
    # Minimize max(A).
    
    best_solution = None
    best_max_val = float('inf')
    
    # Lower bound for M is 36 for k=9.
    # We can try to find a solution with M close to 36.
    
    # Backtracking state:
    # a: list of selected elements, sorted, a[0]=0
    # used_diffs: set of differences generated so far
    # required_diffs: set of differences still needed from 1..30
    
    def backtrack(a, used_diffs, required_diffs):
        nonlocal best_solution, best_max_val, deadline
        
        if time.time() > deadline:
            return False # Timeout
        
        # Check if we have enough elements to cover remaining required diffs
        # If we add x more elements, max new diffs = x * len(a) + x*(x-1)/2
        # We need to cover len(required_diffs)
        
        k = len(a)
        if k >= 9:
            # If we have 9 elements, we must have covered all required diffs
            if not required_diffs:
                m = a[-1]
                if m < best_max_val:
                    best_max_val = m
                    best_solution = a.copy()
                return True # Found a valid one, but maybe a smaller one exists? 
                # Since we are searching for min M, we can stop if we find one with current M?
                # No, we are not iterating M. We are adding elements.
                # We should prune if max(a) >= best_max_val.
            else:
                return False # 9 elements but missed some diffs
        
        if a[-1] >= best_max_val:
            return False
            
        # Heuristic: Pick the next element.
        # To keep M small, we should try small values for the next element.
        # But we also need to generate required diffs.
        
        # Try adding a new element x.
        # x must be > a[-1].
        # x must be < best_max_val (prune).
        # Also, to be efficient, we can bound x.
        # The largest required diff is 30.
        # If we don't cover it yet, we need x - a_i = 30 for some i, or a_j - a_i = 30.
        
        # Let's just iterate x from a[-1] + 1 to best_max_val - 1 (or some bound).
        # Bound for x:
        # We need to cover required_diffs.
        # If x is too large, we might exceed best_max_val.
        
        # Optimization: 
        # If required_diffs is not empty, the next element x should ideally create a diff in required_diffs.
        # Let's prioritize x such that (x - a_i) is in required_diffs for some i.
        
        candidates = set()
        for d in required_diffs:
            for v in a:
                x = v + d
                if x > a[-1] and x < best_max_val:
                    candidates.add(x)
        
        # Also try small x to keep diameter small, even if it doesn't immediately cover a required diff?
        # No, if it doesn't cover a required diff, it "wastes" a slot.
        # With k=9, we have 36 slots. 30 are required. 6 are spare.
        # We can afford 6 "waste" diffs.
        # But it's safer to prioritize covering required diffs.
        
        # Add a range of small x to candidates to ensure we don't miss compact solutions
        # where diffs are formed by later elements?
        # Actually, if x doesn't form a required diff, it forms a "spare" diff.
        # We have limited spares.
        
        # Let's sort candidates by value to try smaller M first.
        for x in sorted(candidates):
            # Check Sidon property
            new_diffs = []
            valid = True
            for v in a:
                d = x - v
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_used = used_diffs.union(new_diffs)
            new_required = required_diffs.difference(new_diffs)
            
            if backtrack(a + [x], new_used, new_required):
                # If we found a solution, we can return immediately if we are doing BFS on M?
                # Here we are doing DFS. 
                # Since we prune x < best_max_val, and we update best_max_val,
                # finding ANY solution with a smaller M will update best.
                # We should continue searching to find if an even smaller M exists?
                # No, we iterate x in increasing order. The first solution found for a given branch 
                # might not be the global min M.
                # But if we return True, we stop this branch.
                # We need to explore other x values?
                # If we return True, the parent will also return True?
                # We want the GLOBAL min M.
                # So we should NOT return True immediately unless we are sure we found the global min.
                # Since we don't know the global min, we must continue searching 
                # with the updated best_max_val.
                
                # However, if we just found a solution, best_max_val is updated.
                # The loop continues to try other x's (which are larger, so won't improve M for this prefix? 
                # Wait, x is the LAST element. The max of the set is x.
                # If we find a solution with x=40, best_max_val becomes 40.
                # Then we try x=41? No, x is determined by the branch.
                # We iterate x in sorted order.
                # If we find a solution at x=40, we update best to 40.
                # We should stop trying larger x in this branch?
                # Yes, because any further x in this branch will have max > 40.
                # But what if a different branch (different previous elements) leads to a smaller max?
                # That will be explored in other iterations of the parent loop.
                # So returning True (meaning "found a solution in this subtree") is correct 
                # to prune the rest of THIS subtree, but we must not stop the ENTIRE search.
                
                # However, if we return True, the parent might also return True and stop?
                # We need a flag or just let it bubble up.
                # If any branch finds a solution, we update best.
                # We should continue the search to see if we can find a smaller one.
                # So we should NOT return True to stop the whole search.
                # We should return a signal that "a solution was found" but let the search continue 
                # with the tighter bound.
                
                # Actually, if we return True, the caller stops iterating further x's.
                # This is correct for the current prefix.
                # But we need to ensure the top-level call doesn't stop after the FIRST solution found.
                # The top-level call will just return the best_solution found.
                
                return True # Prune this branch, but search continues in siblings
        
        return False

    # Initial state
    a = [0]
    used_diffs = set()
    required_diffs = set(range(1, 31))
    
    # We need to ensure we find a solution.
    # If backtrack doesn't find one with k=9, try k=10?
    # But k=9 is theoretically better.
    
    # Run search
    backtrack(a, used_diffs, required_diffs)
    
    if best_solution:
        return best_solution
    else:
        # Fallback: Return a valid set, even if not optimal
        # The seed program logic can be used as fallback
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
