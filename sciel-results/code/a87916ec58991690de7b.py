import sys
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    best_solution = None
    best_max_val = float('inf')
    
    # We try sizes k starting from 9
    for k in range(9, 12):
        if time.time() > deadline - 5:
            break
            
        # DFS state
        # current_set: list of chosen integers
        # used_diffs: set of differences generated so far
        # missing_diffs: set of target diffs not yet generated
        
        def dfs(current_set, used_diffs, missing_diffs, depth):
            nonlocal best_solution, best_max_val
            
            if time.time() > deadline - 2:
                return False
                
            if not missing_diffs:
                # Found a valid set
                current_max = current_set[-1]
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_solution = list(current_set)
                return True # Stop exploring this branch, we have a solution
            
            if depth == k:
                return False
                
            # Pruning: Can we cover remaining missing diffs?
            remaining_marks = k - depth
            # Max new diffs possible:
            # New marks form pairs with each other: r(r-1)/2
            # New marks form pairs with existing marks: r * depth
            max_new_diffs = remaining_marks * (remaining_marks - 1) // 2 + remaining_marks * depth
            if len(missing_diffs) > max_new_diffs:
                return False
            
            # Lower bound for next element
            # The next element x must be > current_set[-1]
            # Also, to cover a missing diff d, we need x - y = d or y - x = d (impossible if x is new max) 
            # So x = y + d for some y in current_set and d in missing_diffs.
            # Thus x must be in {y + d | y in current_set, d in missing_diffs}
            # OR x can be anything if it doesn't cover a missing diff immediately?
            # No, if x doesn't cover a missing diff, we waste a mark.
            # It is better to restrict candidates to those that cover at least one missing diff?
            # Not necessarily, sometimes you need a "filler" mark to enable future diffs?
            # But in a Sidon set, every new mark adds diffs with all previous marks.
            # If none of those diffs are in missing_diffs, it's a waste of potential coverage.
            # So we can restrict candidates to $x$ such that at least one $x-y$ is in missing_diffs.
            
            candidates = set()
            last_val = current_set[-1]
            
            for y in current_set:
                for d in missing_diffs:
                    x = y + d
                    if x > last_val:
                        candidates.add(x)
            
            # Also, we might need to consider x that doesn't cover missing diffs if we are stuck?
            # But if we are stuck, we prune.
            # Let's sort candidates to try smaller ones first (to minimize diameter)
            # But maybe a slightly larger x covers more missing diffs?
            # Let's try all candidates, but prioritize those that reduce missing_diffs more?
            # For now, just sort by value.
            
            for x in sorted(candidates):
                # Check Sidon property
                valid = True
                new_diffs = []
                for y in current_set:
                    diff = x - y
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if not valid:
                    continue
                
                # Check internal duplicates in new_diffs
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Update state
                new_used_diffs = used_diffs.union(new_diffs)
                new_missing_diffs = missing_diffs.difference(new_diffs)
                
                # Pruning: if current max x is already >= best_max_val, no need to go deeper?
                # Actually, if x >= best_max_val, any superset will have max >= x >= best_max_val.
                # So we can prune.
                if x >= best_max_val:
                    continue
                
                if dfs(current_set + [x], new_used_diffs, new_missing_diffs, depth + 1):
                    # We found a solution, but maybe a better one exists?
                    # Since we want to minimize max, and we are doing DFS,
                    # we should continue searching for smaller max?
                    # But we pruned x >= best_max_val.
                    # So any further solution in this branch will have max < best_max_val?
                    # No, the current x is < best_max_val, but deeper x might be larger.
                    # However, if we find a solution, we update best_max_val.
                    # To find the absolute minimum, we need to explore all branches with potential < best_max_val.
                    # So we should NOT return True immediately, unless we are sure we found the global min?
                    # No, DFS finds one solution. To find the min, we need to keep searching.
                    # But if we return True, we stop this branch.
                    # We should return False to keep searching?
                    # Or we can structure it as: find ANY solution with max < current best.
                    # Let's modify: dfs returns True if a solution with max < best_max_val is found in this subtree.
                    # And we update best_max_val when found.
                    pass
            
            return False
        
        # Initialize
        initial_set = [0]
        initial_used = set()
        initial_missing = set(target_diffs)
        
        # We need to handle the first step specially?
        # depth 1 means we have 1 element. We need to add k-1 more.
        # The candidates for the second element x_2:
        # x_2 - 0 = x_2.
        # x_2 must be in missing_diffs? Yes, to be useful.
        
        dfs(initial_set, initial_used, initial_missing, 1)
        
        if best_solution:
            return best_solution
            
    # Fallback: return a valid set even if not optimal, to avoid failure
    # The seed program returns a valid set.
    # If our search fails, we should return the seed's result or a known valid set.
    # Let's just return the seed's logic if we time out or fail to find better.
    # But the task says "Minimize the largest element".
    # If we don't find a better one, we should return the best we found.
    # If we found nothing, we must return a valid set.
    
    # Re-run seed logic as fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

if __name__ == "__main__":
    # Test
    res = construct()
    print(res)
