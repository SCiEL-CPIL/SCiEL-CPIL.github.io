import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
    
    # We need at least 9 elements.
    # Let's search for a Sidon set of size 9 that covers 1..30.
    # If not found, try size 10, etc.
    
    best_solution = None
    best_max_elem = float('inf')
    
    def dfs(current_elements, current_mask, used_diffs, depth, max_depth):
        nonlocal best_solution, best_max_elem
        
        if time.time() > deadline:
            return
        
        # Pruning: if current max element is already >= best_max_elem, stop
        if current_elements and current_elements[-1] >= best_max_elem:
            return
            
        # Check if we have covered all target diffs
        if current_mask == target_mask:
            # We found a valid set.
            # We want to minimize the largest element.
            # Since we are building the set in increasing order, the last element is the largest.
            # But we might have more elements to add? No, if we have covered all diffs,
            # we can stop adding elements?
            # The problem says "at most 32 distinct integers". It doesn't say exactly n.
            # But adding more elements increases the largest element or keeps it same?
            # If we add an element larger than current max, max increases.
            # If we add an element smaller, it's not in increasing order.
            # So once we cover all diffs, the current set is a candidate.
            # However, we might be able to reduce the max element by choosing different previous elements?
            # The DFS explores all possibilities.
            # So if we find a solution, we update best.
            if current_elements[-1] < best_max_elem:
                best_max_elem = current_elements[-1]
                best_solution = list(current_elements)
            return
        
        # If we have reached max_depth, stop
        if depth >= max_depth:
            return
            
        # Heuristic: Try to add elements that cover many missing diffs
        # But we need to ensure Sidon property.
        
        last_elem = current_elements[-1] if current_elements else -1
        
        # Estimate lower bound for next element
        # We need to cover missing diffs.
        # Missing diffs: target_mask & ~current_mask
        # Let missing = [d for d in range(1, 31) if not (current_mask & (1 << d))]
        
        # To cover a difference d, we need x - a = d for some a in current_elements.
        # So x = a + d.
        # The smallest possible x is min(a + d for a in current_elements for d in missing).
        # But x must be > last_elem.
        
        # Let's iterate x from last_elem + 1 to some upper bound.
        # Upper bound: best_max_elem - 1 (if we want to improve) or a reasonable limit.
        # If best_solution is None, we can go up to a larger limit.
        
        upper_bound = best_max_elem - 1 if best_solution else 100
        
        if upper_bound < last_elem + 1:
            return
            
        for x in range(last_elem + 1, upper_bound + 1):
            # Check Sidon property
            new_diffs = set()
            valid = True
            for a in current_elements:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Update mask
            new_mask = current_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << d)
            
            # Recurse
            dfs(current_elements + [x], new_mask, used_diffs.union(new_diffs), depth + 1, max_depth)
            
            if time.time() > deadline:
                return
                
            # Pruning: if we found a solution with max < x, we can break?
            # No, because a larger x might lead to a solution with smaller final max?
            # Unlikely, but possible if the path with smaller x is a dead end.
            # However, if we found a solution, best_max_elem is updated.
            # The loop condition `upper_bound` uses best_max_elem, so it will shrink.
            
    # Start with [0]
    dfs([0], 0, set(), 0, 10) # Max depth 10 (size 11)
    
    if best_solution:
        return best_solution
    
    # Fallback: Greedy construction
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            if v + d not in a:
                candidates.add(v + d)
            if v - d >= 0 and v - d not in a:
                candidates.add(v - d)
        
        for x in sorted(candidates):
            trial = sorted(a + [x])
            d_list = get_diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                d_list = get_diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
