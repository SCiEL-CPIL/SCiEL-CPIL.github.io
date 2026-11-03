import random
import copy
import math

def construct(n_rows, k):
    # 1. Greedy Construction
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    # 2. Simulated Annealing
    # Objective: Minimize max(max(row) for row in rows)
    
    def get_max(rows):
        return max(max(r) for r in rows)

    best_rows = copy.deepcopy(rows)
    best_max = get_max(rows)
    
    # Maintain used_diffs set
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    T = 100.0
    T_min = 0.1
    cooling = 0.9995
    # Increase iterations significantly since each step is cheap
    # 120s limit. Python is slow. Let's estimate.
    # One step: O(k) checks. k=8. ~10 ops.
    # 10 million steps might take ~10-20s in Python if optimized.
    # Let's aim for 5-10 million iterations.
    iterations = 5000000 
    # But we need to be careful with overhead.
    
    random.seed(42)
    
    current_max = best_max
    
    for it in range(iterations):
        # Pick random row and element (index 1..k)
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, k)
        
        row = rows[ri]
        old_val = row[ei]
        
        # Determine valid range for new value
        lo = row[ei-1] + 1
        hi = row[ei+1] - 1 if ei < k else old_val + 50 # Allow some expansion
        
        # If the range is empty or very small, skip or try to expand
        if lo > hi:
            # Try to expand the range by allowing the neighbor to move? 
            # For now, just skip this move.
            T *= cooling
            continue
            
        # Sample new value. 
        # To help minimize max, we can bias towards lower values.
        # But SA needs to explore. Let's use uniform in [lo, hi].
        # However, if hi is very large, uniform might pick large values.
        # We can cap hi to current_max + 10 to avoid wandering too far.
        if hi > current_max + 20:
            hi = current_max + 20
            
        if lo > hi:
             T *= cooling
             continue

        new_val = random.randint(lo, hi)
        
        if new_val == old_val:
            T *= cooling
            continue
            
        # Calculate old diffs to remove
        old_diffs = []
        for j in range(len(row)):
            if j != ei:
                old_diffs.append(abs(old_val - row[j]))
                
        # Check new diffs
        valid = True
        new_diffs = []
        for j in range(len(row)):
            if j != ei:
                d = abs(new_val - row[j])
                # If d is in used_diffs, it's a conflict UNLESS d is one of the old_diffs
                # But we are removing old_diffs first logically.
                # So we check if d is in used_diffs.
                # Note: multiple j's might produce the same d? No, row is strictly increasing.
                # |new_val - row[j]| is unique for each j if new_val is between them?
                # Not necessarily. But within a row, differences are distinct?
                # The problem says "all differences between two entries of the same row are distinct over all rows".
                # This implies differences within a row must be distinct from each other too?
                # "all differences ... are distinct over all rows" usually means the global set of all differences is unique.
                # So yes, no two pairs (i,j) in any row can have the same difference.
                
                if d in used_diffs:
                    # It might be that d is in old_diffs. We need to be careful.
                    # Let's remove old_diffs from a temporary set or just check membership carefully.
                    pass
        
        # More robust check:
        # 1. Remove old_diffs from used_diffs? No, don't modify yet.
        # 2. Check if any new_diff is in used_diffs AND not in old_diffs.
        
        old_diffs_set = set(old_diffs)
        for j in range(len(row)):
            if j != ei:
                d = abs(new_val - row[j])
                if d in used_diffs and d not in old_diffs_set:
                    valid = False
                    break
                new_diffs.append(d)
                
        # Also need to ensure new_diffs don't conflict with each other?
        # Since they come from the same row and the row is strictly increasing,
        # |new_val - row[j1]| == |new_val - row[j2]| implies row[j1] == row[j2] (impossible) 
        # or new_val is the midpoint. If new_val is midpoint, d is same.
        # But row is strictly increasing integers.
        # If row = [0, 2, 4], new_val=2. Diffs: 2, 2. Conflict!
        # So we must check if new_diffs has duplicates.
        
        if valid:
            if len(set(new_diffs)) != len(new_diffs):
                valid = False
                
        if not valid:
            T *= cooling
            continue
            
        # If valid, calculate new_max
        # We only changed row[ri][ei].
        # new_max is max of:
        # 1. max of all other rows
        # 2. max of row[ri] with new_val
        
        # To be efficient, we can precompute max of other rows?
        # Or just calculate max(row[ri]) with new_val and compare with global max.
        
        # Let's just calculate the new max for the whole configuration.
        # Since n_rows=4, k=8, this is fast.
        
        # Apply move temporarily to calculate max?
        # Or just:
        max_row_new = max(row[:ei] + [new_val] + row[ei+1:])
        
        # We need max of all rows.
        # We can maintain a list of row_maxs.
        # Let's do it simply for now, optimize if needed.
        
        # Actually, let's just update the row and get max.
        # But we need to revert if rejected.
        
        # Let's use a helper to get max.
        # To avoid O(n*k) every time, we can track the max of each row.
        
        # Let's stick to simple O(n*k) for now, k=8, n=4 -> 32 ops.
        # 5 million iterations * 32 ops = 160 million ops. Might be too slow in Python.
        # Let's optimize the max calculation.
        
        # Maintain `row_maxes` list.
        # When we change row[ri][ei], update row_maxes[ri].
        
        # Let's refactor to maintain row_maxes.
        
        # For the sake of the edit, I will include the optimization in the code.
        
        # Let's restart the loop logic with row_maxes.
        
        pass # Placeholder, will rewrite the block below

    return best_rows
