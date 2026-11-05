import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 128
    
    # Precompute all 4-term arithmetic progressions
    # An AP is defined by (a, d) such that a, a+d, a+2d, a+3d < n
    # The constraint is on the set of indices {a, a+d, a+2d, a+3d}
    # We will store for each index i, the list of APs it participates in, 
    # and specifically, for forward checking, we need to know which other indices 
    # in the AP are unassigned.
    
    # Let's represent the problem as assigning colors to indices 0..n-1.
    # Constraint for an AP (i0, i1, i2, i3): len({c[i0], c[i1], c[i2], c[i3]}) >= 3.
    
    # To make backtracking efficient, we need to quickly determine if a partial assignment
    # violates a constraint.
    # A constraint is violated if:
    # 1. All 4 are assigned and distinct count < 3.
    # 2. 3 are assigned and they are all the same (then the 4th must be different? No, if 3 are X, 
    #    the set is {X, c4}. Size 1 or 2. So if 3 are X, the 4th MUST be different from X? 
    #    Wait. If c0=c1=c2=X, then {X, X, X, c3} has size 1 (if c3=X) or 2 (if c3!=X). 
    #    Both are < 3. So if 3 terms of an AP are the same, the AP is INVALID regardless of the 4th.
    #    Therefore, a necessary condition is: No 3 terms of any 4-term AP can be identical.
    #    This is a strong local constraint.
    
    # Let's build a list of all 4-term APs.
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each index, store the APs it is part of.
    # Also, for each AP, store its indices.
    index_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for i in ap:
            index_to_aps[i].append(idx)
    
    def solve_with_K(K, max_restarts=1000, max_nodes_per_restart=100000):
        colors = [-1] * n
        domains = [list(range(K)) for _ in range(n)]
        
        # We will use a recursive backtracking search.
        # To avoid recursion depth issues and overhead, we might use an iterative stack, 
        # but Python recursion is fine for depth 128 if we set recursion limit.
        sys.setrecursionlimit(10000)
        
        def is_consistent(i, val):
            # Check if assigning val to i violates any constraint involving i
            # We only need to check APs where i is the last assigned or 
            # where all other indices are assigned.
            # Actually, for forward checking, we just need to check if the assignment
            # makes any AP "impossible" or "violated".
            
            for ap_idx in index_to_aps[i]:
                i0, i1, i2, i3 = aps[ap_idx]
                # Get colors of other indices in this AP
                others = []
                unassigned_count = 0
                for j in (i0, i1, i2, i3):
                    if j == i:
                        continue
                    if colors[j] == -1:
                        unassigned_count += 1
                    else:
                        others.append(colors[j])
                
                # Case 1: All other 3 are assigned.
                if unassigned_count == 0:
                    distinct = set(others)
                    if val in distinct:
                        # distinct size is 1, 2, or 3.
                        # If distinct size is 1: {X, X, X, X} or {X, X, X, Y}.
                        # If val == X, set is {X}. Size 1. Invalid.
                        # If val != X, set is {X, Y}. Size 2. Invalid.
                        # So if all 3 others are same, ANY val is invalid?
                        # Yes, as derived before.
                        if len(distinct) == 1:
                            return False
                        # If distinct size is 2: {X, Y}.
                        # If val == X or Y, set is {X, Y}. Size 2. Invalid.
                        # If val != X and val != Y, set is {X, Y, val}. Size 3. Valid.
                        if len(distinct) == 2:
                            if val in distinct:
                                return False
                        # If distinct size is 3: {X, Y, Z}.
                        # Any val results in size 3 or 4. Valid.
                        # So if len(distinct) == 3, always valid.
                        
                # Case 2: 2 others assigned, 1 unassigned.
                elif unassigned_count == 1:
                    # We need to ensure that the unassigned variable still has at least one valid value.
                    # This is the forward checking part.
                    # The unassigned variable j_unassigned must be able to take a value v such that
                    # the whole AP is valid.
                    # Let the assigned colors in this AP be A (size 2) and the new val.
                    # The set of colors so far is S = {val} U others.
                    # The unassigned variable u must pick v such that len(S U {v}) >= 3.
                    
                    # Find the unassigned index
                    u_idx = -1
                    for j in (i0, i1, i2, i3):
                        if j != i and colors[j] == -1:
                            u_idx = j
                            break
                    
                    # Determine which values for u_idx are valid given current assignment + val
                    valid_vals_for_u = []
                    for v in domains[u_idx]:
                        full_set = set(others) | {val, v}
                        if len(full_set) >= 3:
                            valid_vals_for_u.append(v)
                    
                    if len(valid_vals_for_u) == 0:
                        return False
                    else:
                        # Update domain of u_idx (forward checking)
                        # We'll just check if the intersection with current domain is empty
                        # domains[u_idx] is the current possible values.
                        # If valid_vals_for_u is a subset of domains[u_idx] or vice versa?
                        # Actually, we just need to know if domains[u_idx] contains any valid value.
                        # We don't modify domains in-place to avoid backtracking complexity in this simple version,
                        # but we can prune if we want. For now, just check non-empty.
                        # To be more efficient, we could update domains.
                        pass
                        
                # Case 3: 1 or 0 others assigned.
                # If 1 assigned: The other 2 are unassigned. 
                # It's very unlikely to be constrained immediately unless the 1 assigned is part of a pattern?
                # Actually, if only 1 is assigned, say X, and we assign Y to i.
                # The remaining 2 can always be picked to make 3 distinct?
                # Yes, we have K colors. If K>=3, we can always pick Z and W to make {X,Y,Z,W} have >=3 distinct.
                # Unless K=2? But we start with K=3.
                # So no constraint propagation needed for unassigned_count > 1.
                
                # Wait, there is a subtle constraint: 
                # "No 3 terms of an AP can be identical".
                # If we assign val to i, and two others in the AP are already val, 
                # then we have 3 identical terms. This is INVALID.
                # This is covered in Case 1 (if unassigned_count == 0) and Case 2?
                # In Case 2 (2 assigned, 1 unassigned):
                # If others = [val, val], then distinct = {val}.
                # Then for any v, full_set = {val, v}. Size <= 2. 
                # So valid_vals_for_u will be empty. So is_consistent returns False.
                # Correct.
                
            return True

        def backtrack(idx):
            # idx is the next index to assign? 
            # No, we should use dynamic variable ordering or just fixed order?
            # Fixed order 0..n-1 is simple. Let's try that first.
            # If we use fixed order, "idx" is the index in the array.
            
            if idx == n:
                return True
            
            # Heuristic: Try values in random order to break symmetry
            # Or Least Constraining Value?
            # Let's try random order first, it's often good for CSPs with symmetry.
            values = domains[idx][:]
            random.shuffle(values)
            
            for val in values:
                if not is_consistent(idx, val):
                    continue
                
                # Assign
                colors[idx] = val
                # Save domain state for backtracking? 
                # Since we don't modify domains in is_consistent (we just check), 
                # we don't need to restore domains.
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                colors[idx] = -1
            
            return False

        # Run multiple restarts to handle local minima / bad initial choices
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Randomize domains or just shuffle values in backtrack?
            # Shuffling values in backtrack is enough for random restarts if we reset colors.
            colors = [-1] * n
            # We can also randomize the order of indices if we do dynamic variable ordering,
            # but let's stick to fixed order 0..n-1 for now to keep it simple and fast.
            
            # Optimization: We can use a "watched literals" or similar, but Python is slow.
            # Let's just run the backtracking.
            
            if backtrack(0):
                return colors[:]
            
            # If we get here, this restart failed.
            
        return None

    # Try K=3
    # Note: The problem states "Minimize the number of labels actually used".
    # It is known that for 4-term APs, 3 colors might not be enough for large N?
    # Actually, van der Waerden number W(3, 4) is 75. 
    # This means any 3-coloring of {1..75} contains a monochromatic 4-term AP.
    # Wait, the constraint is "AT LEAST 3 distinct colors". 
    # A monochromatic AP has 1 distinct color. So it violates the constraint.
    # So if W(3, 4) = 75, then for N >= 75, it is IMPOSSIBLE to color with 3 colors 
    # such that every 4-term AP has >= 3 distinct colors?
    # Let's re-read carefully.
    # "Every nonconstant four-term progression ... must use AT LEAST three distinct colors."
    # If W(3, 4) = 75, then any 3-coloring of [1, 75] has a monochromatic 4-term AP.
    # A monochromatic AP has 1 distinct color. 1 < 3. So it violates the condition.
    # Therefore, for N >= 75, we CANNOT use only 3 colors.
    # We must use at least 4 colors?
    # Let's check W(4, 4). It is larger.
    # So K=3 will fail for N=128.
    # We should start trying from K=4, or K=5, etc.
    
    # Let's estimate the minimum K.
    # If W(k, 4) is the least N such that any k-coloring has a mono 4-AP.
    # We need N < W(k, 4) to guarantee existence? No, we just need ONE coloring.
    # If N < W(k, 4), it is NOT guaranteed that a coloring exists. 
    # W(k, 4) is the threshold where it becomes IMPOSSIBLE to avoid mono 4-APs.
    # So if N < W(k, 4), it is POSSIBLE to avoid mono 4-APs.
    # Avoiding mono 4-APs is a WEAKER condition than "every 4-AP has >= 3 colors".
    # "Every 4-AP has >= 3 colors" implies "No 4-AP is monochromatic" AND "No 4-AP is 2-colored".
    # So the constraint is stronger than just avoiding mono APs.
    # So K might need to be larger.
    
    # Let's try K=4, 5, 6...
    
    best_solution = None
    for K in range(3, 10):
        if time.time() > deadline:
            break
        
        # Increase restarts for smaller K as they are harder
        max_restarts = 500 if K <= 4 else 100
        max_nodes = 200000
        
        sol = solve_with_K(K, max_restarts, max_nodes)
        if sol is not None:
            best_solution = sol
            break # Found a solution with this K, which is minimal for this loop
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: Random greedy if CSP fails (should not happen if K is large enough)
    # If K is large enough (e.g., K > N/4), a random greedy will likely work.
    for attempt in range(1000):
        if time.time() > deadline:
            break
        colors = [-1] * n
        success = True
        for i in range(n):
            forbidden = set()
            for d in range(1, i // 3 + 1):
                c0 = colors[i - 3 * d]
                c1 = colors[i - 2 * d]
                c2 = colors[i - d]
                distinct = {c0, c1, c2}
                if len(distinct) == 1:
                    # Dead end, but we assume K is large enough that we don't hit this?
                    # Actually, if we hit this, we failed.
                    success = False
                    break
                elif len(distinct) == 2:
                    forbidden.update(distinct)
                # else len 3, no forbidden
            
            if not success:
                break
            
            valid = [c for c in range(128) if c not in forbidden]
            if not valid:
                success = False
                break
            colors[i] = random.choice(valid)
        
        if success:
            # Verify
            valid = True
            for i in range(n):
                for d in range(1, i // 3 + 1):
                    if len({colors[i], colors[i-d], colors[i-2*d], colors[i-3*d]}) < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return colors

    # Last resort: return a valid coloring with many colors
    # Use 128 distinct colors? No, labels 0..127.
    # If we use 128 colors, every AP has 4 distinct colors. Valid.
    # But we want to minimize.
    # Just return identity if nothing else works.
    return list(range(n))

# Keep the function signature
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    index_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for i in ap:
            index_to_aps[i].append(idx)

    def solve_with_K(K, max_restarts=1000):
        colors = [-1] * n
        
        def is_consistent(i, val):
            for ap_idx in index_to_aps[i]:
                i0, i1, i2, i3 = aps[ap_idx]
                others = []
                unassigned_count = 0
                u_idx = -1
                for j in (i0, i1, i2, i3):
                    if j == i:
                        continue
                    if colors[j] == -1:
                        unassigned_count += 1
                        u_idx = j
                    else:
                        others.append(colors[j])
                
                if unassigned_count == 0:
                    distinct = set(others)
                    if len(distinct) == 1:
                        return False
                    if len(distinct) == 2:
                        if val in distinct:
                            return False
                elif unassigned_count == 1:
                    # Check if u_idx has any valid value
                    valid_vals = []
                    for v in range(K):
                        full_set = set(others) | {val, v}
                        if len(full_set) >= 3:
                            valid_vals.append(v)
                    if not valid_vals:
                        return False
            return True

        def backtrack(idx):
            if idx == n:
                return True
            
            values = list(range(K))
            random.shuffle(values)
            
            for val in values:
                if not is_consistent(idx, val):
                    continue
                
                colors[idx] = val
                if backtrack(idx + 1):
                    return True
                colors[idx] = -1
            
            return False

        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            colors = [-1] * n
            if backtrack(0):
                return colors[:]
        return None

    best_solution = None
    for K in range(3, 10):
        if time.time() > deadline:
            break
        max_restarts = 500 if K <= 4 else 100
        sol = solve_with_K(K, max_restarts)
        if sol is not None:
            best_solution = sol
            break

    if best_solution is not None:
        return best_solution

    # Fallback
    for attempt in range(1000):
        if time.time() > deadline:
            break
        colors = [-1] * n
        success = True
        for i in range(n):
            forbidden = set()
            for d in range(1, i // 3 + 1):
                c0 = colors[i - 3 * d]
                c1 = colors[i - 2 * d]
                c2 = colors[i - d]
                distinct = {c0, c1, c2}
                if len(distinct) == 1:
                    success = False
                    break
                elif len(distinct) == 2:
                    forbidden.update(distinct)
            
            if not success:
                break
            
            valid = [c for c in range(128) if c not in forbidden]
            if not valid:
                success = False
                break
            colors[i] = random.choice(valid)
        
        if success:
            valid = True
            for i in range(n):
                for d in range(1, i // 3 + 1):
                    if len({colors[i], colors[i-d], colors[i-2*d], colors[i-3*d]}) < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return colors

    return list(range(n))

