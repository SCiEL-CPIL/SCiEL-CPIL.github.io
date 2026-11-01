import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    diffs_per_row = row_len * (row_len - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Function to check if a set of differences can form a valid Golomb ruler
    def check_ruler(S, k):
        """
        S: set of differences.
        Returns list of points [0, a1, ..., ak] or None.
        """
        if len(S) != k * (k + 1) // 2:
            return None
        
        S = set(S)
        max_diff = max(S)
        # a0 = 0, ak = max_diff
        # We need to find a1...a{k-1}
        
        # Heuristic: The smallest difference must be a1 - a0 = a1
        # So a1 must be min(S)
        # Wait, is min(S) always a1?
        # Yes, because a1 is the smallest positive difference from 0.
        # Any other difference a_j - a_i (j>i>=1) is >= a_j - a_1 > a_1 if j>1?
        # No, a_2 - a_1 could be smaller than a_1?
        # Example: 0, 2, 3. Diffs: 2, 1, 3. Min is 1. a1=2.
        # So min(S) is not necessarily a1.
        
        # Backtracking search
        # Points: p[0]=0, p[k]=max_diff
        # We need to fill p[1]...p[k-1]
        
        # To speed up, we can use the fact that all differences must be in S.
        # And all elements of S must be generated.
        
        # Let's use a simple recursive search with pruning.
        # State: current points, used differences.
        
        # Optimization: 
        # The largest difference is max_diff.
        # The second largest must be max_diff - p[1] or p[k-1].
        
        # We will try to construct the ruler by choosing p[1], then p[2], etc.
        # But we don't know the order.
        
        # Alternative: 
        # Pick p[1] from S.
        # Then p[2] from S \ {p[1], p[2]-p[1], ...}
        
        # This is still complex. Let's use a randomized restart approach for the partitioning,
        # and a dedicated solver for the ruler check.
        
        # For the ruler check, we can use a constraint solver approach.
        # Given S, find p[0..k] such that {p[j]-p[i]} = S.
        
        # We know p[0]=0, p[k]=max_diff.
        # We need to find p[1..k-1].
        
        # Let's try to deduce p[1].
        # p[1] must be in S.
        # Also, max_diff - p[1] must be in S (as p[k]-p[1]).
        
        candidates_p1 = [x for x in S if (max_diff - x) in S]
        
        # Sort candidates to try "small" ones first? Or random?
        random.shuffle(candidates_p1)
        
        for p1 in candidates_p1:
            # Try to build the rest
            points = [0, p1]
            used_diffs = {p1}
            # We need to add k-2 more points.
            
            # This is a partial construction.
            # We need to ensure we can complete it.
            
            # For simplicity and speed, we will use a recursive function.
            
            def solve(points, used_diffs, remaining_S):
                if len(points) == row_len:
                    # Check if all diffs are used
                    if used_diffs == set(remaining_S):
                        return points
                    return None
                
                last = points[-1]
                # Try adding a new point x
                # x must be such that x - p_i in S for all i
                # And x - p_i not in used_diffs
                # And x - p_i in remaining_S
                
                # Heuristic: try x such that x - last is in remaining_S?
                # Or try all x in remaining_S?
                
                # To be efficient, we only consider x that are in S (as differences from 0? No, x is a point).
                # x must be < max_diff.
                
                # We can restrict x to be in S? No.
                # But x - 0 = x must be in S. So x must be in S.
                
                # So x is a candidate from S.
                
                # To reduce branching, we can prioritize x that are "large" or "small"?
                
                # Let's pick a random x from remaining_S that is < max_diff
                # and such that x - p_i in S for all i.
                
                # This check is expensive.
                
                # Better:
                # The next point x must satisfy:
                # 1. x in S (since x-0=x)
                # 2. max_diff - x in S (since max_diff-x = p_k - x)
                # 3. x - p_i in S for all existing p_i.
                
                # We will iterate through candidates.
                
                candidates = [x for x in remaining_S if x < max_diff and (max_diff - x) in remaining_S]
                # Also check against existing points
                valid_candidates = []
                for x in candidates:
                    ok = True
                    for p in points:
                        d = x - p
                        if d not in remaining_S:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(x)
                
                if not valid_candidates:
                    return None
                
                random.shuffle(valid_candidates)
                
                for x in valid_candidates:
                    new_points = points + [x]
                    new_used = set(used_diffs)
                    new_remaining = set(remaining_S)
                    
                    # Update used and remaining
                    for p in points:
                        d = x - p
                        new_used.add(d)
                        new_remaining.discard(d)
                        
                    # Also discard x-0 and max_diff-x?
                    # x-0 = x is in new_used.
                    # max_diff - x is a difference involving the last point, which is not added yet?
                    # Wait, max_diff is p_k. We are adding x as a new point.
                    # The difference p_k - x is NOT generated yet because p_k is not in points.
                    # So we should NOT discard max_diff-x from remaining_S yet.
                    # We only discard differences between new point and EXISTING points.
                    
                    res = solve(new_points, new_used, new_remaining)
                    if res:
                        return res + [max_diff]
                        
                return None
            
            # Initial call
            # points = [0, p1]
            # used_diffs = {p1}
            # remaining_S = S - {p1}
            # Note: max_diff is not in points yet, so differences involving max_diff are not used.
            
            res = solve([0, p1], {p1}, S - {p1})
            if res:
                return res
        
        return None

    start_time = time.time()
    timeout = 110
    
    # We want to minimize M.
    # Start from lower bound.
    for M in range(total_diffs, total_diffs + 100):
        if time.time() - start_time > timeout:
            break
            
        # We have M available differences: 1..M
        # We need to select total_diffs of them and partition into n_rows sets.
        
        # If M == total_diffs, we must use all.
        # If M > total_diffs, we can skip some.
        
        # Heuristic: Try to use the smallest differences first.
        # So we pick a subset of size total_diffs from 1..M.
        # If M == total_diffs, subset is 1..M.
        # If M > total_diffs, we can randomly drop M - total_diffs elements.
        
        # To respect "bin-packing" and "largest first":
        # We will randomly partition the selected differences.
        
        found = False
        for attempt in range(500):
            if time.time() - start_time > timeout:
                break
                
            # Select differences
            if M == total_diffs:
                selected_diffs = list(range(1, M + 1))
            else:
                # Randomly drop some
                all_diffs = list(range(1, M + 1))
                random.shuffle(all_diffs)
                selected_diffs = all_diffs[:total_diffs]
            
            # Partition into n_rows sets
            random.shuffle(selected_diffs)
            sets = [[] for _ in range(n_rows)]
            for i, d in enumerate(selected_diffs):
                sets[i % n_rows].append(d)
            
            # Check validity
            rows = []
            valid = True
            for s in sets:
                ruler = check_ruler(set(s), k)
                if ruler is None:
                    valid = False
                    break
                rows.append(ruler)
            
            if valid:
                return rows
        
    # Fallback to greedy if search fails
    rows = [[0] for _ in range(n_rows)]
    used = set()
    for _ in range(n_rows * k):
        i = min(range(n_rows), key=lambda idx: (len(rows[idx]), rows[idx][-1]))
        if len(rows[i]) >= row_len:
            continue
        x = rows[i][-1] + 1
        while True:
            new_diffs = {x - v for v in rows[i]}
            if not (new_diffs & used):
                rows[i].append(x)
                used.update(new_diffs)
                break
            x += 1
    return rows
