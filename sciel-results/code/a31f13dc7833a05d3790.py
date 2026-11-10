# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Precompute lines and incidence.
    2. Use a fast greedy algorithm to build a large set S (complement of N) such that 
       every point x in F_49^2 has at least one line through x that is "S-clean" 
       (contains no points of S other than possibly x).
    3. The condition for x to be satisfied is that there exists a line L through x 
       such that (L \\ {x}) is disjoint from S.
    
    We maximize |S| to maximize the score (which is proportional to 2401 - |N| = |S|).
    """
    p_prime = 7
    q = 49
    N_points = q * q  # 2401
    
    # --- Field Arithmetic F_49 ---
    # Represent elements as (a, b) where value is a + b*alpha, alpha^2 = 3.
    # idx = a * 7 + b.
    
    def add_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        return ((a1 + a2) % 7) * 7 + (b1 + b2) % 7

    def mul_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        # (a1 + b1*a) * (a2 + b2*a) = a1a2 + (a1b2 + b1a2)a + b1b2 a^2
        # a^2 = 3
        real = (a1 * a2 + 3 * b1 * b2) % 7
        imag = (a1 * b2 + b1 * a2) % 7
        return real * 7 + imag

    def inv_idx(i):
        # i = a + b*a. Norm = a^2 - 3b^2. Inv = (a - b*a) / Norm.
        a, b = divmod(i, 7)
        norm = (a * a - 3 * b * b) % 7
        if norm == 0:
            return 0 # Should not happen for non-zero i in field
        # Inverse of norm in F_7
        norm_inv = pow(norm, -1, 7) if norm != 0 else 0
        real = (a * norm_inv) % 7
        imag = (6 * b * norm_inv) % 7  # -b
        return real * 7 + imag

    # Precompute addition and multiplication tables for speed?
    # 49x49 tables are small.
    add_table = np.zeros((49, 49), dtype=np.int32)
    mul_table = np.zeros((49, 49), dtype=np.int32)
    inv_table = np.zeros(49, dtype=np.int32)
    
    for i in range(49):
        for j in range(49):
            add_table[i, j] = add_idx(i, j)
            mul_table[i, j] = mul_idx(i, j)
        if i != 0:
            inv_table[i] = inv_idx(i)
        else:
            inv_table[i] = 0
            
    # --- Point Representation ---
    # Point P = (x, y) where x, y in F_49.
    # Index in 2D array: p_idx = x * 49 + y.
    
    # --- Line Representation ---
    # Lines are defined by slope m in F_49 (49 slopes) and vertical lines.
    # Total 50 lines through any point.
    # We represent a line by (type, param).
    # Type 0: Vertical. Param x. Line: {(x, y) for y in F_49}.
    # Type 1: Slope m. Param c. Line: {(x, y) for x in F_49, y = m*x + c}.
    
    # We need to iterate through all lines efficiently.
    # There are 50 * 49 = 2450 lines.
    
    # For the greedy algorithm, we need to know for each point p, 
    # which lines pass through it.
    # And for each line L, which points are on it.
    
    # Let's precompute the points on each line.
    # lines: list of 2450 arrays of 49 point indices.
    
    lines = []
    line_points = [] # list of lists of point indices
    
    # Vertical lines
    for x in range(49):
        pts = []
        for y in range(49):
            pts.append(x * 49 + y)
        line_points.append(np.array(pts, dtype=np.int32))
        
    # Sloped lines
    for m in range(49):
        for c in range(49):
            pts = []
            for x in range(49):
                # y = m*x + c
                y = add_idx(mul_table[m, x], c)
                pts.append(x * 49 + y)
            line_points.append(np.array(pts, dtype=np.int32))
            
    # Precompute: for each point, list of line indices passing through it.
    point_lines = [[] for _ in range(2401)]
    for l_idx, pts in enumerate(line_points):
        for p in pts:
            point_lines[p].append(l_idx)
            
    # Convert to numpy arrays for faster access? 
    # List of lists is probably fine if we just iterate.
    # But checking "is line disjoint from S" requires checking if any point in line is in S.
    
    # --- Greedy Algorithm ---
    
    def greedy_build(seed):
        rng = np.random.RandomState(seed)
        # Order of points to consider adding to S
        order = np.arange(2401)
        rng.shuffle(order)
        
        in_S = np.zeros(2401, dtype=bool)
        # valid_count[x] = number of lines through x that are disjoint from S (excluding x itself if x in S? No, just disjoint from S \ {x}?)
        # Condition: L is valid for x if L \ {x} is disjoint from S.
        # If x in S, we check L \ {x}. If x not in S, we check L.
        # Actually, if x not in S, L \ {x} is disjoint from S iff L is disjoint from S (since x not in S).
        # If x in S, L \ {x} is disjoint from S iff L is disjoint from S \ {x}.
        
        # Let's define a line L as "active" for x if L \ {x} has no points from S.
        
        # Efficient update:
        # When we add p to S:
        # 1. For any line L containing p, L is no longer "disjoint from S".
        # 2. This affects the validity of points x on L.
        #    For x on L:
        #    - If x == p: p is now in S. L \ {p} must be disjoint from S.
        #      Before adding p, L \ {p} was disjoint from S (if L was valid for p previously? No, p wasn't in S).
        #      Actually, if p was not in S, L was valid for p if L disjoint from S.
        #      Now p is in S. L \ {p} is disjoint from S?
        #      Yes, if L was disjoint from S before, then L \ {p} is disjoint from S \ {p} = S (since p was not in S).
        #      So adding p to S does NOT invalidate p's own validity via line L?
        #      Wait. If L was disjoint from S, then L contains no points of S.
        #      So L \ {p} contains no points of S.
        #      So p remains satisfied by L.
        #      
        #    - If x != p and x on L:
        #      L was valid for x if L disjoint from S.
        #      Now L contains p which is in S.
        #      So L is NOT disjoint from S.
        #      So L is no longer valid for x.
        #      Unless x == p (handled above).
        #      So for all x on L, x != p, the line L loses its validity.
        
        # So, adding p to S:
        # - For each line L containing p:
        #   - For each x in L, x != p:
        #     - Decrement valid_count[x] by 1?
        #     - We need to know if L was previously valid for x.
        #     - L is valid for x if L \ {x} is disjoint from S.
        #     - If x was not in S, L valid iff L disjoint from S.
        #     - If x was in S, L valid iff L \ {x} disjoint from S.
        
        # This dynamic is tricky because "disjoint from S" changes.
        
        # Alternative:
        # Maintain for each line L, the set of points in S on L.
        # A line L is "clear" for x if (L \ {x}) has no points in S.
        # This is equivalent to:
        # - If x in S: L contains exactly 1 point of S (which is x).
        # - If x not in S: L contains 0 points of S.
        
        # Let's maintain `S_count_on_line[L]` = number of points in S on line L.
        # And `in_S` array.
        
        # For a point x, a line L through x is valid if:
        # - If in_S[x]: S_count_on_line[L] == 1
        # - If not in_S[x]: S_count_on_line[L] == 0
        
        # We can maintain `valid_lines_count[x]` = number of lines L through x that satisfy the above.
        
        # Initialization:
        # in_S is all False.
        # S_count_on_line[L] = 0 for all L.
        # For each x, all 50 lines through x have count 0.
        # So valid_lines_count[x] = 50 for all x.
        
        S_count_on_line = np.zeros(2450, dtype=np.int32)
        valid_lines_count = np.full(2401, 50, dtype=np.int32)
        in_S = np.zeros(2401, dtype=bool)
        
        # Add point p to S:
        # 1. in_S[p] = True.
        # 2. For each line L containing p:
        #    S_count_on_line[L] += 1
        
        # 3. Update valid_lines_count for all points x on these lines L.
        #    For each L containing p:
        #       For each x in L:
        #          We need to check if L was valid for x before, and is it valid now?
        #          
        #          Let old_count = S_count_on_line[L] - 1 (before adding p).
        #          Let new_count = S_count_on_line[L] (after adding p).
        #          
        #          Case 1: x == p.
        #             Before: in_S[p] was False. Valid if old_count == 0.
        #             After: in_S[p] is True. Valid if new_count == 1.
        #             Note: new_count = old_count + 1.
        #             If old_count == 0, new_count == 1. So if it was valid before, it is valid after.
        #             If old_count > 0, it was not valid before. new_count > 1, so not valid after.
        #             So validity of p with respect to L does not change?
        #             Wait.
        #             If old_count == 0: L had no S points. Valid for p (not in S).
        #             After: L has 1 S point (p). Valid for p (in S).
        #             So valid_lines_count[p] stays the same?
        #             Yes, it was counted as valid, and it is still counted as valid.
        #             
        #          Case 2: x != p, x in L.
        #             Before: 
        #                If in_S[x]: Valid if old_count == 1.
        #                If not in_S[x]: Valid if old_count == 0.
        #             After:
        #                If in_S[x]: Valid if new_count == 1.
        #                If not in_S[x]: Valid if new_count == 0.
        #                
        #             Since new_count = old_count + 1:
        #             - If old_count == 0:
        #               - x not in S: Was valid (0==0). Now new_count=1, not valid (1!=0). -> Decrement valid_lines_count[x].
        #               - x in S: Was not valid (0!=1). Now new_count=1, valid (1==1). -> Increment valid_lines_count[x].
        #             - If old_count == 1:
        #               - x not in S: Was not valid (1!=0). Now new_count=2, not valid (2!=0). -> No change.
        #               - x in S: Was valid (1==1). Now new_count=2, not valid (2!=1). -> Decrement valid_lines_count[x].
        #             - If old_count > 1:
        #               - x not in S: Was not valid. Now not valid. -> No change.
        #               - x in S: Was not valid. Now not valid. -> No change.
        
        # So, to check if we can add p:
        # We need to ensure that for all x, valid_lines_count[x] > 0 AFTER the update.
        # Actually, we just need to check if any x becomes 0.
        
        # To do this efficiently:
        # We simulate the changes.
        # If any x would drop to 0, we cannot add p.
        
        # Implementation:
        # 1. Identify all lines L containing p.
        # 2. For each such L:
        #    old_count = S_count_on_line[L]
        #    new_count = old_count + 1
        #    For each x in L:
        #       Determine if valid_lines_count[x] changes.
        #       If it decreases, check if it becomes 0.
        
        # This is O(50 * 49) = 2450 operations per candidate p.
        # With 2401 candidates, total ~ 6M operations.
        # This is very fast in Python if vectorized or optimized.
        # But the inner loop is over points in L.
        
        # Let's implement this.
        
        # Precompute list of lines for each point
        # point_lines is a list of lists of line indices.
        
        for p_idx in order:
            # Check if adding p_idx is possible
            # We need to see if valid_lines_count[x] > 0 for all x after update.
            
            # Collect affected points
            # affected_points = set()
            # For each L in point_lines[p_idx]:
            #    for x in line_points[L]:
            #        affected_points.add(x)
            
            # Instead of set, we can just iterate and track min count?
            # But we need to revert if it fails.
            
            # Let's calculate the delta for each x.
            # delta[x] = new_valid - old_valid
            
            # We can store the deltas in a dictionary or list.
            # Since number of affected points is at most 50*49 = 2450, but they overlap.
            # Max unique points is 2401.
            
            # Let's just iterate and check.
            # To avoid double counting or complexity, let's just do the update and check min.
            # If min > 0, commit. Else revert.
            
            # To revert, we need to store the old values of valid_lines_count and S_count_on_line.
            # Or just re-apply the inverse.
            
            # Inverse of adding p:
            # For each L in point_lines[p_idx]:
            #    S_count_on_line[L] -= 1
            #    For each x in L:
            #       Reverse the validity change.
            
            # Let's try a "tentative" approach.
            
            # 1. Save state of S_count_on_line for lines in point_lines[p_idx]
            old_counts = []
            for L in point_lines[p_idx]:
                old_counts.append(S_count_on_line[L])
            
            # 2. Apply update
            # We need to update valid_lines_count for all x in union of lines.
            # Let's use a dict to track changes to valid_lines_count to revert easily?
            # Or just iterate all affected x and update valid_lines_count, then check min.
            
            # To check min efficiently, we can't just take np.min(valid_lines_count) if we only update some.
            # We can keep a global min? No, updates are local.
            
            # Since 2401 is small, np.min(valid_lines_count) is fast.
            
            # Let's apply updates to valid_lines_count.
            # We need to know old_count for each L to determine the change for x.
            
            # Map L -> old_count
            # We have old_counts list corresponding to point_lines[p_idx].
            
            # For each L in point_lines[p_idx]:
            #    old_c = old_counts[i]
            #    new_c = old_c + 1
            #    S_count_on_line[L] = new_c
            #    for x in line_points[L]:
            #        # Determine change
            #        if old_c == 0:
            #            if in_S[x]:
            #                # 0 -> 1. x in S. Was not valid (0!=1). Now valid (1==1). +1
            #                valid_lines_count[x] += 1
            #            else:
            #                # 0 -> 1. x not in S. Was valid (0==0). Now not valid (1!=0). -1
            #                valid_lines_count[x] -= 1
            #        elif old_c == 1:
            #            if in_S[x]:
            #                # 1 -> 2. x in S. Was valid (1==1). Now not valid (2!=1). -1
            #                valid_lines_count[x] -= 1
            #            else:
            #                # 1 -> 2. x not in S. Was not valid. Now not valid. 0
            #                pass
            #        else:
            #            # No change in validity status for any x?
            #            # old_c >= 2. new_c >= 3.
            #            # x in S: valid if count==1. No.
            #            # x not in S: valid if count==0. No.
            #            pass
            
            # Wait, what if x == p?
            # in_S[p] is currently False.
            # We are setting in_S[p] = True as part of the commit.
            # But in the check phase, in_S[p] is still False.
            # So for x=p, we treat it as "not in S".
            # If old_c == 0:
            #    x=p (not in S). Was valid. Now not valid (count 1). -1.
            #    But after commit, in_S[p] becomes True.
            #    Then for count 1, x=p (in S) is valid.
            #    So the net change for p is 0?
            #    Let's trace:
            #    Before: in_S[p]=F, count=0. Valid? Yes.
            #    After: in_S[p]=T, count=1. Valid? Yes.
            #    So valid_lines_count[p] should not change.
            #    But my logic above for "not in S" with old_c=0 says -1.
            #    And I haven't applied the "in_S change" yet.
            
            # This is getting complicated.
            # Let's separate the "S count" update and "in_S" update.
            
            # Revised Plan:
            # 1. Tentatively set in_S[p] = True.
            # 2. Update S_count_on_line for lines through p.
            # 3. Update valid_lines_count for all affected x based on NEW in_S and NEW S_count.
            #    To do this, we need to know OLD valid status and NEW valid status.
            
            # Let's just recompute valid_lines_count for affected points?
            # No, that's expensive.
            
            # Let's stick to the delta logic but handle p correctly.
            
            # For x != p: in_S[x] doesn't change.
            # For x == p: in_S[p] changes F -> T.
            
            # Let's iterate.
            # We will store the old valid_lines_count for affected points to revert.
            
            affected_points = set()
            for L in point_lines[p_idx]:
                for x in line_points[L]:
                    affected_points.add(x)
                    
            # This set construction is O(2450).
            # Then for each x in affected_points:
            #    Calculate old_valid_count and new_valid_count?
            #    No, we can calculate the delta for each line L and sum them?
            #    No, validity is per line.
            
            # Let's just update valid_lines_count in place and keep a list of (x, delta) to revert.
            
            changes = []
            
            for i, L in enumerate(point_lines[p_idx]):
                old_c = old_counts[i]
                new_c = old_c + 1
                S_count_on_line[L] = new_c
                
                for x in line_points[L]:
                    # Determine old validity of L for x
                    # old_in_S_x = in_S[x] (before update, so p is False)
                    # new_in_S_x = in_S[x] (after update, so p is True)
                    
                    old_in_s = in_S[x]
                    new_in_s = in_S[x]
                    if x == p_idx:
                        new_in_s = True
                        # old_in_s is False (since we haven't updated in_S array yet)
                    
                    # Old validity:
                    if old_in_s:
                        old_valid = (old_c == 1)
                    else:
                        old_valid = (old_c == 0)
                        
                    # New validity:
                    if new_in_s:
                        new_valid = (new_c == 1)
                    else:
                        new_valid = (new_c == 0)
                        
                    if old_valid and not new_valid:
                        valid_lines_count[x] -= 1
                        changes.append((x, -1))
                    elif not old_valid and new_valid:
                        valid_lines_count[x] += 1
                        changes.append((x, 1))
                        
            # Check if any point has valid_lines_count == 0
            # We only need to check affected points?
            # No, if a point was 1 and we decremented it, it becomes 0.
            # If a point was not affected, it remains > 0 (assuming we started valid).
            # So we only need to check affected points.
            
            valid = True
            for x in affected_points:
                if valid_lines_count[x] == 0:
                    valid = False
                    break
            
            if valid:
                in_S[p_idx] = True
                # Commit: changes are already applied to valid_lines_count
                # S_count_on_line is updated
                # in_S is updated
            else:
                # Revert
                for x, delta in changes:
                    valid_lines_count[x] -= delta
                for i, L in enumerate(point_lines[p_idx]):
                    S_count_on_line[L] = old_counts[i]
                # in_S[p_idx] remains False
        
        return in_S

    # --- Main Execution ---
    start_time = time.time()
    deadline = start_time + 580
    
    best_in_S = None
    best_size = 0
    seed_id = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        in_S = greedy_build(seed_id)
        size = np.sum(in_S)
        if size > best_size:
            best_size = size
            best_in_S = in_S
            # Print progress? No, stdout might be captured or slow.
            # Just update.
        seed_id += 1
        
        # If we found a very large set, maybe stop?
        # Max possible S is roughly 2401 * (1/50) * something?
        # Actually, if S is large, it's hard to satisfy conditions.
        # Let's just keep going.
        
    if best_in_S is None:
        # Fallback
        best_in_S = np.zeros(2401, dtype=bool)
        best_in_S[0] = True
        
    # Construct result N
    # N is points NOT in S.
    result_list = []
    for i in range(2401):
        if not best_in_S[i]:
            x = i // 49
            y = i % 49
            xa, xb = divmod(x, 7)
            ya, yb = divmod(y, 7)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
