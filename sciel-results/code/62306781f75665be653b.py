# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using iterative core extraction.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    q = p * p  # 49
    
    # Field F_49 arithmetic
    # alpha^2 = 3
    # (a,b) represents a + b*alpha
    # (a,b) + (c,d) = (a+c, b+d) mod 7
    # (a,b) * (c,d) = (ac+3bd, ad+bc) mod 7
    
    def f49_add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def f49_mul(x, y):
        return ((x[0]*y[0] + 3*x[1]*y[1]) % p, (x[0]*y[1] + x[1]*y[0]) % p)
    
    def f49_sub(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
        
    # Map points to indices 0..2400
    # Point (xa, xb, ya, yb) -> index
    # index = ((xa * 7 + xb) * 7 + ya) * 7 + yb
    def get_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb
        
    def get_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return (xa, xb, ya, yb)

    # Precompute all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))
                    
    # Precompute lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    # For each direction, and each line in that direction, store the list of point indices on it.
    
    directions = []
    # Non-vertical: dx=1, dy=m
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    # Vertical: dx=0, dy=1
    directions.append(((0, 1), (0, 0)))
    
    # For each direction, generate lines.
    # A line is determined by a point on it.
    # To avoid duplicates, we can just generate all lines and store them.
    # Total lines: 50 directions * 49 lines/direction = 2450 lines.
    # Each line has 49 points.
    
    # lines[i] is a set of indices
    lines = []
    # point_to_lines[i] is a list of line indices containing point i
    point_to_lines = [[] for _ in range(q*q)]
    
    for dx, dy in directions:
        # Generate 49 lines for this direction
        # A line is { x + t*v : t in F_49 }
        # To get unique lines, we can fix the "first" point in some canonical way.
        # For non-vertical (1, m), fix x=0? No, x is a pair.
        # Let's just iterate over all points as potential "starts" and use a set to deduplicate lines?
        # That's inefficient.
        
        # Better:
        # For direction v, the lines are parallel.
        # We can parameterize lines by their intersection with a specific "axis".
        # For non-vertical (1, m), the line intersects the "y-axis" (x=0) at a unique point (0, y0).
        # So y0 determines the line. There are q choices for y0.
        # For vertical (0, 1), the line is x = x0. There are q choices for x0.
        
        if dx[0] == 1 and dx[1] == 0:
            # Non-vertical
            for ya0 in range(p):
                for yb0 in range(p):
                    # Start point (0, 0, ya0, yb0)
                    x_start = (0, 0)
                    y_start = (ya0, yb0)
                    line_pts = []
                    for ta in range(p):
                        for tb in range(p):
                            t = (ta, tb)
                            tx = f49_add(x_start, t)
                            ty = f49_add(y_start, f49_mul(t, dy))
                            idx = get_idx(tx[0], tx[1], ty[0], ty[1])
                            line_pts.append(idx)
                    
                    # Store line
                    line_set = set(line_pts)
                    line_id = len(lines)
                    lines.append(line_set)
                    for idx in line_set:
                        point_to_lines[idx].append(line_id)
                        
        else:
            # Vertical
            for xa0 in range(p):
                for xb0 in range(p):
                    # Start point (xa0, xb0, 0, 0)
                    x_start = (xa0, xb0)
                    y_start = (0, 0)
                    line_pts = []
                    for ta in range(p):
                        for tb in range(p):
                            t = (ta, tb)
                            tx = x_start
                            ty = f49_add(y_start, t)
                            idx = get_idx(tx[0], tx[1], ty[0], ty[1])
                            line_pts.append(idx)
                    
                    line_set = set(line_pts)
                    line_id = len(lines)
                    lines.append(line_set)
                    for idx in line_set:
                        point_to_lines[idx].append(line_id)

    # Initialize N with a known valid set.
    # Use the seed's construction: N = F^2 \ (U \ B)
    # U: 2*xa + ya^2 - yb^2*3 == 0 (mod 7) ? 
    # The seed used k=3 (smallest non-residue).
    # in_unital: 2*xa + ya*ya - yb*yb*k % p == 0
    
    k = 3
    def in_unital(xa, xb, ya, yb):
        return (2 * xa + ya * ya - yb * yb * k) % p == 0
        
    U_set = set()
    for idx in range(q*q):
        xa, xb, ya, yb = get_point(idx)
        if in_unital(xa, xb, ya, yb):
            U_set.add(idx)
            
    # B: points in U with ya=0
    B_set = set()
    for idx in U_set:
        xa, xb, ya, yb = get_point(idx)
        if ya == 0:
            B_set.add(idx)
            
    # N = F^2 \ (U \ B)
    # Points NOT in N are U \ B
    # So N contains all points NOT in U, and points in U that are in B.
    N_mask = np.ones(q*q, dtype=bool)
    for idx in U_set:
        if idx not in B_set:
            N_mask[idx] = False
            
    # Greedy Pruning Loop
    # We want to remove points from N to maximize complement.
    # A point p can be removed if no point x becomes uncovered.
    
    # For each x, count how many lines L through x are "good" (L \ {x} subset N).
    # good_count[x]
    
    def compute_good_counts(N_mask):
        good_counts = np.zeros(q*q, dtype=int)
        # For each x, check all lines through x
        for x in range(q*q):
            count = 0
            for line_id in point_to_lines[x]:
                line_pts = lines[line_id]
                # Check if all other points in line_pts are in N
                # line_pts contains x.
                # We can precompute line sums?
                # Sum of N_mask over line_pts. If sum == len(line_pts) - 1 (since x is in N, and we check others), 
                # wait, x is in N_mask? Yes, we are checking coverage for x assuming x is in N?
                # No, x can be anywhere. But Nikodym condition:
                # "every point x ... lies on a line ... whose points other than x all belong to N"
                # So x itself does NOT need to be in N.
                # But we are constructing N. We only remove points from N.
                # So x is either in N or not.
                # If x is not in N, we still need it to be covered by a line where others are in N.
                
                # Let's check: sum of N_mask over (line_pts \ {x})
                # If sum == len(line_pts) - 1, then the line is good.
                
                # Optimization: Precompute line_sum[line_id] = sum(N_mask[i] for i in lines[line_id])
                # Then sum(line_pts \ {x}) = line_sum[line_id] - N_mask[x]
                # We need this to be len(line_pts) - 1.
                
                pass # We'll do this in the main loop for efficiency
                
        return good_counts

    # To speed up, we maintain line_sums.
    line_sums = np.array([sum(N_mask[i] for i in line) for line in lines], dtype=int)
    
    # good_count[x] = number of lines L through x such that sum(L \ {x}) == len(L) - 1
    # len(L) is always q=49.
    # Condition: line_sums[L] - N_mask[x] == 48
    
    def update_good_counts(x, N_mask, line_sums):
        count = 0
        for line_id in point_to_lines[x]:
            # Check if line is good for x
            # sum(L \ {x}) = line_sums[line_id] - N_mask[x]
            if line_sums[line_id] - N_mask[x] == 48:
                count += 1
        return count

    # Initial good counts
    good_counts = np.array([update_good_counts(x, N_mask, line_sums) for x in range(q*q)], dtype=int)
    
    # A point p in N is "safe" to remove if for all x, removing p does not reduce good_counts[x] to 0.
    # Removing p affects line_sums[L] for all L containing p.
    # If p is in L, line_sums[L] decreases by 1.
    # This might make a line L no longer good for some x in L.
    # Specifically, if L was good for x, and p in L \ {x}, then after removal, sum(L \ {x}) decreases by 1, so it becomes 47 != 48. L is no longer good.
    # If p == x, then N_mask[x] changes? No, we are removing p from N.
    # If p == x, then x is no longer in N.
    # Does x need to be in N? No.
    # But the condition for x being covered depends on N.
    # If x is removed from N, N_mask[x] becomes 0.
    # For x itself: sum(L \ {x}) = line_sums[L] - N_mask[x].
    # If N_mask[x] was 1, now 0.
    # line_sums[L] also decreases by 1 (since x is in L).
    # So sum(L \ {x}) = (line_sums[L] - 1) - 0 = line_sums[L] - 1.
    # Previously sum(L \ {x}) = line_sums[L] - 1.
    # So the value for x itself doesn't change?
    # Wait.
    # Before: N_mask[x]=1. line_sums[L] includes x.
    # Sum(L \ {x}) = line_sums[L] - 1.
    # After: N_mask[x]=0. line_sums[L] decreases by 1.
    # Sum(L \ {x}) = (line_sums[L] - 1) - 0 = line_sums[L] - 1.
    # So the goodness of lines for x itself is unchanged.
    
    # However, for other points y in L (y != x):
    # Before: Sum(L \ {y}) = line_sums[L] - N_mask[y].
    # After: line_sums[L] decreases by 1. N_mask[y] unchanged.
    # Sum(L \ {y}) decreases by 1.
    # So if L was good for y (Sum=48), it becomes 47 (not good).
    
    # So, removing p from N:
    # 1. Decrements line_sums[L] for all L containing p.
    # 2. For each such L, and each y in L (y != p), if L was good for y, it is no longer good.
    #    So good_counts[y] decreases by 1 for each such L.
    # 3. If any good_counts[y] becomes 0, y is uncovered.
    
    # So p is safe if for all y != p such that p is on a line L good for y, good_counts[y] > 1.
    # i.e., p is not on the *unique* good line for any y.
    
    # Let's implement the removal.
    
    removed = True
    while removed and time.time() < deadline - 10:
        removed = False
        # Identify points in N that are safe
        safe_points = []
        
        # For each p in N:
        # Check if p is on any "critical" line.
        # A line L is critical for y if good_counts[y] == 1 and L is the unique good line for y.
        
        # Precompute critical lines for each y?
        # critical_line_for_y[y] = line_id if good_counts[y]==1 else -1
        
        critical_lines = np.full(q*q, -1, dtype=int)
        for y in range(q*q):
            if good_counts[y] == 1:
                # Find the unique good line
                for line_id in point_to_lines[y]:
                    if line_sums[line_id] - N_mask[y] == 48:
                        critical_lines[y] = line_id
                        break
        
        # For each p in N, check if it is on any critical_lines[y]
        # If yes, p is unsafe.
        
        unsafe_points = np.zeros(q*q, dtype=bool)
        for y in range(q*q):
            if critical_lines[y] != -1:
                L = lines[critical_lines[y]]
                for p in L:
                    if p != y and N_mask[p]:
                        unsafe_points[p] = True
                        # We can break if we just want to mark unsafe, but we need to mark all unsafe.
                        # Actually, if p is on ANY critical line, it is unsafe.
        
        # Safe points are in N and not unsafe
        for p in range(q*q):
            if N_mask[p] and not unsafe_points[p]:
                safe_points.append(p)
                
        if not safe_points:
            break
            
        # Remove all safe points?
        # Removing multiple points at once is tricky because removing one might make another unsafe.
        # But if p is safe, it means it's not on any *current* critical line.
        # If we remove a set S of safe points, do they become unsafe?
        # Removing p reduces line_sums. This might make new lines non-good, increasing criticality.
        # So removing a batch is risky.
        # Let's remove one by one? Or a small batch?
        # Given time constraints, let's remove one by one, or try to remove all and re-verify?
        # Re-verifying is O(N * 50).
        
        # Let's try removing all safe points at once and then re-computing good_counts.
        # If it fails, we can revert. But reverting is easy (just set N_mask back).
        
        old_N_mask = N_mask.copy()
        for p in safe_points:
            N_mask[p] = False
            # Update line_sums
            for line_id in point_to_lines[p]:
                line_sums[line_id] -= 1
                
        # Recompute good_counts
        new_good_counts = np.array([update_good_counts(x, N_mask, line_sums) for x in range(q*q)], dtype=int)
        
        # Check if any point has good_count 0
        if np.any(new_good_counts == 0):
            # Revert
            N_mask[:] = old_N_mask
            # Revert line_sums
            for p in safe_points:
                for line_id in point_to_lines[p]:
                    line_sums[line_id] += 1
            # Try removing only one safe point?
            # Pick the first safe point
            p = safe_points[0]
            N_mask[p] = False
            for line_id in point_to_lines[p]:
                line_sums[line_id] -= 1
            new_good_counts = np.array([update_good_counts(x, N_mask, line_sums) for x in range(q*q)], dtype=int)
            if np.any(new_good_counts == 0):
                # Revert even this
                N_mask[p] = True
                for line_id in point_to_lines[p]:
                    line_sums[line_id] += 1
                # If even one fails, maybe we are stuck?
                # Try next safe point?
                # For simplicity, if batch fails and single fails, break.
                break
            else:
                good_counts = new_good_counts
                removed = True
        else:
            good_counts = new_good_counts
            removed = True

    # Convert N_mask to list of points
    points = []
    for idx in range(q*q):
        if N_mask[idx]:
            xa, xb, ya, yb = get_point(idx)
            points.append((xa, xb, ya, yb))
            
    return np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in points], dtype=np.int64)

# EVOLVE-BLOCK-END
