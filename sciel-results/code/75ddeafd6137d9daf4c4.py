import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by finding a large complement S
    using local search, where S satisfies:
    - Every x in S has an allowed line through x containing no other S-point
    - Every x in N has an allowed line through x containing no S-point
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Precompute field elements: (a, b) represents a + b*alpha where alpha^2 = 3
    # Field operations
    def add(f1, f2):
        return ((f1[0] + f2[0]) % p, (f1[1] + f2[1]) % p)
    
    def mul(f1, f2):
        a1, b1 = f1
        a2, b2 = f2
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
        # = (a1*a2 + 3*b1*b2) + (a1*b2 + a2*b1)*alpha
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def neg(f1):
        return ((-f1[0]) % p, (-f1[1]) % p)
    
    def inv(f1):
        # Inverse of a + b*alpha: (a - b*alpha) / (a^2 - 3*b^2)
        a, b = f1
        det = (a*a - 3*b*b) % p
        if det == 0:
            return (0, 0)  # shouldn't happen for non-zero
        # Find inverse of det in F_7
        for d in range(1, p):
            if (det * d) % p == 1:
                det_inv = d
                break
        return ((a * det_inv) % p, (-b * det_inv) % p)
    
    def sub(f1, f2):
        return add(f1, neg(f2))
    
    # Represent points as integers 0..2400
    # Point index: ((xa * p + xb) * p + ya) * p + yb
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = idx // (p*p*p)
        return (xa, xb, ya, yb)
    
    # All field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Precompute all allowed lines
    # Direction (1, m) for m in F_49, and direction (0, 1)
    # A line with direction (1, m) through point (x0, y0) contains points (x0 + t, y0 + t*m) for t in F_49
    # A line with direction (0, 1) through point (x0, y0) contains points (x0, y0 + t) for t in F_49
    
    # For each point, precompute which lines (identified by direction + intercept) pass through it
    # Line ID: direction_idx * q + intercept_idx
    # For direction (1, m): intercept is y - x*m (in field), so for point (x, y), intercept = y - x*m
    # For direction (0, 1): intercept is x, so for point (x, y), intercept = x
    
    # Let's precompute for each point, the list of line IDs that contain it
    # And for each line, the list of points on it
    
    # 50 directions: 49 slopes m + 1 vertical
    # Total lines: 50 * 49 = 2450
    
    # Build line -> points mapping
    line_points = []  # list of lists, each list is set of point indices on that line
    point_lines = [[] for _ in range(total_points)]  # for each point, list of line IDs containing it
    
    for dir_idx in range(50):
        if dir_idx < 49:
            # Direction (1, m) where m = field_elements[dir_idx]
            m = field_elements[dir_idx]
            for intercept in range(q):
                line_id = dir_idx * q + intercept
                points_on_line = set()
                for x_idx in range(q):
                    x = field_elements[x_idx]
                    # y = x * m + intercept_val where intercept_val = field_elements[intercept]
                    intercept_val = field_elements[intercept]
                    y = add(mul(x, m), intercept_val)
                    # Find point index
                    xa, xb = x
                    ya, yb = y
                    pidx = point_to_idx(xa, xb, ya, yb)
                    points_on_line.add(pidx)
                line_points.append(list(points_on_line))
                for pidx in points_on_line:
                    point_lines[pidx].append(line_id)
        else:
            # Direction (0, 1): vertical lines, x = constant
            for x_idx in range(q):
                line_id = dir_idx * q + x_idx
                points_on_line = set()
                x = field_elements[x_idx]
                xa, xb = x
                for y_idx in range(q):
                    y = field_elements[y_idx]
                    ya, yb = y
                    pidx = point_to_idx(xa, xb, ya, yb)
                    points_on_line.add(pidx)
                line_points.append(list(points_on_line))
                for pidx in points_on_line:
                    point_lines[pidx].append(line_id)
    
    # Now we have:
    # line_points[line_id] = list of point indices on that line
    # point_lines[pidx] = list of line IDs containing that point
    
    # Nikodym condition check:
    # S is valid if:
    # 1. For every x in S: exists line L in point_lines[x] such that L ∩ S = {x}
    # 2. For every x in N (= complement of S): exists line L in point_lines[x] such that L ∩ S = ∅
    
    # Start with S = empty set (N = all points) - trivially valid
    # Try to grow S via local search
    
    deadline = time.time() + 580  # leave 20s safety margin
    
    # Initial S: try a structured initial guess
    # Let's start with S being a random subset and see if we can improve
    
    best_S = set()
    best_score = 0
    
    def is_valid(S):
        """Check if S is a valid complement (i.e., N = complement is Nikodym)"""
        S_set = set(S)
        # Check condition 1: every x in S has a line through it with no other S-point
        for x in S_set:
            found = False
            for lid in point_lines[x]:
                # Check if any other point on this line is in S
                has_other = False
                for y in line_points[lid]:
                    if y != x and y in S_set:
                        has_other = True
                        break
                if not has_other:
                    found = True
                    break
            if not found:
                return False
        # Check condition 2: every x not in S has a line through it with no S-point
        for x in range(total_points):
            if x in S_set:
                continue
            found = False
            for lid in point_lines[x]:
                # Check if any point on this line is in S
                has_S = False
                for y in line_points[lid]:
                    if y in S_set:
                        has_S = True
                        break
                if not has_S:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Faster validity check using precomputed structures
    def is_valid_fast(S_set):
        """Faster check using line-level counting"""
        # For each line, count how many S-points it contains
        line_S_count = [0] * len(line_points)
        for x in S_set:
            for lid in point_lines[x]:
                line_S_count[lid] += 1
        
        # Condition 1: for each x in S, exists line through x with count == 1 (only x)
        for x in S_set:
            found = False
            for lid in point_lines[x]:
                if line_S_count[lid] == 1:
                    found = True
                    break
            if not found:
                return False
        
        # Condition 2: for each x not in S, exists line through x with count == 0
        for x in range(total_points):
            if x in S_set:
                continue
            found = False
            for lid in point_lines[x]:
                if line_S_count[lid] == 0:
                    found = True
                    break
            if not found:
                return False
        
        return True
    
    # Local search: start with empty S, try to add points
    S = set()
    
    # Precompute for efficiency: for each point, its lines
    # Try adding points one by one
    
    # First, let's try a greedy approach: repeatedly add the point that's easiest to add
    # A point x can be added to S if after adding, validity is maintained
    
    # For speed, let's track line_S_count incrementally
    line_S_count = [0] * len(line_points)
    
    def try_add_point(x, S_set, line_S_count_arr):
        """Try to add x to S_set. Return True if valid after adding."""
        # Check condition for x: exists line through x with count 0 (before adding)
        x_ok = False
        for lid in point_lines[x]:
            if line_S_count_arr[lid] == 0:
                x_ok = True
                break
        if not x_ok:
            return False, None
        
        # Temporarily add x
        S_set.add(x)
        old_counts = {}
        for lid in point_lines[x]:
            old_counts[lid] = line_S_count_arr[lid]
            line_S_count_arr[lid] += 1
        
        # Check if all existing S-points still have a witness line
        # Only points that share a line with x might be affected
        affected = set()
        for lid in point_lines[x]:
            for y in line_points[lid]:
                if y in S_set and y != x:
                    affected.add(y)
        
        for y in affected:
            found = False
            for lid2 in point_lines[y]:
                if line_S_count_arr[lid2] == 1:
                    found = True
                    break
            if not found:
                # Revert
                S_set.remove(x)
                for lid in point_lines[x]:
                    line_S_count_arr[lid] = old_counts[lid]
                return False, None
        
        # Check condition 2 for points NOT in S that are affected
        # Points not in S that share a line with x might lose their witness
        # But adding x to S can only HURT condition 2 (more S-points on lines)
        # So we need to check if any non-S point that previously had a zero-count line
        # now has all lines with count > 0
        
        # This is expensive. Let's just do a full check for now during development
        # For the actual run, we'll be more careful
        
        # Actually, condition 2 for non-S points: they need a line with count 0.
        # Adding x increases counts on lines through x. A non-S point y that is on a line
        # through x might lose its zero-count line.
        
        # Check all non-S points that share a line with x
        affected_nonS = set()
        for lid in point_lines[x]:
            for y in line_points[lid]:
                if y not in S_set:
                    affected_nonS.add(y)
        
        for y in affected_nonS:
            found = False
            for lid2 in point_lines[y]:
                if line_S_count_arr[lid2] == 0:
                    found = True
                    break
            if not found:
                # Revert
                S_set.remove(x)
                for lid in point_lines[x]:
                    line_S_count_arr[lid] = old_counts[lid]
                return False, None
        
        return True, old_counts
    
    # Greedy: repeatedly try to add points
    order = list(range(total_points))
    random.shuffle(order)
    
    for x in order:
        if time.time() > deadline:
            break
        success, _ = try_add_point(x, S, line_S_count)
        if success:
            pass  # x is now in S
    
    if len(S) > best_score:
        best_S = set(S)
        best_score = len(S)
    
    # Now do local search: try to swap or add more
    # Try adding points that weren't added
    for x in order:
        if time.time() > deadline:
            break
        if x in S:
            continue
        success, _ = try_add_point(x, S, line_S_count)
        if success:
            pass
    
    if len(S) > best_score:
        best_S = set(S)
        best_score = len(S)
    
    # Try random restarts with different orders
    for restart in range(20):
        if time.time() > deadline:
            break
        S = set()
        line_S_count = [0] * len(line_points)
        order = list(range(total_points))
        random.shuffle(order)
        
        for x in order:
            if time.time() > deadline:
                break
            success, _ = try_add_point(x, S, line_S_count)
            if success:
                pass
        
        if len(S) > best_score:
            best_S = set(S)
            best_score = len(S)
    
    # Convert best_S to N (complement)
    N = [i for i in range(total_points) if i not in best_S]
    
    # Build the array
    result = np.zeros((len(N), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N):
        xa, xb, ya, yb = idx_to_point(idx)
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
    
    return result
