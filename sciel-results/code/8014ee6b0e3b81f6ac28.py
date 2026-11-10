# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy: Maximize the size of the complement S (which is F^2 \ N).
    We hypothesize that the Hermitian Unital U is a valid S.
    If not, we minimally remove points from S to ensure validity.
    Metric: Maximize |S|.
    """
    start_time = time.time()
    deadline = start_time + 550.0

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Field operations for F_49
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def trace(a):
        return (2 * a[0]) % p

    def norm(a):
        return (a[0] * a[0] - a[1] * a[1] * k) % p

    # Generate all points and their indices
    # idx = xa + p*xb + p^2*ya + p^3*yb
    # Actually, let's use a flat index 0..2400
    # x = xa + xb*alpha, y = ya + yb*alpha
    # idx = xa + 7*xb + 49*ya + 343*yb
    
    def get_coords(idx):
        xa = idx % p
        xb = (idx // p) % p
        ya = (idx // (p*p)) % p
        yb = (idx // (p*p*p)) % p
        return ((xa, xb), (ya, yb))

    def get_idx(x, y):
        xa, xb = x
        ya, yb = y
        return xa + p*xb + p*p*ya + p*p*p*yb

    total_points = q * q
    all_indices = set(range(total_points))

    # Identify Unital U
    # Using trace(x) + norm(y) == 0 as a proxy for Hermitian structure
    # Or better: standard Hermitian unital x^q x + ... 
    # For F_{49}, x^7 = conjugate.
    # Let's use the code's previous definition to ensure consistency if it was valid,
    # but optimize the logic to MAXIMIZE S.
    
    unital_indices = set()
    for i in range(total_points):
        x, y = get_coords(i)
        if (trace(x) + norm(y)) % p == 0:
            unital_indices.add(i)
            
    # If the set is empty or too small, use a different definition or default
    if len(unital_indices) < 100:
        # Fallback: just use a subset or the whole space if needed?
        # But let's assume the definition is okay.
        pass

    # S starts as the Unital
    S = set(unital_indices)
    N = all_indices - S

    # Precompute lines for validity checks
    # Directions: (1, m) for m in F_49, and (0, 1).
    # There are 49 + 1 = 50 directions.
    # For each direction, lines are parallel.
    
    # To check validity of S:
    # 1. For each x in S: exists line L through x s.t. |L cap S| <= 1.
    # 2. For each y in N: exists line L through y s.t. L cap S is empty.
    
    # We can check these conditions. If S=U fails condition 2, we must remove points from S.
    # Removing points from S (adding to N) helps condition 2 for others, 
    # but the removed point x (now in N) must satisfy condition 2 for itself.
    
    # Optimization: 
    # Represent S as a bitmask or boolean array.
    # For each line, we can count intersections with S.
    
    lines_by_point = {i: [] for i in range(total_points)}
    all_lines = [] # List of sets of indices
    
    # Generate lines
    # Direction (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            # Line equation: y = m*x + c
            # We need to iterate over parallel lines (constants c)
            # In F_49, lines with direction (1, m) are determined by their intersection with x=0 line?
            # No, just iterate over all points to find unique lines? Expensive.
            # Better: iterate over "base" points for each direction.
            
            # For direction (1, m), the lines are disjoint.
            # A line is { (x, m*x+c) | x in F_49 }.
            # We can parametrize by c.
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_set = set()
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = get_idx(x, y)
                            line_set.add(idx)
                    line_tuple = frozenset(line_set)
                    if line_tuple not in all_lines: # Rough check, might use a set of tuples
                         all_lines.append(line_tuple)
                         for idx in line_set:
                             lines_by_point[idx].append(len(all_lines) - 1)
                             
    # Direction (0, 1) - Vertical lines
    for ca in range(p):
        for cb in range(p):
            x = (ca, cb)
            line_set = set()
            for ya in range(p):
                for yb in range(p):
                    y = (ya, yb)
                    idx = get_idx(x, y)
                    line_set.add(idx)
            line_tuple = frozenset(line_set)
            if line_tuple not in all_lines:
                all_lines.append(line_tuple)
                for idx in line_set:
                    lines_by_point[idx].append(len(all_lines) - 1)

    # To make 'in' fast, convert all_lines to sets (they are frozensets already)
    # But 'line_tuple not in all_lines' is O(N^2) if not careful.
    # Let's use a set of tuples for uniqueness.
    unique_lines_set = set()
    clean_lines = []
    # Regenerate cleanly? The above generation might be slow or duplicate-heavy.
    # Given 2401 points, 50 directions, ~49 lines per direction -> 2450 lines.
    # This is small. We can just build a list of lines and a point->line map.
    
    point_to_lines = {i: [] for i in range(total_points)}
    valid_lines = []
    seen = set()
    
    # Re-generate lines more efficiently
    # For each direction, generate one representative line per "orbit" of the parallel class.
    # Or just generate all lines and dedupe.
    
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            line_pts.append(get_idx(x, y))
                    ls = frozenset(line_pts)
                    if ls not in seen:
                        seen.add(ls)
                        valid_lines.append(ls)
                        for pt in line_pts:
                            point_to_lines[pt].append(len(valid_lines)-1)
                            
    for ca in range(p):
        for cb in range(p):
            x = (ca, cb)
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    y = (ya, yb)
                    line_pts.append(get_idx(x, y))
            ls = frozenset(line_pts)
            if ls not in seen:
                seen.add(ls)
                valid_lines.append(ls)
                for pt in line_pts:
                    point_to_lines[pt].append(len(valid_lines)-1)

    # Check if S is valid
    def is_valid(S_set):
        # 1. For all x in S: exists line L through x with len(L & S) <= 1
        for x in S_set:
            ok = False
            for lid in point_to_lines[x]:
                L = valid_lines[lid]
                # count intersection
                # Since S is small-ish, we can just iterate L
                inter = 0
                for pt in L:
                    if pt in S_set:
                        inter += 1
                        if inter > 1:
                            break
                if inter <= 1:
                    ok = True
                    break
            if not ok:
                return False
        
        # 2. For all y in N (complement): exists line L through y with len(L & S) == 0
        for y in all_indices:
            if y in S_set:
                continue
            ok = False
            for lid in point_to_lines[y]:
                L = valid_lines[lid]
                inter = 0
                for pt in L:
                    if pt in S_set:
                        inter += 1
                        if inter > 0:
                            break
                if inter == 0:
                    ok = True
                    break
            if not ok:
                return False
        return True

    # Initial S
    current_S = set(unital_indices)
    
    if not is_valid(current_S):
        # Need to remove points from S to fix validity.
        # Greedy removal: Find a "bad" point y in N that has no disjoint line.
        # Remove a point s in S that blocks y.
        
        # We might need to iterate.
        max_iters = 500
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find bad y in N
            bad_y = None
            blocking_points = set()
            for y in (all_indices - current_S):
                has_disjoint = False
                for lid in point_to_lines[y]:
                    L = valid_lines[lid]
                    is_disjoint = True
                    for pt in L:
                        if pt in current_S:
                            is_disjoint = False
                            break
                    if is_disjoint:
                        has_disjoint = True
                        break
                if not has_disjoint:
                    bad_y = y
                    # Identify all points in S that block y
                    # Actually, if NO line is disjoint, then EVERY line through y intersects S.
                    # We need to pick one point to remove.
                    # Let's pick the first point in S that lies on ANY line through y.
                    for lid in point_to_lines[y]:
                        L = valid_lines[lid]
                        for pt in L:
                            if pt in current_S:
                                blocking_points.add(pt)
                                break
                        if blocking_points:
                            break
                    break
            
            if bad_y is None:
                # No bad y in N. Check S points.
                bad_x = None
                for x in current_S:
                    has_tangent = False
                    for lid in point_to_lines[x]:
                        L = valid_lines[lid]
                        inter = 0
                        for pt in L:
                            if pt in current_S:
                                inter += 1
                        if inter <= 1:
                            has_tangent = True
                            break
                    if not has_tangent:
                        bad_x = x
                        break
                        # In this case, we might need to remove other points from S to make a line "cleaner"?
                        # Or remove x from S?
                        # If x is removed, it moves to N. Then it needs a disjoint line.
                        # This is tricky. Let's assume S points are fine if U is a unital.
                        pass
                
                if bad_x is not None:
                    # Just remove bad_x?
                    current_S.remove(bad_x)
                    continue
                else:
                    break # Valid
            
            # Remove one blocking point
            # Choose randomly or the first one
            to_remove = next(iter(blocking_points))
            current_S.remove(to_remove)

    # Final N is complement of current_S
    N = all_indices - current_S
    
    # Convert to array
    points = []
    for idx in N:
        xa, xb, ya, yb = get_coords(idx)
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
