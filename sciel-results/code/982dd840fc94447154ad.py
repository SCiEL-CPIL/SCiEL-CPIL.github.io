import numpy as np
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # Field operations for F_49 = F_7[alpha] where alpha^2 = 3
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0b0 + a1b1*k) + (a0b1 + a1b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    # Precompute all field elements as pairs
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            prod = mul_f(a, b)
            mul_table[i, j] = field_elements.index(prod)
    
    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int64)
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            s = add_f(a, b)
            add_table[i, j] = field_elements.index(s)
    
    # All points in F_49^2
    # Point index: (x_idx, y_idx) -> x_idx * q + y_idx
    # x_idx is index of x in field_elements, y_idx is index of y in field_elements
    
    # Directions: 49 slopes m (as (1, m)) + 1 vertical direction (0, 1)
    # Total 50 directions
    # For direction d:
    #   If d < 49: direction is (1, m_d) where m_d = field_elements[d]
    #   If d == 49: direction is (0, 1)
    
    n_dirs = q + 1  # 50
    
    # For each point and each direction, precompute the list of points on that line
    # line_pts[point_idx, dir_idx] = list of point indices on the line through point_idx with direction dir_idx
    # Actually, let's precompute line membership: for each line (identified by point and direction), 
    # but that's 2401 * 50 lines. Instead, let's precompute for each point and direction, the set of other points on the line.
    
    # Better: precompute for each direction and each point, the line it belongs to.
    # A line with direction d through point p can be identified by its "base point" - 
    # the unique point on the line with x-coordinate 0 (for non-vertical) or y-coordinate 0 (for vertical).
    # Actually, let's just precompute the list of points on each line.
    
    # For direction d < 49 (slope m): line through (x0, y0) is {(x0 + t, y0 + t*m) : t in F_49}
    # For direction d = 49 (vertical): line through (x0, y0) is {(x0, y0 + t) : t in F_49}
    
    # Let's precompute: for each point_idx and dir_idx, the list of all point indices on that line.
    # point_idx = x_idx * q + y_idx
    
    # Precompute line membership
    # lines[point_idx][dir_idx] = array of point indices on the line
    lines = [None] * (q * q)
    for point_idx in range(q * q):
        x_idx = point_idx // q
        y_idx = point_idx % q
        x = field_elements[x_idx]
        y = field_elements[y_idx]
        
        lines[point_idx] = [None] * n_dirs
        for d in range(n_dirs):
            if d < q:
                # Slope m = field_elements[d]
                m = field_elements[d]
                pts = []
                for t_idx in range(q):
                    t = field_elements[t_idx]
                    # new_x = x + t
                    # new_y = y + t * m
                    nx_idx = add_table[x_idx, t_idx]
                    tm_idx = mul_table[t_idx, d]
                    ny_idx = add_table[y_idx, tm_idx]
                    pt_idx = nx_idx * q + ny_idx
                    pts.append(pt_idx)
                lines[point_idx][d] = pts
            else:
                # Vertical: direction (0, 1)
                pts = []
                for t_idx in range(q):
                    t = field_elements[t_idx]
                    # new_x = x
                    # new_y = y + t
                    nx_idx = x_idx
                    ny_idx = add_table[y_idx, t_idx]
                    pt_idx = nx_idx * q + ny_idx
                    pts.append(pt_idx)
                lines[point_idx][d] = pts
    
    # Greedy algorithm to find a large independent set S
    # S is the set of excluded points. N = complement of S.
    # Constraint: for every point y and every direction d, 
    #   if y in S: exactly 1 point on line(y, d) is in S (namely y itself)
    #   if y not in S: 0 points on line(y, d) are in S
    
    # This means: for each point y in S and each direction d, no other point on line(y, d) can be in S.
    # And for each point y not in S and each direction d, no point on line(y, d) can be in S.
    
    # So S must be such that:
    # 1. For any two distinct points y1, y2 in S, there is no direction d such that y2 is on line(y1, d) or y1 is on line(y2, d).
    #    But y2 is on line(y1, d) iff y1 is on line(y2, d) (since lines are symmetric in this sense? No, not quite.)
    #    Actually, line(y1, d) contains y2 iff y2 - y1 is a scalar multiple of the direction vector of d.
    #    For d < 49: direction is (1, m). y2 - y1 = (dx, dy). y2 on line(y1, d) iff dx != 0 and dy = dx * m.
    #    For d = 49: direction is (0, 1). y2 on line(y1, d) iff dx = 0.
    
    # 2. For any point y not in S and any direction d, no point on line(y, d) is in S.
    
    # Let's use a greedy approach:
    # Start with S = empty.
    # For each point in random order, try to add it to S.
    # A point x can be added to S iff:
    #   a) For all y in S: x is not on line(y, d) for any d, and y is not on line(x, d) for any d.
    #      Actually, this is the same condition: x and y are not on the same line with any direction.
    #   b) For all y not in S (including x itself after addition): no point on line(y, d) is in S.
    #      But if we add x to S, then for any y not in S, x is now on line(y, d) iff y is on line(x, d).
    #      So we need: for all y not in S, y is not on line(x, d) for any d where line(y, d) already has a point in S.
    #      Actually, the condition is: after adding x, for every y not in S and every d, line(y, d) has 0 points in S.
    #      Before adding x, line(y, d) had 0 points in S for all y not in S and all d.
    #      After adding x, line(y, d) has 1 point in S iff x is on line(y, d).
    #      So we need: x is not on line(y, d) for any y not in S and any d.
    #      But x is on line(y, d) iff y is on line(x, d).
    #      So we need: no y not in S is on line(x, d) for any d.
    #      But line(x, d) contains q points. If any of them (other than x) is not in S, that's a problem?
    #      No: we need that for all y not in S, y is NOT on line(x, d) for any d.
    #      This means: all points on line(x, d) for all d must be in S (except x itself, but x is being added to S).
    #      Wait, that's too strong. Let me re-read.
    
    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    
    # So for point x:
    #   If x in N (x not in S): there exists a direction d such that all points on line(x, d) except x are in N (not in S).
    #   If x in S (x not in N): there exists a direction d such that all points on line(x, d) except x are in N (not in S).
    #   Wait, "whose points other than x all belong to N". So for x in S, we need a direction d such that line(x, d) \ {x} subset of N.
    #   For x in N, we need a direction d such that line(x, d) \ {x} subset of N.
    
    # So the condition is:
    #   For every point x, there exists a direction d such that line(x, d) \ {x} is disjoint from S.
    
    # This is equivalent to:
    #   For every point x, there exists a direction d such that no point on line(x, d) other than x is in S.
    
    # Let's re-derive the greedy condition:
    # We want to build S such that for every x, there exists d with line(x, d) \ {x} disjoint from S.
    
    # Greedy: start with S = empty. Try to add points to S.
    # When adding x to S, we must ensure that the condition still holds for all points.
    # For points y not on any line with x: their valid directions are unchanged.
    # For points y on line(x, d) for some d:
    #   - If y == x: x is now in S. We need a direction d' such that line(x, d') \ {x} is disjoint from S.
    #     Before adding x, line(x, d') \ {x} was disjoint from S for some d'. After adding, x is in S but x is excluded from the check, so it's still fine.
    #   - If y != x and y is on line(x, d): 
    #     Before: y had some valid direction d_y such that line(y, d_y) \ {y} disjoint from S.
    #     After: if d_y == d, then line(y, d) \ {y} now contains x (since x is on line(y, d) and x != y). So x is in S, so line(y, d) \ {y} is NOT disjoint from S.
    #     So y needs a different valid direction.
    
    # This is complex. Let's use a simpler sufficient condition for adding x to S:
    #   For all y != x: if y is on line(x, d) for some d, then y must have at least one valid direction d' != d such that line(y, d') \ {y} is disjoint from S.
    
    # Even simpler: let's just check after adding x whether the condition holds for all affected y.
    
    # Let's implement the greedy with backtracking-free approach:
    # 1. Precompute for each point and direction, the line points.
    # 2. Maintain S as a set.
    # 3. Maintain for each point y, the set of directions d such that line(y, d) \ {y} is disjoint from S.
    # 4. To add x: 
    #    a) For each direction d of x, for each y on line(x, d) with y != x:
    #       - Remove d from y's valid directions (since x is now in S and on line(y, d)).
    #    b) Check that x has at least one valid direction (i.e., line(x, d) \ {x} is disjoint from S for some d).
    #    c) Check that all affected y still have at least one valid direction.
    #    d) If all checks pass, add x to S and update.
    
    # Let's implement this.
    
    # Precompute: for each point y, the list of directions d such that y is on line(x, d) for a given x.
    # Actually, y is on line(x, d) iff x is on line(y, d). So we can use the precomputed lines.
    
    # For efficiency, let's precompute for each pair (x, y) with x != y, the set of directions d such that y is on line(x, d).
    # This is 2401 * 2400 pairs, each with up to 50 directions. That's a lot.
    
    # Alternative: for each x, for each d, the line(x, d) gives all y such that y is on line(x, d).
    # So for a given x, the affected y's are the union of line(x, d) \ {x} for all d.
    
    # Let's just do it directly.
    
    def greedy_nikodym(n_restarts=20):
        best_S = set()
        best_size = 0
        
        for restart in range(n_restarts):
            S = set()
            # valid_dirs[y] = set of directions d such that line(y, d) \ {y} is disjoint from S
            valid_dirs = [set(range(n_dirs)) for _ in range(q * q)]
            
            # Order of points to try
            order = list(range(q * q))
            random.shuffle(order)
            
            for x in order:
                # Check if x can be added to S
                # 1. x must have at least one valid direction
                if not valid_dirs[x]:
                    continue
                
                # 2. Find all affected y's: y != x such that y is on line(x, d) for some d
                affected = set()
                for d in range(n_dirs):
                    for y in lines[x][d]:
                        if y != x:
                            affected.add(y)
                
                # 3. For each affected y, check if removing the relevant directions still leaves y with a valid direction
                #    The directions to remove for y are those d such that x is on line(y, d).
                #    But x is on line(y, d) iff y is on line(x, d).
                #    So for each d, the y's on line(x, d) have d as a direction to remove.
                
                # Let's compute for each affected y, which directions to remove
                dirs_to_remove = {y: set() for y in affected}
                for d in range(n_dirs):
                    for y in lines[x][d]:
                        if y != x:
                            dirs_to_remove[y].add(d)
                
                # Check if all affected y's still have valid directions after removal
                can_add = True
                for y in affected:
                    remaining = valid_dirs[y] - dirs_to_remove[y]
                    if not remaining:
                        can_add = False
                        break
                
                if not can_add:
                    continue
                
                # Add x to S
                S.add(x)
                
                # Update valid_dirs for affected y's
                for y in affected:
                    valid_dirs[y] = valid_dirs[y] - dirs_to_remove[y]
                
                # Update valid_dirs for x: 
                # x is now in S. For x to be valid, we need a direction d such that line(x, d) \ {x} is disjoint from S.
                # But we already checked that valid_dirs[x] is non-empty.
                # However, valid_dirs[x] was computed before adding x. After adding x, 
                # line(x, d) \ {x} is disjoint from S iff no point on line(x, d) other than x is in S.
                # This is exactly what valid_dirs[x] tracks. So it's still valid.
                
                # Actually, we should also check: for x itself, is there a direction d such that line(x, d) \ {x} is disjoint from S?
                # valid_dirs[x] already tells us this. And we checked it's non-empty.
                
                # But wait: when we add x, do we need to update valid_dirs for points on lines with x?
                # Yes, we did that for affected y's.
                
                # Also, we need to update valid_dirs for x itself? 
                # No, valid_dirs[x] is the set of d such that line(x, d) \ {x} is disjoint from S.
                # Adding x to S doesn't change this, because x is excluded from the check.
                
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
        
        return best_S
    
    S = greedy_nikodym(n_restarts=20)
    
    # N = complement of S
    N = [i for i in range(q * q) if i not in S]
    
    # Convert to array of shape (k, 2, 2)
    arr = np.zeros((len(N), 2, 2), dtype=np.int64)
    for i, point_idx in enumerate(N):
        x_idx = point_idx // q
        y_idx = point_idx % q
        x = field_elements[x_idx]
        y = field_elements[y_idx]
        arr[i, 0, 0] = x[0] % p
        arr[i, 0, 1] = x[1] % p
        arr[i, 1, 0] = y[0] % p
        arr[i, 1, 1] = y[1] % p
    
    return arr
