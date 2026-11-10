# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses a greedy set cover approach to minimize the removed set R from the Hermitian unital,
    thereby maximizing the complement S.
    """
    start_time = time.time()
    deadline = start_time + 500.0

    p = 7
    q = p * p
    k = 3

    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def trace(a):
        return (2 * a[0]) % p

    def norm(a):
        return (a[0] * a[0] - a[1] * a[1] * k) % p

    # Generate all points in F_49^2
    # Index: x = (xa, xb), y = (ya, yb)
    # idx = ((xa * p + xb) * p + ya) * p + yb
    
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    # Identify points on the Hermitian Unital U
    # Tr(x) + Norm(y) = 0
    unital_points = []
    unital_indices = set()
    for i, (x, y) in enumerate(all_points):
        if (trace(x) + norm(y)) % p == 0:
            unital_points.append(i)
            unital_indices.add(i)
            
    # N is the complement of S.
    # S = U \ R.
    # We want to find R subset of U such that for all x in (F^2 \ U) U R,
    # there is a line through x disjoint from S.
    # This is equivalent to: for all x in N, there is a tangent T_u (u in R) passing through x.
    # Note: The tangent T_u is the unique line through u and P_inf? No.
    # The tangent at u is the unique line meeting U only at u.
    
    # We need to compute the tangent line for each u in U.
    # T_u is the set of points x such that the line ux meets U only at u.
    
    # Efficient way:
    # Precompute lines.
    # Directions: (1, m) for m in F_49, and (0, 1).
    # Total q+1 = 50 directions.
    
    # For each direction v, the lines are parallel classes.
    # We need to map each point to the lines passing through it.
    
    # Let's represent lines by their direction and a representative point.
    # Or just store the set of points for each line.
    # Number of lines = q * (q+1) = 49 * 50 = 2450.
    # Each line has 49 points.
    
    lines = []
    point_to_lines = {i: [] for i in range(q*q)}
    
    # Directions
    # v = (vx, vy)
    # We need to generate unique directions.
    # (1, m) for m in F_49, and (0, 1).
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb))) # (1, m)
    directions.append(((0, 1), (0, 0))) # (0, 1)
    
    # For each direction, generate lines.
    # A line is determined by a point x and direction v.
    # To avoid duplicates, we can use the lexicographically smallest point on the line.
    
    seen_lines = set()
    
    for vx, vy in directions:
        # Generate lines in this direction
        # We can iterate over all points and "claim" the line if this point is the smallest.
        # Or simpler: for each line, the points are {x + t*v : t in F_49}.
        # We can iterate over a "base" point for each line.
        
        # For direction (1, m), the lines are graphs y = m*x + c.
        # For direction (0, 1), the lines are x = c.
        
        if vx[0] == 1:
            # y = m*x + c
            m = vy
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            # y = m*x + c
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = ((xa * p + xb) * p + y[0]) * p + y[1]
                            line_points.append(idx)
                    
                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        for idx in line_points:
                            point_to_lines[idx].append(len(lines) - 1)
                        
        else:
            # x = c
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for ya in range(p):
                        for yb in range(p):
                            idx = ((ca * p + cb) * p + ya) * p + yb
                            line_points.append(idx)
                    
                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        for idx in line_points:
                            point_to_lines[idx].append(len(lines) - 1)

    # For each u in U, find the tangent line.
    # T_u is the line L such that L intersect U == {u}.
    
    tangent_line_for_u = {}
    for u_idx in unital_points:
        u = all_points[u_idx]
        # Check all lines passing through u
        candidate_tangents = []
        for line_id in point_to_lines[u_idx]:
            line_set = lines[line_id]
            # Count intersection with U
            inter_count = 0
            for p_idx in line_set:
                if p_idx in unital_indices:
                    inter_count += 1
                    if inter_count > 1:
                        break
            if inter_count == 1:
                candidate_tangents.append(line_id)
        
        # There should be exactly one tangent
        if len(candidate_tangents) == 1:
            tangent_line_for_u[u_idx] = candidate_tangents[0]
        else:
            # Fallback: pick the first one or handle error
            if candidate_tangents:
                tangent_line_for_u[u_idx] = candidate_tangents[0]
            else:
                # Should not happen
                tangent_line_for_u[u_idx] = -1

    # Set Cover Problem
    # Universe: Points in N = F^2 \ U
    # Sets: For each u in U, the set of points in N covered by T_u.
    # S_u = (T_u \ {u}) \cap N
    
    universe = [i for i in range(q*q) if i not in unital_indices]
    universe_set = set(universe)
    
    sets_for_u = {}
    for u_idx in unital_points:
        line_id = tangent_line_for_u[u_idx]
        if line_id == -1:
            continue
        line_set = lines[line_id]
        # Points in N on this line
        s_u = set()
        for p_idx in line_set:
            if p_idx in universe_set:
                s_u.add(p_idx)
        sets_for_u[u_idx] = s_u

    # Greedy Set Cover with Local Search
    # We first run greedy to get a baseline.
    # Then we try to remove points from R if possible (improving the cover).
    
    uncovered = set(universe)
    R = []
    
    while uncovered:
        best_u = None
        best_covered = set()
        
        for u_idx in unital_points:
            if u_idx in R:
                continue
            s_u = sets_for_u[u_idx]
            covered = s_u.intersection(uncovered)
            if len(covered) > len(best_covered):
                best_covered = covered
                best_u = u_idx
        
        if best_u is None:
            break
            
        R.append(best_u)
        uncovered.difference_update(best_covered)
        
        if time.time() > deadline:
            break

    # Local Search: Try to remove points from R.
    # A point u in R can be removed if every point in S_u is covered by some other u' in R \ {u}.
    # We can check this by maintaining a count of how many sets in R cover each point in the universe.
    
    R_set = set(R)
    # count_covered[p] = number of u in R such that p in S_u
    count_covered = {p: 0 for p in universe}
    for u in R:
        for p in sets_for_u[u]:
            count_covered[p] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove points from R in random order
        R_list = list(R)
        random.shuffle(R_list)
        
        for u in R_list:
            if u not in R_set:
                continue
            # Check if u can be removed
            can_remove = True
            for p in sets_for_u[u]:
                # If p is covered by exactly 1 set (which is u), then we cannot remove u
                if count_covered[p] == 1:
                    can_remove = False
                    break
                    
            if can_remove:
                # Remove u
                R_set.remove(u)
                R.remove(u)
                for p in sets_for_u[u]:
                    count_covered[p] -= 1
                improved = True
                if time.time() > deadline:
                    break

    # Construct S = U \ R
    S = [i for i in unital_points if i not in R_set]
    
    # N = F^2 \ S
    N = [i for i in range(q*q) if i not in set(S)]
    
    # Convert indices to points
    points = []
    for idx in N:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
