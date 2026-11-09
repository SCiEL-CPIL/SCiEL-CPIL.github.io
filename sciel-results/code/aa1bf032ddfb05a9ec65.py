# EVOLVE-BLOCK-START
import time
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49, p = 7, alpha^2 = 3.
    Field: F_49 = F_7[alpha], elements are (a,b) representing a + b*alpha.
    Point: [[x_a, x_b], [y_a, y_b]].
    
    The Nikodym set N must satisfy: for every point x in F_49^2, there exists a
    direction v = (1, m) for some m in F_49, or v = (0, 1), such that all points
    on the line {x + t*v : t in F_49} except x itself are in N.
    
    We aim to minimize |N| (equivalently maximize complement size).
    
    Strategy:
    1. Start with the baseline construction from the seed program.
    2. Iteratively try to remove points from N while maintaining the Nikodym property.
    3. Use greedy removal: a point can be removed if for every direction v through it,
       there's already another point on that line (besides the point being removed and
       the center point) that is in N, OR the line has only the center + this point.
    
    Actually, let me think more carefully about the Nikodym condition.
    
    For a point x to be "covered" by N, there must exist a direction v such that
    all points on line L(x,v) = {x + t*v : t in F_49} except x are in N.
    
    So if we want to remove a point p from N, we need to check: for every point x 
    that currently uses a line through p as its covering line, does x still have 
    another valid covering line?
    
    This is complex. Let me use a different approach.
    
    Alternative: Use the structure of the problem. The directions are (1,m) for m in F_49 
    (49 directions) and (0,1) (1 direction), total 50 directions.
    
    For each direction v, the lines form a partition of F_49^2 into 49 lines of 49 points each.
    
    A point x is covered by direction v if all other 48 points on the line through x in 
    direction v are in N.
    
    Let me try a greedy algorithm:
    1. Start with the baseline N.
    2. For each point p in N (in some order), try to remove it.
    3. After removing p, check if the Nikodym property still holds.
    4. If yes, keep it removed; if no, restore it.
    
    But checking the full Nikodym property is expensive. Let me think of a better way.
    
    Actually, for each point x, let's track which directions "cover" it (i.e., all other 
    points on that line are in N). When we remove a point p, it may affect the coverage 
    of points x that share a line with p in some direction.
    
    Let me implement this with a time limit.
    """
    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3
    
    # Time limit guard
    start_time = time.time()
    time_limit = 620  # seconds, with safety margin
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # = (a0*b0 + a1*b1*3) + (a0*b1 + a1*b0)*alpha
        r0 = (a[0] * b[0] + a[1] * b[1] * k_const) % p
        r1 = (a[0] * b[1] + a[1] * b[0]) % p
        return (r0, r1)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_pow(a, n):
        if n == 0:
            return (1, 0)
        if n == 1:
            return a
        result = a
        for _ in range(1, n):
            result = field_mul(result, a)
        return result
    
    # Generate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    field_elem_set = set(field_elements)
    
    # Map field element to index
    elem_to_idx = {elem: i for i, elem in enumerate(field_elements)}
    
    # Point representation: (x, y) where x, y are field elements
    # All points in F_49^2
    all_points = [(x, y) for x in field_elements for y in field_elements]
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = [(1, m) for m in field_elements]
    directions.append((0, 1))
    n_directions = len(directions)  # 50
    
    # For each direction, compute the line structure
    # A line in direction v through point x: {x + t*v : t in F_49}
    # Precompute for each direction and each point, the set of points on the line
    
    # Actually, let's think about it differently. For a fixed direction v, the lines
    # are determined by the "perpendicular" coordinate. But since we're in a field,
    # for v = (1, m), the lines are parameterized by the value of y - m*x (in field arithmetic).
    # For v = (0, 1), the lines are vertical lines with fixed x.
    
    # Let me precompute the line memberships.
    # For each direction d, and for each point, which line does it belong to, and what are the other points?
    
    # This is 50 * 2401 * 49 = 5,882,450 operations for precomputation, which is fine.
    
    # line_membership[dir_idx][point_idx] = list of point indices on the same line
    line_membership = []
    for d_idx, d in enumerate(directions):
        dm, dn = d  # direction is (dm, dn) where each is a field element
        
        # Group points by line
        # For direction (1, m): lines are y - m*x = c, so group by (y - m*x)
        # For direction (0, 1): lines are x = c, so group by x
        
        lines = {}
        for pt_idx, pt in enumerate(all_points):
            x, y = pt
            if dm[0] == 1 and dm[1] == 0:  # direction (1, 0)... wait, directions are (1, m) where m is field elem
                # Actually dm is a field element, not a tuple
                pass
        
        # Let me redo this. Directions are v = (vx, vy) where vx, vy are field elements.
        # For v = (1, m) where m is a field element: line through (x,y) is {(x+t, y+m*t) : t in F_49}
        # The "line identifier" can be (y - m*x) since (y - m*x) is invariant along the line.
        # For v = (0, 1): line through (x,y) is {(x, y+t) : t in F_49}, identifier is x.
        
        for pt_idx, pt in enumerate(all_points):
            x, y = pt
            if d_idx < q:  # direction (1, m) where m = field_elements[d_idx]
                m = field_elements[d_idx]
                # line_id = y - m*x
                line_id = field_sub(y, field_mul(m, x))
            else:  # direction (0, 1)
                line_id = x
            
            if line_id not in lines:
                lines[line_id] = []
            lines[line_id].append(pt_idx)
        
        # Now build line_membership for this direction
        lm = [[] for _ in range(len(all_points))]
        for line_id, pts in lines.items():
            for pt_idx in pts:
                lm[pt_idx] = pts
        
        line_membership.append(lm)
    
    # Baseline construction from seed program
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    in_N = [True] * len(all_points)  # N starts as all points
    for i, pt in enumerate(all_points):
        x, y = pt
        xa, xb = x
        ya, yb = y
        in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            in_N[i] = False  # excluded
    
    # Count current N size
    n_size = sum(in_N)
    print(f"Baseline N size: {n_size}, complement: {len(all_points) - n_size}")
    
    # For each point, track which directions "cover" it
    # A point x is covered by direction d if all other points on line(x, d) are in N
    
    def is_covered_by_direction(pt_idx, d_idx):
        """Check if pt_idx is covered by direction d_idx (all other points on line are in N)."""
        line_pts = line_membership[d_idx][pt_idx]
        for other_idx in line_pts:
            if other_idx != pt_idx and not in_N[other_idx]:
                return False
        return True
    
    def is_covered(pt_idx):
        """Check if pt_idx is covered by at least one direction."""
        for d_idx in range(n_directions):
            if is_covered_by_direction(pt_idx, d_idx):
                return True
        return False
    
    # Greedy removal: try to remove points from N
    # Order: try to remove points that are "redundant" - i.e., their removal doesn't break coverage
    
    # For efficiency, let's track for each point how many directions cover it
    cover_count = [0] * len(all_points)
    for pt_idx in range(len(all_points)):
        for d_idx in range(n_directions):
            if is_covered_by_direction(pt_idx, d_idx):
                cover_count[pt_idx] += 1
    
    print(f"Cover counts computed")
    
    # Greedy: try to remove points with high cover count first (they're less critical)
    # Actually, we want to remove points that can be removed without breaking anything.
    # A point p can be removed if for every point x that is on a line with p in some direction,
    # x still has at least one other covering direction after p is removed.
    
    # This is complex. Let's use a simpler heuristic:
    # Try to remove points in random order, checking if the Nikodym property holds.
    
    import random
    random.seed(42)
    
    # For faster checking, let's maintain for each point the number of covering directions
    # When we remove a point p, for each direction d, the points on line(p, d) may lose coverage.
    
    # Let me implement a simpler version with time limit
    removed = [False] * len(all_points)
    removed_count = 0
    
    # Try to remove points, starting with those that have the most covering directions
    candidates = sorted(range(len(all_points)), key=lambda i: -cover_count[i])
    
    attempts = 0
    for pt_idx in candidates:
        if time.time() - start_time > time_limit:
            break
        if not in_N[pt_idx]:
            continue
        if removed[pt_idx]:
            continue
        
        attempts += 1
        if attempts % 10000 == 0:
            print(f"Attempt {attempts}, removed {removed_count}, N size: {sum(in_N)}")
        
        # Temporarily remove pt_idx
        in_N[pt_idx] = False
        removed[pt_idx] = True
        removed_count += 1
        
        # Check if Nikodym property still holds
        # Only need to check points that share a line with pt_idx in some direction
        affected = set()
        for d_idx in range(n_directions):
            for other_idx in line_membership[d_idx][pt_idx]:
                if other_idx != pt_idx:
                    affected.add(other_idx)
        
        valid = True
        for other_idx in affected:
            if not in_N[other_idx]:
                continue  # not in N, don't need to check
            # Check if other_idx is still covered
            if not is_covered(other_idx):
                valid = False
                break
        
        if not valid:
            # Restore
            in_N[pt_idx] = True
            removed[pt_idx] = False
            removed_count -= 1
            if attempts % 1000 == 0:
                print(f"  Restored pt {pt_idx}")
    
    print(f"Final removed: {removed_count}, N size: {sum(in_N)}, complement: {len(all_points) - sum(in_N)}")
    
    # Build output
    points = []
    for i, pt in enumerate(all_points):
        if in_N[i]:
            x, y = pt
            points.append([[x[0], x[1]], [y[0], y[1]]])
    
    if not points:
        # Should not happen, but safety
        points = [[[0, 0], [0, 0]]]
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
