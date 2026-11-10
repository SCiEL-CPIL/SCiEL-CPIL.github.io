import numpy as np
from itertools import product

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy approach to maximize the
    complement size (i.e., minimize the Nikodym set size).
    
    The approach:
    1. Start with the trivial Nikodym set N = F_49^2 (complement size 0).
    2. Greedily try to remove points from N (add to complement S) while maintaining
       the Nikodym property.
    3. For each point, check if removing it from N still satisfies the Nikodym
       condition for all remaining points in N.
    
    This is equivalent to finding a large independent set in a conflict graph where
    vertices represent points and edges represent conflicts (removing both would
    violate the Nikodym property).
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Field arithmetic helpers for F_49 = F_7[alpha] where alpha^2 = 3
    k = 3  # alpha^2 = 3, smallest quadratic non-residue mod 7
    
    def field_add(a, b):
        """Add two elements of F_49. Each is a tuple (a, b) representing a + b*alpha."""
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        """Subtract two elements of F_49."""
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        """Multiply two elements of F_49. (a1 + a2*alpha) * (b1 + b2*alpha)
        = a1*b1 + (a1*b2 + a2*b1)*alpha + a2*b2*alpha^2
        = (a1*b1 + a2*b2*k) + (a1*b2 + a2*b1)*alpha"""
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_neg(a):
        """Negate an element of F_49."""
        return ((-a[0]) % p, (-a[1]) % p)
    
    def point_from_indices(idx):
        """Convert a linear index 0..2400 to a point [[xa, xb], [ya, yb]]."""
        # Index ordering: x varies slower, y varies faster
        # idx = (x_idx * q + y_idx)
        # x_idx = xa * p + xb
        # y_idx = ya * p + yb
        y_idx = idx % q
        x_idx = idx // q
        yb = y_idx % p
        ya = y_idx // p
        xb = x_idx % p
        xa = x_idx // p
        return [[xa, xb], [ya, yb]]
    
    def point_to_indices(pt):
        """Convert a point [[xa, xb], [ya, yb]] to a linear index."""
        xa, xb = pt[0]
        ya, yb = pt[1]
        x_idx = xa * p + xb
        y_idx = ya * p + yb
        return x_idx * q + y_idx
    
    # Precompute all lines and their points
    # Lines are defined by direction v = (1, m) for m in F_49, or v = (0, 1)
    # For v = (1, m): line through point (x, y) is {(x + t, y + t*m) : t in F_49}
    # For v = (0, 1): line through point (x, y) is {(x, y + t) : t in F_49}
    
    # Generate all possible directions
    directions = []
    # v = (1, m) for all m in F_49
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, mb))  # v = (1, ma + mb*alpha)
    # v = (0, 1)
    directions.append((0, 0, 1))
    
    # Precompute points on each line through each point
    # lines_through_point[i] = list of line_ids for lines passing through point i
    # points_on_line[line_id][t] = point index at position t
    
    # Actually, let's think differently. For each point and each direction, 
    # we can compute the line. But we need to identify lines uniquely.
    
    # A line is determined by its direction and a base point. But we can normalize
    # by using the "lowest" point on the line as the representative.
    
    # Alternative approach: For each point x in N, we need to check that for every
    # point y in N (y != x), there exists a direction such that the line through x
    # in that direction contains no other points of N except x... wait, no.
    
    # Re-reading the task: "every point x of F_49^2 (in N or not) must lie on a line 
    # {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), 
    # whose points other than x all belong to N."
    
    # So for EVERY point x (whether in N or not), there must exist a direction v 
    # such that all points x + t*v for t != 0 are in N.
    
    # This means: for each point x, at least one of the 50 lines through x must have
    # all its other 48 points in N.
    
    # Strategy: Start with N = all points. Greedily remove points from N if possible.
    # When we remove a point p from N, we need to verify that for every point x,
    # there still exists a direction where all other points on that line are in N.
    
    # This is expensive to check naively. Let's use a more efficient approach.
    
    # For each point x, maintain the number of "good" directions (directions where
    # all other points on the line are in N). Initially, all 50 directions are good.
    # When we remove a point p from N, for each point x and each direction v, if p
    # is on the line through x in direction v (and p != x), then that direction may
    # no longer be good.
    
    # Precompute for each point p, which (x, direction) pairs are affected when p is removed.
    
    # Let's precompute lines more carefully.
    
    # For each direction, we can partition F_49^2 into parallel lines.
    
    # Direction (1, m): lines are of the form {(x, y) : y - m*x = c} for various c
    # Direction (0, 1): lines are of the form {(x, y) : x = c} for various c
    
    # Let's precompute for each direction and each point, the line it belongs to.
    
    # Actually, let's just use a simpler greedy approach with periodic validation.
    
    # Given the complexity, let's use the algebraic construction but try to optimize it.
    
    # Alternative: Use a randomized greedy approach.
    
    # Start with N = all points.
    # Randomly order the points.
    # For each point in the random order, try to remove it from N.
    # To check if removal is valid, we need to verify the Nikodym property.
    
    # For efficiency, maintain for each point x the set of directions where all
    # other points on the line are in N.
    
    # Let's implement this properly.
    
    # First, precompute for each point and each direction, the other points on the line.
    
    # There are 2401 points and 50 directions.
    # For each (point, direction) pair, there are 48 other points on the line.
    
    # Memory: 2401 * 50 * 48 integers = about 5.7 million integers, which is fine.
    
    # But we also need to quickly check if all points in a list are in N.
    
    # Let's use a boolean array for N membership and maintain counts.
    
    # For each point x and direction d, let count[x][d] = number of points on the line
    # through x in direction d (excluding x) that are NOT in N.
    # Initially count[x][d] = 0 for all x, d.
    # When we remove a point p from N, for each point x and direction d such that p
    # is on the line through x in direction d (and p != x), increment count[x][d].
    # A direction d is "good" for point x iff count[x][d] == 0.
    # The Nikodym property holds iff for every point x, at least one direction d is good.
    
    # Precomputation: for each point p, for each point x != p, for which directions d
    # is p on the line through x in direction d?
    
    # Actually, for a fixed p and x, p is on the line through x in direction d iff
    # p - x is a scalar multiple of d.
    
    # Let's precompute for each pair (x, p) with x != p, the set of directions d such that
    # p is on the line through x in direction d.
    
    # This is expensive: 2401 * 2400 pairs, and for each we need to find directions.
    
    # Better: For each direction d, the lines are parallel classes. For a fixed direction d,
    # each point p lies on exactly one line in that direction. So for a fixed d, we can
    # group points by their line. Then for each point p on a line L_d, for each other point
    # x on L_d, p is on the line through x in direction d.
    
    # Let's precompute lines for each direction.
    
    # Directions: 49 of type (1, m) and 1 of type (0, 1).
    
    # For direction (1, m) where m = (ma, mb):
    # The line through (x, y) is {(x + t, y + t*m) : t in F_49}
    # Two points (x1, y1) and (x2, y2) are on the same line iff y2 - y1 = m * (x2 - x1)
    
    # For direction (0, 1):
    # The line through (x, y) is {(x, y + t) : t in F_49}
    # Two points are on the same line iff they have the same x-coordinate.
    
    # Let's precompute for each direction, the lines as lists of point indices.
    
    directions_list = []
    for ma in range(p):
        for mb in range(p):
            directions_list.append((1, ma, mb))
    directions_list.append((0, 0, 1))
    
    num_directions = len(directions_list)
    
    # lines_by_direction[d] = list of lines, each line is a list of point indices
    lines_by_direction = []
    
    for d_idx, (dx, dma, dmb) in enumerate(directions_list):
        lines = []
        used = [False] * total_points
        
        if dx == 1:
            # Group points by the invariant: y - m*x
            # For point (x, y), compute c = y - m*x
            for idx in range(total_points):
                if used[idx]:
                    continue
                pt = point_from_indices(idx)
                xa, xb = pt[0]
                ya, yb = pt[1]
                x = (xa, xb)
                y = (ya, yb)
                # Compute c = y - m*x
                cx = field_sub(y, field_mul((dma, dmb), x))
                line_points = []
                for idx2 in range(total_points):
                    pt2 = point_from_indices(idx2)
                    xa2, xb2 = pt2[0]
                    ya2, yb2 = pt2[1]
                    x2 = (xa2, xb2)
                    y2 = (ya2, yb2)
                    c2 = field_sub(y2, field_mul((dma, dmb), x2))
                    if field_sub(cx, c2) == (0, 0):
                        line_points.append(idx2)
                        used[idx2] = True
                lines.append(line_points)
        else:
            # Direction (0, 1): group by x-coordinate
            for xa in range(p):
                for xb in range(p):
                    line_points = []
                    for ya in range(p):
                        for yb in range(p):
                            idx = (xa * p + xb) * q + (ya * p + yb)
                            line_points.append(idx)
                    lines.append(line_points)
        
        lines_by_direction.append(lines)
    
    # Now precompute: for each point p, and for each direction d, which line in direction d
    # contains p? And for each other point x on that line, p is "on the line through x in direction d".
    
    # For efficiency, let's precompute for each point p, a list of (x, d) pairs such that
    # p is on the line through x in direction d (x != p).
    
    # Actually, let's maintain count[x][d] as described.
    
    # Initialize count as a 2D array: count[x][d] for x in range(total_points), d in range(num_directions)
    # count[x][d] = number of points on the line through x in direction d (excluding x) that are NOT in N.
    
    # Initially, all points are in N, so count[x][d] = 0 for all x, d.
    
    # When we remove point p from N:
    # For each direction d:
    #   Find the line L in direction d that contains p.
    #   For each point x in L (x != p):
    #     count[x][d] += 1
    
    # When we add point p back to N (undo removal):
    # For each direction d:
    #   Find the line L in direction d that contains p.
    #   For each point x in L (x != p):
    #     count[x][d] -= 1
    
    # The Nikodym property holds iff for every point x, there exists d such that count[x][d] == 0.
    
    # Let's precompute for each point p and direction d, the line index in lines_by_direction[d]
    # that contains p, and the list of other points on that line.
    
    # line_of_point[p][d] = line index
    # other_points[p][d] = list of points on the same line as p in direction d, excluding p
    
    line_of_point = [[None] * num_directions for _ in range(total_points)]
    other_points = [[None] * num_directions for _ in range(total_points)]
    
    for d in range(num_directions):
        for line_idx, line in enumerate(lines_by_direction[d]):
            for p in line:
                line_of_point[p][d] = line_idx
                other_points[p][d] = [q for q in line if q != p]
    
    # Now implement the greedy algorithm.
    
    # in_N: boolean array, initially all True.
    in_N = [True] * total_points
    
    # count[x][d]: number of points on line through x in direction d (excluding x) not in N.
    count = [[0] * num_directions for _ in range(total_points)]
    
    # good_directions_count[x]: number of directions d such that count[x][d] == 0.
    good_count = [num_directions] * total_points
    
    # valid[x]: True if good_count[x] > 0.
    # The Nikodym property holds iff all valid[x] are True.
    
    # Initially, all points are in N, so all counts are 0, all good_counts are 50, all valid.
    
    # Random order for greedy removal.
    order = list(range(total_points))
    np.random.seed(42)
    np.random.shuffle(order)
    
    # Greedily try to remove each point.
    for p in order:
        if not in_N[p]:
            continue
        
        # Try to remove p from N.
        # Check if after removal, the Nikodym property still holds.
        
        # Tentatively remove p.
        in_N[p] = False
        
        # Update counts for all x and d where p is on the line through x in direction d.
        affected = []
        for d in range(num_directions):
            line = lines_by_direction[d][line_of_point[p][d]]
            for x in line:
                if x != p:
                    old_good = (count[x][d] == 0)
                    count[x][d] += 1
                    new_good = (count[x][d] == 0)
                    if old_good and not new_good:
                        good_count[x] -= 1
                        affected.append((x, 'good_dec'))
                    elif not old_good and new_good:
                        good_count[x] += 1
                        affected.append((x, 'good_inc'))
        
        # Check if Nikodym property holds.
        # We need to check if any x has good_count[x] == 0.
        # But we only changed good_count for x in affected.
        # However, we also need to check that p itself is fine, but p is no longer in N,
        # so we don't care about p's good_count for the Nikodym property (the property
        # applies to all points in F_49^2, including those not in N).
        
        # Wait, re-reading: "every point x of F_49^2 (in N or not) must lie on a line..."
        # So the property applies to ALL points, not just those in N.
        # So we need good_count[x] > 0 for ALL x, including p.
        
        # But p's good_count may have changed because other points on lines through p
        # might have been affected... no, removing p only affects counts for x where p
        # is on the line through x. It doesn't affect counts for x = p directly, unless
        # some other point q is on a line through p, but that's not the case here.
        
        # Actually, when we remove p, we update count[x][d] for x on the same line as p.
        # p itself is not updated because we skip x == p.
        # So good_count[p] remains unchanged.
        
        # Check if all good_count[x] > 0.
        valid = True
        for x, _ in affected:
            if good_count[x] == 0:
                valid = False
                break
        if valid and good_count[p] == 0:
            valid = False
        
        if not valid:
            # Revert.
            in_N[p] = True
            for d in range(num_directions):
                line = lines_by_direction[d][line_of_point[p][d]]
                for x in line:
                    if x != p:
                        old_good = (count[x][d] == 0)
                        count[x][d] -= 1
                        new_good = (count[x][d] == 0)
                        if old_good and not new_good:
                            good_count[x] -= 1
                        elif not old_good and new_good:
                            good_count[x] += 1
    
    # Build the result: all points not in S, i.e., points in N.
    points = []
    for idx in range(total_points):
        if in_N[idx]:
            points.append(point_from_indices(idx))
    
    return np.array(points, dtype=np.int64)
