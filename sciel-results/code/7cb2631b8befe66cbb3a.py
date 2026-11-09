# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a Hermitian unital as the excluded set.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    Element (a, b) represents a + b*alpha.
    
    Trace((a,b)) = 2a mod 7
    Norm((a,b)) = a^2 - 3*b^2 mod 7
    
    Hermitian unital: U = {(x,y) in F_49^2 : Tr(x) + N(y) = 0}
    
    For the Nikodym property:
    - For each point in U, we need a direction v=(1,m) or v=(0,1) such that
      the punctured line through that point in direction v contains no other point of U.
    - For each point NOT in U, we need a direction such that the punctured line
      contains no point of U (i.e., all other points on the line are in N = F_49^2 \ U).
    
    We verify this computationally and return N = F_49^2 \ U.
    """
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin
    
    p = 7
    q = p * p  # 49
    
    # Precompute field operations
    # (a, b) represents a + b*alpha, alpha^2 = 3
    def field_add(x, y):
        """Add two field elements represented as (a, b)"""
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def field_sub(x, y):
        """Subtract two field elements"""
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
    
    def field_mul(x, y):
        """Multiply two field elements: (a+b*alpha)*(c+d*alpha) = (ac+3bd) + (ad+bc)*alpha"""
        a, b = x
        c, d = y
        return ((a*c + 3*b*d) % p, (a*d + b*c) % p)
    
    def field_neg(x):
        """Negate a field element"""
        return ((-x[0]) % p, (-x[1]) % p)
    
    def trace(x):
        """Trace of (a,b) = 2a mod p"""
        return (2 * x[0]) % p
    
    def norm(x):
        """Norm of (a,b) = a^2 - 3*b^2 mod p"""
        return (x[0]**2 - 3 * x[1]**2) % p
    
    # Generate all field elements
    all_field_elems = [(a, b) for a in range(p) for b in range(p)]
    
    # Build the Hermitian unital: Tr(x) + N(y) = 0
    unital = set()
    for x in all_field_elems:
        for y in all_field_elems:
            if (trace(x) + norm(y)) % p == 0:
                unital.add((x, y))
    
    # The unital should have q + 1 = 50 points
    # Let's verify
    unital_list = list(unital)
    
    # For each point in the unital, find a valid direction
    # A direction is either v = (1, m) for m in F_49, or v = (0, 1)
    # The line through point (x0, y0) in direction v is: {(x0 + t*v_x, y0 + t*v_y) : t in F_49}
    
    def get_line_points(point, direction):
        """Get all points on the line through 'point' in 'direction'"""
        x0, y0 = point
        dx, dy = direction
        points = []
        for t in all_field_elems:
            # point + t * direction
            tx = field_mul(t, dx)
            ty = field_mul(t, dy)
            px = field_add(x0, tx)
            py = field_add(y0, ty)
            points.append((px, py))
        return points
    
    # For each point in the unital, find a direction where the punctured line has no other unital points
    unital_valid_directions = {}
    for pt in unital:
        found = False
        # Try vertical direction (0, 1)
        v = ((0, 0), (0, 1))
        line_pts = get_line_points(pt, v)
        other_unital = [q2 for q2 in line_pts if q2 != pt and q2 in unital]
        if len(other_unital) == 0:
            unital_valid_directions[pt] = v
            found = True
            break
        
        # Try directions (1, m) for m in F_49
        for m in all_field_elems:
            v = ((1, 0), m)
            line_pts = get_line_points(pt, v)
            other_unital = [q2 for q2 in line_pts if q2 != pt and q2 in unital]
            if len(other_unital) == 0:
                unital_valid_directions[pt] = v
                found = True
                break
        
        if not found:
            # If no direction works for this point, the unital doesn't work
            # Fall back to the axis construction
            break
    
    # Check if all unital points have valid directions
    if len(unital_valid_directions) != len(unital):
        # Fall back to the simple axis construction
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        if xa == 0 and xb == 0:
                            continue
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    # For points NOT in the unital, we need to verify there's a valid direction
    # This is the expensive part. Let's check a sample or all points.
    # For efficiency, let's check all points but with early termination if invalid.
    
    # Actually, for the Nikodym property, for points NOT in U, we need a line through them
    # such that ALL other points on that line are in N (i.e., NOT in U).
    # This means the line can intersect U at most at the point itself (but the point is not in U),
    # so the line must be disjoint from U entirely, OR the point is not in U so we just need
    # no other points of U on the line.
    
    # Wait: the point x is NOT in U. We need a line through x such that all OTHER points
    # on the line are in N = F_49^2 \ U. So the line can contain x (which is not in U) and
    # no other points of U. So we need a line through x that intersects U in at most 0 points
    # (since x is not in U, any intersection with U would be "other points").
    
    # So for x not in U: need a line through x that is disjoint from U.
    
    # Let's verify this for all points not in U
    all_points = [(x, y) for x in all_field_elems for y in all_field_elems]
    
    valid = True
    for pt in all_points:
        if pt in unital:
            continue  # Already handled
        
        # Try vertical direction
        v = ((0, 0), (0, 1))
        line_pts = get_line_points(pt, v)
        unital_on_line = [q2 for q2 in line_pts if q2 in unital]
        if len(unital_on_line) == 0:
            continue  # This direction works
        
        # Try directions (1, m)
        found = False
        for m in all_field_elems:
            v = ((1, 0), m)
            line_pts = get_line_points(pt, v)
            unital_on_line = [q2 for q2 in line_pts if q2 in unital]
            if len(unital_on_line) == 0:
                found = True
                break
        
        if not found:
            valid = False
            break
    
    if not valid:
        # Fall back to axis construction
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        if xa == 0 and xb == 0:
                            continue
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    # Build N = F_49^2 \ U
    points = []
    for x in all_field_elems:
        for y in all_field_elems:
            pt = (x, y)
            if pt not in unital:
                points.append([[x[0], x[1]], [y[0], y[1]]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
