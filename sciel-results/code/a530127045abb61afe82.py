# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    
    Strategy:
    1. Build a valid initial Nikodym set using the unital-based construction.
    2. Greedily remove points that are not essential for covering any point's
       required line, thereby maximizing the complement size.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0  # seconds before deadline to stop
    
    p = 7
    q = p * p  # 49
    n_points = q * q  # 2401
    
    # Find smallest quadratic non-residue mod 7
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    # For p=7, k=3 (since 3^3 = 27 = 6 mod 7)
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    def f_add(a, b):
        """Add two elements of F_49. a, b are (a0, a1) representing a0 + a1*alpha"""
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def f_sub(a, b):
        """Subtract two elements of F_49"""
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def f_mul(a, b):
        """Multiply two elements of F_49"""
        # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def f_neg(a):
        """Negate an element of F_49"""
        return ((-a[0]) % p, (-a[1]) % p)
    
    def f_inv(a):
        """Inverse of a in F_49. Returns (0,0) if a is zero."""
        if a[0] == 0 and a[1] == 0:
            return (0, 0)
        # Norm of a = a0^2 - a1^2*k
        norm = (a[0]*a[0] - a[1]*a[1]*k) % p
        norm_inv = pow(norm, p-2, p)
        # a^{-1} = conjugate(a) / norm(a) = (a0 - a1*alpha) / norm
        return (a[0]*norm_inv % p, (-a[1]*norm_inv) % p)
    
    # Precompute all elements of F_49 as tuples (a, b)
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Map from tuple to index
    elem_to_idx = {}
    for i, e in enumerate(field_elements):
        elem_to_idx[e] = i
    
    # Precompute addition and multiplication tables
    add_table = np.zeros((q, q, 2), dtype=np.int64)
    mul_table = np.zeros((q, q, 2), dtype=np.int64)
    
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            add_table[i, j] = f_add(a, b)
            mul_table[i, j] = f_mul(a, b)
    
    # Precompute inverses
    inv_table = np.zeros((q, 2), dtype=np.int64)
    for i, a in enumerate(field_elements):
        inv_table[i] = f_inv(a)
    
    # Point representation: index = x_a * p^3 + x_b * p^2 + y_a * p + y_b
    # This gives indices 0 to 2400.
    
    def point_to_idx(x, y):
        """Convert point (x, y) where x=(xa, xb), y=(ya, yb) to linear index"""
        return x[0] * p**3 + x[1] * p**2 + y[0] * p + y[1]
    
    def idx_to_point(idx):
        """Convert linear index to point (x, y)"""
        xb = (idx // p**2) % p
        xa = (idx // p**3) % p
        yb = idx % p
        ya = (idx // p) % p
        return ((xa, xb), (ya, yb))
    
    # Trace and Norm functions
    def trace(x):
        return (2 * x[0]) % p
    
    def norm(y):
        return (y[0]*y[0] - y[1]*y[1]*k) % p
    
    # Y = {(u^2 mod p, v) : u, v in F_p}
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    # Build initial Nikodym set
    # N = F_49^2 \ S, where S = U \ B
    # B = points of U whose y-coordinate lies in Y
    
    in_N = np.ones(n_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    idx = point_to_idx(x, y)
                    in_unital = (trace(x) + norm(y)) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        in_N[idx] = False
    
    # For each point x in F_49^2, find a valid direction.
    # We'll pick the first valid direction we find.
    
    # For point x = (x0, y0), direction v = (1, m):
    # Line: x + t*v = (x0 + t, y0 + t*m) for t in F_49
    # Punctured line: t != 0, so 48 points.
    
    # For direction v = (0, 1):
    # Line: x + t*v = (x0, y0 + t) for t in F_49
    # Punctured line: t != 0, so 48 points.
    
    # Coverage tracking
    # For each point n in N, count how many points x it covers.
    coverage_count = np.zeros(n_points, dtype=np.int64)
    # For each point x, which direction did we choose?
    chosen_direction = np.full(n_points, -1, dtype=np.int64)  # -1 means not covered yet
    
    # Directions:
    # Directions 0 to 48: v = (1, m) where m = field_elements[d]
    # Direction 49: v = (0, 1)
    
    num_directions = q + 1  # 50 directions
    
    # Precompute the punctured lines for each point and direction
    # This is expensive but we'll do it lazily or precompute key parts
    
    # For efficiency, precompute t*v for all t, v
    # t is in field_elements, v is either (1, m) or (0, 1)
    
    # Let's precompute the line points for each point and direction
    # But that's 2401 * 50 * 48 = ~5.7M points, too much to store all at once
    
    # Instead, compute on the fly but use numpy for speed where possible
    
    # Process all points to find valid directions
    for x_idx in range(n_points):
        x = idx_to_point(x_idx)
        found = False
        for d in range(num_directions):
            if d < q:
                # v = (1, m)
                m = field_elements[d]
                v = ((1, 0), m)
            else:
                # v = (0, 1)
                v = ((0, 0), (0, 1))
            
            # Check if all points on punctured line are in N
            valid = True
            line_points = []
            for t_idx in range(1, q):  # t != 0
                t = field_elements[t_idx]
                # x + t*v
                # (x0 + t*v0, y0 + t*v1)
                tv0 = f_mul(t, v[0])
                tv1 = f_mul(t, v[1])
                nx0 = f_add(x[0], tv0)
                ny0 = f_add(x[1], tv1)
                n_idx = point_to_idx(nx0, ny0)
                line_points.append(n_idx)
                if not in_N[n_idx]:
                    valid = False
                    break
            
            if valid:
                found = True
                chosen_direction[x_idx] = d
                for n_idx in line_points:
                    coverage_count[n_idx] += 1
                break
        
        if not found:
            # This shouldn't happen if the initial set is valid
            # But let's handle it by marking as uncovered
            pass
    
    # Now greedily remove points from N.
    # We'll use a more sophisticated approach:
    # For each point n in N, we track which points x it covers (i.e., x for which n is on the chosen line).
    # When we remove n, we need to find new directions for those x's.
    
    # Build reverse mapping: for each n in N, list of x's that n covers
    # This is expensive to build fully, but we can do it incrementally
    
    # Alternative: use a priority queue or repeated passes
    # Let's do repeated passes: in each pass, try to remove each point n in N
    # by checking if all x's that depend on n can be re-covered
    
    # For efficiency, let's first do a simple pass removing points with coverage_count == 0
    changed = True
    while changed:
        changed = False
        for n_idx in range(n_points):
            if in_N[n_idx] and coverage_count[n_idx] == 0:
                # Remove n_idx
                in_N[n_idx] = False
                changed = True
                # Check time
                if time.time() > deadline - safety_margin:
                    break
        
        if time.time() > deadline - safety_margin:
            break
    
    # Now do a more aggressive pruning:
    # For each point n in N, check if it can be removed by finding alternative directions
    # for all x's that currently use n in their chosen line.
    
    # Build the reverse mapping
    # covered_by[n] = set of x indices such that n is on the chosen line for x
    covered_by = [set() for _ in range(n_points)]
    
    # Rebuild by iterating over all x and their chosen lines
    for x_idx in range(n_points):
        if chosen_direction[x_idx] == -1:
            continue
        d = chosen_direction[x_idx]
        x = idx_to_point(x_idx)
        if d < q:
            m = field_elements[d]
            v = ((1, 0), m)
        else:
            v = ((0, 0), (0, 1))
        
        for t_idx in range(1, q):
            t = field_elements[t_idx]
            tv0 = f_mul(t, v[0])
            tv1 = f_mul(t, v[1])
            nx0 = f_add(x[0], tv0)
            ny0 = f_add(x[1], tv1)
            n_idx = point_to_idx(nx0, ny0)
            covered_by[n_idx].add(x_idx)
    
    # Now try to remove points greedily
    # We'll iterate multiple times, trying to remove each point in N
    # and reassigning directions for affected x's
    
    max_iterations = 100
    for iteration in range(max_iterations):
        if time.time() > deadline - safety_margin:
            break
        
        removed_any = False
        for n_idx in range(n_points):
            if time.time() > deadline - safety_margin:
                break
            
            if not in_N[n_idx]:
                continue
            
            # Get all x's that depend on n_idx
            dependent_xs = list(covered_by[n_idx])
            if not dependent_xs:
                # Can remove safely
                in_N[n_idx] = False
                removed_any = True
                continue
            
            # Try to find new directions for all dependent x's
            # without using n_idx
            can_reassign = True
            new_directions = {}
            
            for x_idx in dependent_xs:
                x = idx_to_point(x_idx)
                found_new = False
                for d in range(num_directions):
                    if d == chosen_direction[x_idx]:
                        # Skip the current direction, try others
                        pass
                    
                    if d < q:
                        m = field_elements[d]
                        v = ((1, 0), m)
                    else:
                        v = ((0, 0), (0, 1))
                    
                    # Check if all points on punctured line are in N
                    # and n_idx is not on this line (or is, but we're removing it)
                    valid = True
                    line_points = []
                    for t_idx in range(1, q):
                        t = field_elements[t_idx]
                        tv0 = f_mul(t, v[0])
                        tv1 = f_mul(t, v[1])
                        nx0 = f_add(x[0], tv0)
                        ny0 = f_add(x[1], tv1)
                        pt_idx = point_to_idx(nx0, ny0)
                        line_points.append(pt_idx)
                        if not in_N[pt_idx]:
                            valid = False
                            break
                    
                    if valid:
                        found_new = True
                        new_directions[x_idx] = d
                        break
                
                if not found_new:
                    can_reassign = False
                    break
            
            if can_reassign:
                # Remove n_idx and reassign directions
                in_N[n_idx] = False
                removed_any = True
                
                # Update coverage_count and covered_by
                for x_idx in dependent_xs:
                    d = chosen_direction[x_idx]
                    x = idx_to_point(x_idx)
                    if d < q:
                        m = field_elements[d]
                        v = ((1, 0), m)
                    else:
                        v = ((0, 0), (0, 1))
                    
                    for t_idx in range(1, q):
                        t = field_elements[t_idx]
                        tv0 = f_mul(t, v[0])
                        tv1 = f_mul(t, v[1])
                        nx0 = f_add(x[0], tv0)
                        ny0 = f_add(x[1], tv1)
                        pt_idx = point_to_idx(nx0, ny0)
                        coverage_count[pt_idx] -= 1
                        covered_by[pt_idx].discard(x_idx)
                    
                    # Assign new direction
                    new_d = new_directions[x_idx]
                    chosen_direction[x_idx] = new_d
                    
                    if new_d < q:
                        m = field_elements[new_d]
                        v = ((1, 0), m)
                    else:
                        v = ((0, 0), (0, 1))
                    
                    for t_idx in range(1, q):
                        t = field_elements[t_idx]
                        tv0 = f_mul(t, v[0])
                        tv1 = f_mul(t, v[1])
                        nx0 = f_add(x[0], tv0)
                        ny0 = f_add(x[1], tv1)
                        pt_idx = point_to_idx(nx0, ny0)
                        coverage_count[pt_idx] += 1
                        covered_by[pt_idx].add(x_idx)
        
        if not removed_any:
            break
    
    # Build the result
    points = []
    for idx in range(n_points):
        if in_N[idx]:
            x, y = idx_to_point(idx)
            points.append([[x[0], x[1]], [y[0], y[1]]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
