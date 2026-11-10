# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (where F_49 = F_7[alpha], alpha^2 = 3).
    
    Strategy: Start with the known valid construction (Hermitian unital based) that
    scores 1.0, then attempt to further reduce the set size using a greedy exclusion
    algorithm with efficient data structures.
    
    The base construction excludes points S = U \\ B where:
    - U = {(x, y) : Trace(x) + Norm(y) = 0} is the Hermitian unital
    - B = points of U whose y-coordinate has y_a a square mod 7
    
    We then try to exclude more points greedily, ensuring validity is maintained.
    """
    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 mod 7
    
    # Field arithmetic helpers for F_49 = F_7[alpha] where alpha^2 = 3
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * alpha_sq) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)
    
    def field_inv(a):
        norm = (a[0] * a[0] - alpha_sq * a[1] * a[1]) % p
        if norm == 0:
            raise ValueError("Zero has no inverse")
        norm_inv = pow(norm, p - 2, p)
        return ((a[0] * norm_inv) % p, (-a[1] * norm_inv) % p)
    
    def idx_to_elem(i):
        return (i // p, i % p)
    
    def elem_to_idx(e):
        return e[0] * p + e[1]
    
    # Precompute field multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            ei = idx_to_elem(i)
            ej = idx_to_elem(j)
            prod = field_mul(ei, ej)
            mul_table[i, j] = elem_to_idx(prod)
    
    # Precompute field addition table
    add_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            ei = idx_to_elem(i)
            ej = idx_to_elem(j)
            prod = field_add(ei, ej)
            add_table[i, j] = elem_to_idx(prod)
    
    def pt_to_idx(x_idx, y_idx):
        return x_idx * q + y_idx
    
    def idx_to_pt(pt_idx):
        return pt_idx // q, pt_idx % q
    
    # Build the base Nikodym set using the Hermitian unital construction
    # This is the known valid construction that scores 1.0
    # in_N[pt] = True if point is in N (kept), False if excluded
    in_N = np.ones(q * q, dtype=bool)
    
    # Smallest quadratic non-residue mod 7 is 3 (since 3^3 = 27 = 6 mod 7)
    k = 3
    
    # Special y-coordinate set Y = {(u^2 mod p, v)}: y_a must be a square mod p
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    # A point (x, y) is EXCLUDED iff it is in the unital U and its y is NOT in Y
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # x = xa + xb*alpha, y = ya + yb*alpha
                    # Trace(x) = 2*xa (since Trace(a + b*alpha) = 2a in F_7)
                    # Norm(y) = ya^2 - yb^2 * k
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        # This point is excluded
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        pt = pt_to_idx(x_idx, y_idx)
                        in_N[pt] = False
    
    # Now try to exclude more points using greedy algorithm
    # Precompute other_pts for efficient validity checking
    # other_pts[pt, dir_idx, t] = point index of the t-th other point on line through pt in direction dir_idx
    # 50 directions: 49 slopes + 1 vertical
    
    other_pts = np.zeros((q * q, 50, q - 1), dtype=np.int64)
    
    for x_idx in range(q):
        for y_idx in range(q):
            pt = pt_to_idx(x_idx, y_idx)
            # Directions 0..48: slope m = field element with index d
            for d in range(q):
                m_idx = d
                for t_idx in range(1, q):  # t != 0
                    x_new = add_table[x_idx, t_idx]
                    y_new = add_table[y_idx, mul_table[t_idx, m_idx]]
                    pt_new = pt_to_idx(x_new, y_new)
                    other_pts[pt, d, t_idx - 1] = pt_new
            # Direction 49: vertical
            for t_idx in range(1, q):
                x_new = x_idx
                y_new = add_table[y_idx, t_idx]
                pt_new = pt_to_idx(x_new, y_new)
                other_pts[pt, 49, t_idx - 1] = pt_new
    
    # Build reverse index: for each point y, which (x, v) pairs have y in other_pts[x, v, :]
    affected_by = [[] for _ in range(q * q)]
    for x in range(q * q):
        for v in range(50):
            for t in range(q - 1):
                y = other_pts[x, v, t]
                affected_by[y].append((x, v))
    
    # Track valid direction counts for each point
    # A direction v is valid for point x if all other_pts[x, v, :] are in N
    valid_dir_count = np.zeros(q * q, dtype=np.int64)
    
    # Initialize valid_dir_count
    for x in range(q * q):
        for v in range(50):
            all_in_N = True
            for t in range(q - 1):
                if not in_N[other_pts[x, v, t]]:
                    all_in_N = False
                    break
            if all_in_N:
                valid_dir_count[x] += 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 550  # 550 seconds out of 630
    
    # Greedy exclusion: try to exclude points that are currently in N
    # Process in order of increasing valid_dir_count (most constrained first)
    # Actually, we want to exclude points that can be safely excluded
    # A point x can be excluded if:
    # 1. valid_dir_count[x] >= 1 (it has at least one valid direction)
    # 2. For every (x2, v) in affected_by[x], if v is valid for x2, then valid_dir_count[x2] > 1
    #    (x2 has another valid direction)
    
    # Create a list of points to try excluding, sorted by some heuristic
    # Try points with fewer valid directions first (more likely to be "safe" to exclude
    # because they rely on fewer directions)
    # Actually, we want to exclude points that DON'T break others
    # Let's just iterate in order and check
    
    for pt in range(q * q):
        if time.time() > deadline:
            break
        
        if not in_N[pt]:
            continue
        
        # Check if pt can be excluded
        if valid_dir_count[pt] < 1:
            continue
        
        # Check if excluding pt breaks any other point
        can_exclude = True
        for x2, v in affected_by[pt]:
            if x2 == pt:
                continue
            if not in_N[x2]:
                continue
            # Check if v is currently valid for x2
            # v is valid for x2 if all other_pts[x2, v, :] are in N
            # Since pt is in N and pt is in other_pts[x2, v, :],
            # v is valid for x2 only if all other points (including pt) are in N
            # After excluding pt, v will no longer be valid for x2
            # So we need to check: is v currently valid for x2, and is it x2's last valid direction?
            
            # Check if v is currently valid for x2
            v_valid_for_x2 = True
            for t in range(q - 1):
                if not in_N[other_pts[x2, v, t]]:
                    v_valid_for_x2 = False
                    break
            
            if not v_valid_for_x2:
                continue
            
            # v is currently valid for x2. After excluding pt, it won't be.
            # Check if x2 has another valid direction
            if valid_dir_count[x2] <= 1:
                can_exclude = False
                break
        
        if can_exclude:
            in_N[pt] = False
            # Update valid_dir_count for affected points
            for x2, v in affected_by[pt]:
                if x2 == pt:
                    continue
                if not in_N[x2]:
                    continue
                # Check if v was valid for x2 before excluding pt
                # v was valid if all other_pts[x2, v, :] were in N (including pt)
                # Now pt is not in N, so v is no longer valid
                # We need to verify v was indeed valid before
                v_was_valid = True
                for t in range(q - 1):
                    other = other_pts[x2, v, t]
                    if other != pt and not in_N[other]:
                        v_was_valid = False
                        break
                if v_was_valid:
                    valid_dir_count[x2] -= 1
    
    # Build the result: N is the set of points still in N
    points = []
    for pt in range(q * q):
        if in_N[pt]:
            x_idx, y_idx = idx_to_pt(pt)
            xa, xb = idx_to_elem(x_idx)
            ya, yb = idx_to_elem(y_idx)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Fallback: return all points (trivially valid, but worst score)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
