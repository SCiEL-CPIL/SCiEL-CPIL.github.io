# EVOLVE-BLOCK-START
import time
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy local search to maximize the
    size of the complement S (points not in N). A set N is Nikodym if every point
    x in F_49^2 lies on a line {x + t*v : t in F_49} with v = (1, m) or v = (0, 1)
    such that all points on that line other than x belong to N. Equivalently, the
    complement S is a set of points such that for every point x (in S or N), there
    exists a direction v where the punctured line L(x, v) \\ {x} is fully contained
    in N, i.e., L(x, v) \\ {x} is disjoint from S.

    We model the problem as finding a large set S such that no two points in S
    "block" each other's unique directions. Specifically, if x, y in S, and y lies
    on L(x, v), then x cannot use direction v. If x loses all 50 directions because
    of other points in S, then x cannot be in S.

    We use a greedy algorithm starting from a small seed S (e.g., a vertical line or
    parabola) and iteratively add points that don't violate the Nikodym property.
    To maximize speed and result quality, we use a "valid counts" approach:
    for each point x, we track how many directions v are still "open" (i.e., the
    punctured line L(x, v) \\ {x} is disjoint from current S).
    A point x can be added to S only if it has at least one open direction.
    However, adding a point y to S reduces the open direction counts of points x
    where y lies on L(x, v).

    We return N = F_49^2 \\ S.
    """
    start_time = time.time()
    deadline = 600.0

    p = 7
    q = 49
    k = 3  # alpha^2 = 3

    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * k) % p, (a0 * b1 + a1 * b0) % p)

    # Precompute all points and field elements
    points = []
    for a in range(p):
        for b in range(p):
            points.append((a, b))
    
    # Map from point tuple to index
    pt_to_idx = {pt: i for i, pt in enumerate(points)}

    # Directions: 49 slopes m in F_49, plus vertical (0, 1). Total 50 directions.
    # Let's index directions 0..48 for slopes m, and 49 for vertical.
    num_dirs = 50

    # Precompute lines: for each point x and direction v, the set of points on the line.
    # Actually, we only need to know, for a pair (x, y), which direction v has y on L(x, v).
    # If y == x, no direction.
    # If y != x:
    #   If x and y have same x-coordinate, they are on the vertical line through x.
    #   Else, the slope m = (y_x - x_x) * (y_y - x_y)^(-1) ... wait, line is y = m * x + b?
    #   No, line direction is (1, m). Points are x + t*(1, m) = (x0 + t, y0 + t*m).
    #   So if y = (y0, y1), x = (x0, x1), then y = x + t*(1, m) implies:
    #   y0 = x0 + t => t = y0 - x0
    #   y1 = x1 + t*m => m = (y1 - x1) * t^(-1)
    #   If t = 0 (i.e., y0 == x0), then we need y1 == x1 (which is y==x) or the vertical direction.
    #   So:
    #   If y0 == x0:
    #       If y1 == x1: y == x
    #       Else: y is on vertical line through x. Direction index 49.
    #   Else:
    #       t = (y0 - x0) mod p
    #       m = (y1 - x1) * t^(-1) mod p
    #       Direction index = m (where m is an element of F_49, mapped to 0..48).
    #       Note: m is a field element (a,b). We need a consistent mapping to 0..48.
    #       Let's say index = a + b*p.

    # Precompute inverse for field elements
    inv = {}
    for a in range(p):
        for b in range(p):
            if (a, b) == (0, 0):
                continue
            # (a, b)^(-1) = (a - b*alpha)/(a^2 - 3b^2)
            # Denominator = a^2 - 3b^2 mod p
            den = (a * a - 3 * b * b) % p
            if den == 0:
                # This should not happen for non-zero elements if 3 is non-residue?
                # 3 is a non-residue mod 7? 3^3 = 27 = 6 = -1. Yes, non-residue.
                # So a^2 - 3b^2 = 0 implies a^2 = 3b^2 => (a/b)^2 = 3. No solution.
                pass
            den_inv = pow(den, p - 2, p)
            # (a, b)^(-1) = (a * den_inv, -b * den_inv)
            inv[(a, b)] = (a * den_inv % p, -b * den_inv % p)
    inv[(0,0)] = (0,0) # Just for safety, won't be used for t=0

    # Precompute direction for pair (x, y)
    # dir_idx[x_idx][y_idx] = direction index or -1 if x==y
    dir_idx = [[-1] * q for _ in range(q)]
    for i in range(q):
        x = points[i]
        for j in range(q):
            y = points[j]
            if i == j:
                dir_idx[i][j] = -1
                continue
            if x[0] == y[0]:
                # Vertical
                dir_idx[i][j] = 49
            else:
                t = (y[0] - x[0]) % p
                m = (y[1] - x[1]) % p
                # m = t * slope => slope = m * t^(-1)
                # t is an integer in 1..6, so its inverse in F_49 is (t_inv, 0)
                t_inv = pow(t, p - 2, p)
                # slope = (m * t_inv, 0)
                slope_a = m * t_inv % p
                slope_b = 0
                dir_idx[i][j] = slope_a + slope_b * p # 0..48

    # Initialize S (complement)
    # Start with a small set, e.g., the vertical line x = 0? Or parabola?
    # Let's start with the vertical line x=0. Points (0, b) for b in F_49.
    S = set()
    for b in range(p):
        S.add(pt_to_idx[(0, b)])
    
    # valid_counts[x][v] = 1 if L(x, v) \ {x} is disjoint from S, else 0
    # Initially S is small, so most are valid.
    valid_counts = np.ones((q, num_dirs), dtype=np.uint8)
    
    # We need to initialize valid_counts correctly based on initial S.
    # For each x, and each v, check if any y in S \ {x} lies on L(x, v).
    # This is O(|S| * q * num_dirs) which is small.
    for x_idx in range(q):
        x = points[x_idx]
        for v in range(num_dirs):
            # Check if any y in S lies on L(x, v)
            # Actually, it's easier to just recompute from scratch for each x.
            # But we can use dir_idx.
            # For a fixed x and v, the points on L(x, v) are x + t*(1, m) or vertical.
            # We just need to check if S intersects L(x, v) \ {x}.
            # Let's iterate y in S.
            is_valid = 1
            for y_idx in S:
                if y_idx == x_idx:
                    continue
                if dir_idx[x_idx][y_idx] == v:
                    is_valid = 0
                    break
            valid_counts[x_idx, v] = is_valid

    # Greedy loop
    # We want to add points to S to maximize |S|.
    # A point x can be added if sum(valid_counts[x]) > 0.
    # When we add y to S, we must update valid_counts for all x.
    # For each x != y, let v = dir_idx[x][y]. If v != -1, then L(x, v) now contains y.
    # So valid_counts[x][v] becomes 0.
    
    # We'll keep a list of candidate points that are currently valid (have at least one open dir).
    # Actually, any point not in S is a candidate.
    # We can just iterate through all points not in S, and try to add them if possible.
    # To be efficient, we can prioritize points that have many open directions?
    # Or just random order?
    # The "valid count" greedy usually works well with any order, but random helps avoid local minima.
    
    # Let's use a simple loop:
    # While time allows:
    #   Pick a random point x not in S.
    #   If sum(valid_counts[x]) > 0, add it to S.
    #   Update valid_counts.
    #   If no point can be added, restart or stop?
    #   Restarting with a different seed might help.
    
    # But we need to be careful: adding a point might make other points invalid.
    # So we should re-check validity after each addition?
    # No, valid_counts is maintained incrementally.
    # If valid_counts[x][v] was 1, and we add y such that dir_idx[x][y]==v, we set it to 0.
    # It never goes back to 1. So valid_counts[x] is non-increasing.
    # Thus, if a point x is invalid now (all 0s), it will remain invalid.
    # So we only need to consider points that are currently valid.
    
    # Let's maintain a list of "active" candidates.
    # Initially, all points not in S are active if they have any valid direction.
    # When we add y, we check all x. If x becomes invalid (sum becomes 0), we remove it from active.
    
    # For simplicity and speed, let's just scan all points in random order repeatedly.
    # Since q=49, the number of points is small.
    
    rng = np.random.default_rng(42)
    
    # To ensure we find the maximum, we might need to be smart.
    # But a simple greedy is often good.
    
    # Let's try a few random restarts if time permits.
    best_S = S
    best_size = len(S)
    
    for attempt in range(10): # 10 restarts max
        if time.time() > deadline - 60:
            break
            
        # Reset S to initial seed
        S = set()
        for b in range(p):
            S.add(pt_to_idx[(0, b)])
        
        valid_counts = np.ones((q, num_dirs), dtype=np.uint8)
        for x_idx in range(q):
            for v in range(num_dirs):
                is_valid = 1
                for y_idx in S:
                    if y_idx == x_idx:
                        continue
                    if dir_idx[x_idx][y_idx] == v:
                        is_valid = 0
                        break
                valid_counts[x_idx, v] = is_valid
        
        # Candidate list: all points not in S
        candidates = [i for i in range(q) if i not in S]
        rng.shuffle(candidates)
        
        added = True
        while added:
            if time.time() > deadline - 60:
                break
            added = False
            # Pick next candidate
            if not candidates:
                break
            
            x_idx = candidates.pop()
            # Check if x_idx is still valid
            if np.sum(valid_counts[x_idx]) > 0:
                # Add x_idx to S
                S.add(x_idx)
                # Update valid_counts
                for y_idx in range(q):
                    if y_idx == x_idx:
                        continue
                    v = dir_idx[y_idx][x_idx]
                    if v != -1:
                        valid_counts[y_idx, v] = 0
                # If adding x_idx made some candidates invalid, we don't remove them from `candidates` list
                # explicitly, we just check when we pop them.
                added = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()

    # Construct N
    N_indices = [i for i in range(q) if i not in best_S]
    
    # Convert to array
    points_N = np.array([list(points[i]) for i in N_indices], dtype=np.int64)
    # Shape should be (k, 2, 2)
    # points[i] is (a, b) for x and (c, d) for y?
    # Wait, points list contains (a, b) which represents a single F_49 element.
    # A point in F_49^2 is (x, y) where x, y in F_49.
    # So a point is ((x_a, x_b), (y_a, y_b)).
    # My `points` list only contains elements of F_49.
    # I need to represent points in F_49^2.
    
    # Let's redo the point representation.
    # Total points = 49 * 49 = 2401.
    
    # I made a mistake in the implementation above.
    # `points` should be all pairs (x, y) in F_49 x F_49.
    
    # Let's rewrite the core logic properly.

    # --- Correct Implementation ---
    
    p = 7
    k = 3
    q = 49
    
    # Field ops
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def mul_f(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * k) % p, (a0 * b1 + a1 * b0) % p)
        
    # All field elements
    field_elems = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(field_elems)}
    
    # Inverses
    inv_f = {}
    for e in field_elems:
        if e == (0, 0):
            continue
        a, b = e
        den = (a*a - k*b*b) % p
        den_inv = pow(den, p-2, p)
        inv_f[e] = (a*den_inv % p, -b*den_inv % p)
    inv_f[(0,0)] = (0,0)

    # All points in F_49^2
    all_points = []
    for x in field_elems:
        for y in field_elems:
            all_points.append((x, y))
            
    total_points = len(all_points) # 2401
    pt_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Directions
    # 0..48: slope m
    # 49: vertical
    num_dirs = 50
    
    # dir_idx[x_idx][y_idx] = v if y lies on L(x, v), else -1
    # L(x, v) for v<49: x + t*(1, m) where m = field_elems[v]
    # L(x, 49): vertical line through x
    
    # Precompute dir_idx
    # This is a 2401x2401 matrix. 2401^2 = 5.7M entries.
    # Storing as int8 or int16 should be fine memory-wise (5.7MB).
    # But computing it might take a moment.
    dir_idx = np.full((total_points, total_points), -1, dtype=np.int16)
    
    for i in range(total_points):
        x = all_points[i]
        for j in range(total_points):
            y = all_points[j]
            if i == j:
                continue
            x0, x1 = x
            y0, y1 = y
            if x0[0] == y0[0] and x0[1] == y0[1]:
                # Same x-coordinate
                if x1 == y1:
                    continue # Same point
                # Vertical
                dir_idx[i, j] = 49
            else:
                # t = y0 - x0
                t = add_f(y0, inv_f[(x0[0], -x0[1] % p)]) # Wait, inv_f is for field elements.
                # t = y0 - x0
                t = ((y0[0] - x0[0]) % p, (y0[1] - x0[1]) % p)
                # m = (y1 - x1) * t^(-1)
                diff_y = ((y1[0] - x1[0]) % p, (y1[1] - x1[1]) % p)
                if t == (0, 0):
                    # Should be handled by vertical case
                    continue
                t_inv = inv_f[t]
                m = mul_f(diff_y, t_inv)
                v = elem_to_idx[m]
                dir_idx[i, j] = v

    # Greedy Search
    best_S = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Time limit check
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
            
        # Initialize S
        # Start with a random line? Or empty?
        # Empty is safest, but might grow slowly.
        # Let's start with a random point.
        S = set()
        if total_points > 0:
            S.add(rng.integers(0, total_points))
            
        valid_counts = np.ones((total_points, num_dirs), dtype=np.uint8)
        
        # Update valid_counts for initial S
        for x_idx in range(total_points):
            for y_idx in S:
                if x_idx == y_idx:
                    continue
                v = dir_idx[x_idx, y_idx]
                if v != -1:
                    valid_counts[x_idx, v] = 0
                    
        # Candidates
        candidates = [i for i in range(total_points) if i not in S]
        rng.shuffle(candidates)
        
        while candidates:
            if time.time() > deadline - 30:
                break
                
            x_idx = candidates.pop()
            if np.sum(valid_counts[x_idx]) > 0:
                S.add(x_idx)
                # Update
                # For all y, if dir_idx[y, x_idx] != -1, set valid_counts[y, v] = 0
                # This is O(total_points) per addition.
                # Total additions <= total_points.
                # So O(total_points^2) per restart.
                # 2401^2 = 5.7M ops. Fast.
                
                # Optimization: only update y where valid_counts[y, v] is 1?
                # We can just do it for all y.
                for y_idx in range(total_points):
                    if y_idx == x_idx:
                        continue
                    v = dir_idx[y_idx, x_idx]
                    if v != -1:
                        valid_counts[y_idx, v] = 0

        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # Construct N
    N_indices = [i for i in range(total_points) if i not in best_S]
    
    # Format output
    # Shape (k, 2, 2)
    # Each point is [[x_a, x_b], [y_a, y_b]]
    out = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        x, y = all_points[idx]
        out[i, 0, 0] = x[0]
        out[i, 0, 1] = x[1]
        out[i, 1, 0] = y[0]
        out[i, 1, 1] = y[1]
        
    return out

# EVOLVE-BLOCK-END
