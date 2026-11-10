# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).
    
    A Nikodym set N in F_q^2 is a set such that for every point x in F_q^2,
    there exists a direction v (either (1,m) for some m in F_q, or (0,1))
    such that all other points on the line x + t*v (t != 0) are in N.
    
    Strategy: Start with all points in N, then greedily remove points (move to complement)
    while maintaining the Nikodym property. A point can be removed if for every point
    that relies on this point for its Nikodym condition, there's an alternative direction.
    """
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element (a, b) represents a + b*alpha
    # Addition: (a1,b1) + (a2,b2) = ((a1+a2)%p, (b1+b2)%p)
    # Multiplication: (a1,b1)*(a2,b2) = ((a1*a2 + 3*b1*b2)%p, (a1*b2 + a2*b1)%p)
    
    def field_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def field_sub(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)
    
    def field_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All field elements as tuples (a, b)
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(field_elements)}
    
    # Points in F_49^2: each point is (x, y) where x, y are field elements
    total_points = q * q  # 2401
    point_to_idx = {}
    point_coords = []  # (x_idx, y_idx) for each point
    for x_idx, x in enumerate(field_elements):
        for y_idx, y in enumerate(field_elements):
            pt = (x_idx, y_idx)
            point_to_idx[pt] = len(point_coords)
            point_coords.append(pt)
    
    # Directions: 50 total = 49 slopes (1, m) for m in F_49 + 1 vertical (0, 1)
    # For each direction d, and each point x, the line through x in direction d
    # consists of x + t*v_d for t in F_49
    # 
    # For a Nikodym set, for every point x, there must exist at least one direction d
    # such that all points on the line through x in direction d (except x itself) are in N.
    #
    # We'll track for each point x which directions are "satisfied" (all other points on
    # that line are in N). Initially all directions are satisfied (all points in N).
    # When we remove a point, we need to update the satisfied directions for all points
    # that share a line with the removed point.
    
    # For each direction d, precompute for each point x, which other points are on the same line
    # This is expensive but we can optimize
    
    # Alternative approach: For each direction d, group points into lines.
    # A line in direction d through point x contains x and q-1 other points.
    # For point x to be "covered" by direction d, all q-1 other points on that line must be in N.
    
    # Precompute line membership
    # lines[d][line_id] = list of points on that line
    # For each direction, there are q lines (one for each "offset")
    
    # For direction (1, m): line through (x0, y0) contains (x0 + t, y0 + m*t) for t in F_49
    # For direction (0, 1): line through (x0, y0) contains (x0, y0 + t) for t in F_49
    
    # Let's precompute for each direction d and each point x, the set of other points on the same line
    # This will be used to check if removing a point breaks coverage
    
    # More efficient: For each direction d, for each point x, store the line_id and line members
    # Actually, let's think about this differently.
    
    # State: in_N[i] = True if point i is in N
    # valid_dir[x][d] = True if direction d "covers" point x (all other points on that line are in N)
    # covered[x] = any(valid_dir[x][d] for d in range(50))
    
    # Initially, all points are in N, so all directions cover all points.
    # valid_dir is a 2D array: [total_points, 50], initially all True
    
    # When we remove point p from N:
    # For each direction d, find all points x such that p is on the line through x in direction d
    # For those x, if p was the only missing point on that line (or now becomes missing),
    # valid_dir[x][d] may become False.
    
    # Let's precompute: for each point p and direction d, which points x have p on their line in direction d?
    # This is the same as: which lines in direction d contain p? For each such line, which points x are on it?
    
    # For direction d = (1, m):
    # A line is determined by its "intercept". For line through (x0, y0):
    # Points are (x0 + t, y0 + m*t) for t in F_49.
    # Given a point p = (px, py), the lines in direction d containing p are... wait,
    # for a fixed direction d, each point is on exactly one line.
    # So for point p and direction d, there's exactly one line, containing q points.
    # The other q-1 points on that line are the ones that need to be in N for p to be covered by d.
    
    # So: for each point p and direction d, let line_members[p][d] = set of other points on the same line.
    # When p is removed, for each x in line_members[p][d], we need to check if all other
    # points on x's line (in direction d) are still in N. If not, valid_dir[x][d] = False.
    
    # But this is O(q) per (p, d) pair, and we have q * 50 = 2401 * 50 = 120050 such pairs.
    # Each removal involves checking up to 50 directions, each with up to q-1 = 48 other points.
    # Total per removal: 50 * 48 = 2400 checks.
    # Total for all 2401 points: 2401 * 2400 = ~5.7M checks. Feasible.
    
    # Let's precompute line_members[p][d] as a list of other point indices
    
    # Directions: d = 0 to 48 for slopes (1, m) where m = field_elements[d], d = 49 for vertical (0,1)
    
    # Precomputation
    # For each point p and direction d, compute the other points on the same line
    
    line_members = {}  # (p, d) -> list of other point indices
    
    # For direction (1, m) where m = field_elements[d] for d in 0..48
    for d in range(49):
        m = field_elements[d]
        for p_idx in range(total_points):
            px, py = point_coords[p_idx]
            p_elem = field_elements[p_idx]
            # Wait, p_idx is not (px, py) directly. Let me reindex.
            # Actually, point_coords[p_idx] = (x_idx, y_idx) where x_idx, y_idx are indices into field_elements
            # So the actual field element for x is field_elements[point_coords[p_idx][0]]
            # And for y is field_elements[point_coords[p_idx][1]]
            x_elem = field_elements[point_coords[p_idx][0]]
            y_elem = field_elements[point_coords[p_idx][1]]
            
            # Line through p in direction (1, m): points are (x + t, y + m*t) for t in F_49
            # For t = 0: (x, y) = p
            # For t != 0: other points
            other_points = []
            for t_idx in range(1, q):  # t = field_elements[t_idx]
                t_elem = field_elements[t_idx]
                new_x = field_add(x_elem, t_elem)
                new_y = field_add(y_elem, field_mul(m, t_elem))
                new_x_idx = elem_to_idx[new_x]
                new_y_idx = elem_to_idx[new_y]
                other_pt_idx = point_to_idx[(new_x_idx, new_y_idx)]
                other_points.append(other_pt_idx)
            
            line_members[(p_idx, d)] = other_points
    
    # For direction (0, 1) [vertical], d = 49
    for p_idx in range(total_points):
        x_elem = field_elements[point_coords[p_idx][0]]
        y_elem = field_elements[point_coords[p_idx][1]]
        
        # Line: (x, y + t) for t in F_49
        other_points = []
        for t_idx in range(1, q):
            t_elem = field_elements[t_idx]
            new_y = field_add(y_elem, t_elem)
            new_y_idx = elem_to_idx[new_y]
            other_pt_idx = point_to_idx[(point_coords[p_idx][0], new_y_idx)]
            other_points.append(other_pt_idx)
        
        line_members[(p_idx, 49)] = other_points
    
    # Now greedy removal
    start_time = time.time()
    deadline = start_time + 500  # Safety margin before 600s limit
    
    in_N = np.ones(total_points, dtype=bool)
    valid_dir = np.ones((total_points, 50), dtype=bool)
    valid_count = np.ones(total_points, dtype=np.int32)  # number of valid directions for each point
    
    # Count how many times each point is "needed" for each direction
    # needed[p][d] = number of points x such that p is on x's line in direction d
    # This is useful for quick checks
    
    # Actually, let's just do the greedy:
    # Order points by some heuristic (e.g., random, or by degree)
    # Try to remove each point; if it breaks coverage for any point, skip it.
    
    order = list(range(total_points))
    random.seed(42)
    random.shuffle(order)
    
    best_excluded = 0
    best_state = None
    
    for p_idx in order:
        if time.time() > deadline:
            break
        
        if not in_N[p_idx]:
            continue
        
        # Check if we can remove p_idx
        # For each direction d, find all x such that p_idx is on x's line in direction d
        # For those x, check if removing p_idx makes valid_dir[x][d] = False
        # If any x loses all directions, we cannot remove p_idx
        
        can_remove = True
        affected = []  # (x, d) pairs that might be invalidated
        
        for d in range(50):
            # Find all x such that p_idx is on x's line in direction d
            # This is equivalent to: x is on the same line as p_idx in direction d
            # The line through p_idx in direction d contains p_idx and 48 other points
            # All those other points x have p_idx on their line in direction d
            other_pts = line_members[(p_idx, d)]
            for x in other_pts:
                # x is on the same line as p_idx in direction d
                # Currently, valid_dir[x][d] might be True or False
                # If valid_dir[x][d] is True, removing p_idx might invalidate it
                # We need to check: are all other points on x's line (in direction d) in N?
                # p_idx is one of those points. After removing p_idx, we need to check the rest.
                # But we don't want to do O(48) work per (x, d) pair.
                
                # Optimization: maintain for each (x, d) the count of missing points on x's line
                # missing_count[x][d] = number of points on x's line in direction d that are NOT in N
                # valid_dir[x][d] = (missing_count[x][d] == 0)
                
                # Let's use this approach instead.
                pass
        
        # Let me restructure with missing_count
        break  # This approach needs rethinking
    
    # Restart with better data structure
    in_N = np.ones(total_points, dtype=bool)
    # missing_count[x][d] = number of points on x's line in direction d that are not in N (excluding x itself)
    # Initially 0 for all
    missing_count = np.zeros((total_points, 50), dtype=np.int32)
    
    # For each point p and direction d, we need to know which points x have p on their line
    # Let's precompute reverse: rev[p][d] = list of x such that p is on x's line in direction d
    # But this is the same as: for each x, line_members[(x, d)] gives the other points
    # So rev[p][d] = [x for x if p in line_members[(x, d)]]
    # Which is the same as: the points on the same line as p in direction d (including p? No, just others)
    # Actually, if p is on x's line in direction d, then x is on p's line in direction d (symmetric)
    # So rev[p][d] = line_members[(p, d)]  (the other points on p's line)
    
    # Great! So rev[p][d] = line_members[(p, d)]
    
    # Now greedy:
    for p_idx in order:
        if time.time() > deadline:
            break
        
        if not in_N[p_idx]:
            continue
        
        # Tentatively remove p_idx
        # For each direction d, for each x in rev[p_idx][d] = line_members[(p_idx, d)]:
        #   missing_count[x][d] += 1
        #   If missing_count[x][d] becomes 1 (was 0), valid_dir[x][d] becomes False
        #   If valid_count[x] becomes 0, cannot remove p_idx
        
        can_remove = True
        changes = []  # (x, d) pairs where missing_count increased
        
        for d in range(50):
            for x in line_members[(p_idx, d)]:
                old_mc = missing_count[x, d]
                missing_count[x, d] = old_mc + 1
                changes.append((x, d, old_mc))
                if old_mc == 0:  # valid_dir[x][d] was True, now becomes False
                    valid_count[x] -= 1
                    if valid_count[x] == 0:
                        can_remove = False
                        # Roll back
                        for (cx, cd, cmd) in changes:
                            missing_count[cx, cd] = cmd
                            if cmd == 0:
                                valid_count[cx] += 1
                        break
            if not can_remove:
                break
        
        if can_remove:
            in_N[p_idx] = False
            best_excluded += 1
            # Save state if improved
            # Actually, we keep going and track best at the end
    
    # Build result
    points = []
    for i in range(total_points):
        if in_N[i]:
            x_idx, y_idx = point_coords[i]
            xa, xb = field_elements[x_idx]
            ya, yb = field_elements[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
