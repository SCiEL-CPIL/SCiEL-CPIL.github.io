# EVOLVE-BLOCK-START
import time
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2, p = 7, q = 49.
    F_49 = F_7[alpha], alpha^2 = 3.
    
    The set N is the complement of an excluded set S. For every point x in F_49^2,
    there must exist a direction v (either (1, m) for some m in F_49, or (0,1))
    such that all other points on the line {x + t*v : t in F_49} (i.e., t != 0) are in N.
    
    Strategy:
    1. Start with a known valid Nikodym set (all points in N, S empty).
    2. Greedily try to add points to S (remove from N) in a good order.
    3. A point can be added to S if for at least one direction d, ALL other points
       on that line through the point are still in N.
    
    We aim to maximize |S| = 2401 - |N|.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit, leave 30s margin
    safe_deadline = deadline - 30
    
    p = 7
    q = 49
    k_field = 3  # alpha^2 = 3
    
    # Field arithmetic: element is (a, b) representing a + b*alpha
    def field_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def field_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + (a1*b2 + b1*a2)*alpha + b1*b2*alpha^2
        # = a1*a2 + b1*b2*3 + (a1*b2 + b1*a2)*alpha
        return ((a1 * a2 + 3 * b1 * b2) % p, (a1 * b2 + b1 * a2) % p)
    
    def field_scalar_mul(s, e):
        return ((s * e[0]) % p, (s * e[1]) % p)
    
    # Directions: 49 directions of type (1, m) for m in F_49, plus 1 vertical (0, 1)
    # Total 50 directions
    all_m = [(a, b) for a in range(p) for b in range(p)]
    directions = [(1, m) for m in all_m]  # 49 horizontal-ish directions
    directions.append((0, 1))  # vertical direction
    num_dirs = len(directions)  # 50
    
    # All 2401 points, indexed
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    n_pts = len(all_points)  # 2401
    
    # Map from point tuple to index
    point_to_idx = {}
    for i, pt in enumerate(all_points):
        point_to_idx[pt] = i
    
    # For each point and each direction, precompute the other points on the line
    # Line through point i in direction d: {point_i + t * d : t in F_49}
    # The "other" points are t != 0, so 48 other points
    
    # For efficiency, precompute:
    # line_other[i][d] = list of indices of other points on line through i in direction d
    # This is 2401 * 50 * 48 = ~5.7M entries, might be too much memory
    # Instead, let's compute on the fly or use a smarter approach
    
    # Actually, for a given point x and direction v, the line is {x + t*v : t in F_49}
    # We need to check if all points x + t*v for t != 0 are in N.
    
    # Let's think about this differently. For a point x to be excludable in direction d,
    # all other 48 points on that line must be in N.
    
    # Precompute for each line (point, direction pair) the set of other points
    # But that's 2401 * 50 lines, each with 49 points total.
    
    # Better: for each direction d, the lines partition F_49^2 into 49 parallel lines of 49 points each.
    # For a fixed direction d, there are 49 lines. Each line has 49 points.
    # A point on a line can be excluded in direction d if all other 48 points on that line are in N.
    
    # For each direction d, group points into lines.
    # For direction (1, m): line through (x, y) contains {(x + t, y + t*m) : t in F_49}
    # Two points (x1, y1) and (x2, y2) are on the same line in direction (1, m) iff
    # x2 - x1 = t and y2 - y1 = t*m for some t, i.e., (y2 - y1) = m*(x2 - x1) in F_49.
    # So the line is determined by the "slope intercept": y - x*m (in F_49).
    
    # For direction (0, 1): line through (x, y) contains {(x, y + t) : t in F_49}
    # Points are on same line iff they have the same x coordinate.
    
    # Precompute line assignments for each direction
    # For direction d = (1, m): line_id of point (x, y) is y - x*m (in F_49)
    # For direction d = (0, 1): line_id of point (x, y) is x (in F_49)
    
    # We need to track, for each point and each direction, how many "witness" directions
    # are still valid (i.e., all other points on that line are in N).
    
    # Let's define:
    # in_N[i] = whether point i is in N
    # For each point i and direction d, valid[i][d] = whether the line through i in direction d
    #   has all other points in N (so i could be excluded using this direction)
    
    # Initially all points are in N, so for every point and direction, valid[i][d] = True
    # because all other 48 points are in N.
    
    in_N = [True] * n_pts
    # deficit[i][d] = number of points on the line through i in direction d that are NOT in N
    # (excluding i itself)
    # Initially all 0.
    deficit = np.zeros((n_pts, num_dirs), dtype=np.int32)
    
    # For each direction, we need to know which points share the same line,
    # so we can efficiently update deficit when a point is removed from N.
    
    # Precompute: for each direction d, a mapping from line_id to list of point indices on that line.
    # For direction (1, m) with m = (ma, mb):
    #   line_id of (x, y) is y - x*m in F_49
    # For direction (0, 1):
    #   line_id of (x, y) is x in F_49
    
    dir_lines = []  # dir_lines[d] = dict mapping line_id -> list of point indices
    
    for d_idx in range(num_dirs):
        d = directions[d_idx]
        dx, dy = d[0], d[1]
        line_map = {}
        for i in range(n_pts):
            x, y = all_points[i]
            if dx[0] == 0:
                # Vertical direction (0, 1): line_id is x
                lid = x
            else:
                # Direction (1, m): line_id is y - x*m
                # x*m in F_49
                xm = field_mul(x, dy)
                lid = field_sub(y, xm)
            if lid not in line_map:
                line_map[lid] = []
            line_map[lid].append(i)
        dir_lines.append(line_map)
    
    def field_sub(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)
    
    # Now, for each point i and direction d, the line through i in direction d
    # contains the points in dir_lines[d][line_id(i, d)].
    
    # We need to compute line_id(i, d) quickly.
    # Let's precompute line_id[i][d]
    line_id = np.zeros((n_pts, num_dirs), dtype=np.int32)
    for d_idx in range(num_dirs):
        d = directions[d_idx]
        dx, dy = d[0], d[1]
        for i in range(n_pts):
            x, y = all_points[i]
            if dx[0] == 0:
                # Vertical: line_id is x = (xa, xb)
                # Encode (xa, xb) as xa * p + xb
                line_id[i, d_idx] = x[0] * p + x[1]
            else:
                # (1, m): line_id = y - x*m
                xm = field_mul(x, dy)
                lid = field_sub(y, xm)
                line_id[i, d_idx] = lid[0] * p + lid[1]
    
    # Now, for a point i to be excludable in direction d, we need:
    #   deficit[i][d] == 0 (no other points on that line are excluded)
    # AND point i is currently in N.
    
    # When we remove point j from N:
    #   For each direction d, for each other point i on the same line as j in direction d,
    #   deficit[i][d] increases by 1.
    
    # We need to efficiently find, for a given point j and direction d, all other points
    # on the same line as j in direction d.
    
    # Let's precompute for each (i, d) the list of other points on the same line.
    # This is n_pts * num_dirs * 48 = 2401 * 50 * 48 ≈ 5.7M integers, about 23MB. OK.
    
    same_line_others = []
    for d_idx in range(num_dirs):
        d_lines_map = dir_lines[d_idx]
        d_sl = []
        for i in range(n_pts):
            lid = line_id[i, d_idx]
            # Find the line in d_lines_map
            # We stored lines as dict with tuple keys, but line_id is int-encoded.
            # Let's rebuild: for each d, we need line_id -> list
            pass
        same_line_others.append(d_sl)
    
    # Actually, let's redo this more carefully.
    # For each direction d, build: encoded_line_id -> list of point indices
    enc_dir_lines = []
    for d_idx in range(num_dirs):
        d_lines_map = {}
        for i in range(n_pts):
            lid = line_id[i, d_idx]
            if lid not in d_lines_map:
                d_lines_map[lid] = []
            d_lines_map[lid].append(i)
        enc_dir_lines.append(d_lines_map)
    
    # Now same_line_others[i][d] = list of j != i such that line_id[j][d] == line_id[i][d]
    same_line_others = []
    for d_idx in range(num_dirs):
        d_sl = []
        d_map = enc_dir_lines[d_idx]
        for i in range(n_pts):
            lid = line_id[i, d_idx]
            pts_on_line = d_map[lid]
            others = [j for j in pts_on_line if j != i]
            d_sl.append(others)
        same_line_others.append(d_sl)
    # same_line_others[d][i] = list of other points
    
    # Now the greedy algorithm:
    # Maintain in_N[] and deficit[][]
    # A point i can be excluded if in_N[i] and min over d of deficit[i][d] == 0
    #   (i.e., there exists a direction d with deficit[i][d] == 0)
    
    # We want to exclude as many points as possible.
    # Greedy: repeatedly find a point that can be excluded, exclude it, update deficits.
    
    # For efficiency, maintain a set of "excludable" points.
    # Initially, all points are excludable (deficit is 0 everywhere).
    
    # When we exclude point j:
    #   For each direction d, for each i in same_line_others[d][j]:
    #       deficit[i][d] += 1
    #       If deficit[i][d] was 0 and becomes 1, and if i had no other direction with deficit 0,
    #       then i is no longer excludable.
    
    # Let's track: for each point i, the count of directions d where deficit[i][d] == 0.
    zero_dir_count = np.zeros(n_pts, dtype=np.int32)
    for i in range(n_pts):
        zero_dir_count[i] = num_dirs  # Initially all deficits are 0
    
    # Excludable set
    import heapq
    # We'll use a priority queue or just iterate.
    # For speed, let's just iterate over all points and try to exclude them.
    
    excluded = set()
    
    # Order of exclusion: we want to exclude points that are "least useful" for others.
    # A point j, when excluded, affects deficit of its 48*50 = 2400 line-mates.
    # But some line-mates might lose their last valid direction.
    
    # Heuristic: exclude points that are on many lines where many others already have
    # high deficit. Or just try all points in some order.
    
    # Let's try: repeatedly scan all points, exclude the first one that's excludable.
    # This is O(n_pts * something) per iteration, and we do at most 2401 iterations.
    
    # Better: maintain a list of currently excludable points.
    excludable = set(range(n_pts))
    
    def try_exclude(j):
        """Try to exclude point j. Returns True if successful."""
        if j not in excludable:
            return False
        if not in_N[j]:
            return False
        
        # Check if j is actually excludable (has at least one direction with deficit 0)
        has_valid_dir = False
        for d in range(num_dirs):
            if deficit[j, d] == 0:
                has_valid_dir = True
                break
        if not has_valid_dir:
            return False
        
        # Before excluding, check if any point would become non-excludable due to losing
        # its last valid direction. We don't strictly need this for correctness of the
        # final set (we just need the final set to be valid), but for the greedy to work,
        # we should ensure we don't create a situation where some point can't be excluded
        # later but we've already committed. Actually, the greedy doesn't require this -
        # we just need the FINAL set to be valid.
        
        # Wait, the validity condition is: for every point x (in N or not), there exists
        # a direction d such that all other points on line x,d are in N.
        # Points in S (excluded) also need this property!
        # So when we exclude j, we need to make sure that j itself still has a valid direction.
        # That's what has_valid_dir checks.
        # But also, we need to make sure that excluding j doesn't break the validity of
        # other points in N. A point i in N is valid if it has at least one direction d
        # where deficit[i][d] == 0. When we exclude j, deficit[i][d] increases for
        # i on the same line as j in direction d. If deficit[i][d] was 0 and becomes 1,
        # and if i had no other direction with deficit 0, then i becomes invalid.
        # So we MUST check this.
        
        # Check: for each direction d, for each i in same_line_others[d][j]:
        #   old_def = deficit[i][d]
        #   if old_def == 0:
        #       zero_dir_count[i] would decrease by 1
        #       if zero_dir_count[i] == 1:
        #           i would become invalid -> can't exclude j
        
        changes = []  # (i, d, old_def)
        for d in range(num_dirs):
            for i in same_line_others[d][j]:
                old_def = deficit[i, d]
                if old_def == 0:
                    if zero_dir_count[i] == 1:
                        # i would lose its last valid direction
                        return False
                changes.append((i, d, old_def))
        
        # Also, j itself: j's zero_dir_count is unchanged by this operation (we're not
        # changing deficit[j][d] for any d, since j is on its own line but we only update
        # OTHER points on the line). Wait, deficit[j][d] counts other points on line j,d
        # that are not in N. When we remove j from N, does deficit[j][d] change? No,
        # because deficit[j][d] counts OTHER points. So deficit[j][d] is unchanged.
        # But j is no longer in N, so j's validity is determined by whether it had a
        # valid direction BEFORE being excluded, which we checked.
        
        # Apply changes
        in_N[j] = False
        excluded.add(j)
        excludable.discard(j)
        
        for i, d, old_def in changes:
            deficit[i, d] = old_def + 1
            if old_def == 0:
                zero_dir_count[i] -= 1
                if zero_dir_count[i] == 0:
                    # i is no longer excludable (and also no longer valid if in N)
                    excludable.discard(i)
        
        return True
    
    # Greedy loop
    # To make it faster, let's try to exclude points in a smart order.
    # Points on the "unital" or special structures might be good candidates.
    
    # Let's just iterate: for each point in order, try to exclude it.
    # If we exclude one, the set of excludable points changes, so we need to rescan.
    
    # Actually, let's do multiple passes.
    for _pass in range(10):
        if time.time() > safe_deadline:
            break
        made_progress = False
        for j in range(n_pts):
            if time.time() > safe_deadline:
                break
            if j in excluded:
                continue
            if not in_N[j]:
                continue
            # Quick check: is j excludable?
            if zero_dir_count[j] == 0:
                continue
            if try_exclude(j):
                made_progress = True
        if not made_progress:
            break
    
    # Build result
    result_points = []
    for i in range(n_pts):
        if in_N[i]:
            x, y = all_points[i]
            result_points.append([[x[0], x[1]], [y[0], y[1]]])
    
    result = np.array(result_points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
