import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # Field arithmetic helpers
    # F_q = F_p[alpha], alpha^2 = 3
    k = 3  # alpha^2 = 3, smallest QNR mod 7
    
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def f_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Enumerate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
    
    n_points = len(points)  # 2401
    
    # Index points
    point_index = {pt: i for i, pt in enumerate(points)}
    
    # Directions: (1, m) for m in F_q, and (0, 1)
    # Represent m as (ma, mb)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 0), (0, 1)))  # (0, 1)
    
    n_dirs = len(directions)  # 50
    
    # For each point and direction, compute the set of other points on the line
    # line[i][d] = list of other point indices on the line through point i in direction d
    # This is expensive to precompute, so let's compute on the fly
    
    # Initial construction: exclude points in unital U whose y is not in Y
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    S = set()  # excluded set
    for i, pt in enumerate(points):
        xa, xb = pt[0]
        ya, yb = pt[1]
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            S.add(i)
    
    # Coverage: for each point, count how many directions cover it
    # A direction d covers point i if no other S-point is on the line through i in direction d
    
    def compute_coverage(S_set):
        """Compute for each point how many directions cover it."""
        coverage = np.zeros(n_points, dtype=np.int32)
        for i in range(n_points):
            pt_i = points[i]
            for d in range(n_dirs):
                v = directions[d]
                # Check if any other point on the line is in S
                covered = True
                for t in range(1, q):  # t = 1..48
                    # Compute pt_i + t*v
                    tx = f_add(pt_i[0], (t * v[0][0]) % p, (t * v[0][1]) % p)
                    ty = f_add(pt_i[1], (t * v[1][0]) % p, (t * v[1][1]) % p)
                    other = (tx, ty)
                    if other in point_index:
                        j = point_index[other]
                        if j in S_set:
                            covered = False
                            break
                if covered:
                    coverage[i] += 1
        return coverage
    
    # This is too slow. Let me think of a better way.
    
    # For a given direction v, the lines partition the 2401 points into 49 lines of 49 points each.
    # For a point to be covered by direction v, its entire line (except itself) must have no S-points.
    
    # Let me precompute for each direction, the line membership.
    # For direction v, point i is on line L if i = base + t*v for some base and t.
    # Two points are on the same line in direction v iff their difference is a multiple of v.
    
    # This is still complex. Let me just try a simpler greedy approach with a time limit.
    
    start_time = time.time()
    deadline = start_time + 600  # 600s deadline, leave 30s margin
    
    # Simple greedy: try to add each point to S if it maintains validity
    # Validity: every point has at least one covering direction
    
    # Precompute: for each point i and direction d, the list of other points on the line
    # This uses 2401 * 50 * 48 ≈ 5.76M entries, which is feasible
    
    # Actually, let me just compute the line members on the fly but cache them
    
    line_cache = {}
    def get_line_members(i, d):
        key = (i, d)
        if key in line_cache:
            return line_cache[key]
        pt_i = points[i]
        v = directions[d]
        members = []
        for t in range(1, q):
            tx = ((pt_i[0][0] + t * v[0][0]) % p, (pt_i[0][1] + t * v[0][1]) % p)
            ty = ((pt_i[1][0] + t * v[1][0]) % p, (pt_i[1][1] + t * v[1][1]) % p)
            other = (tx, ty)
            j = point_index[other]
            members.append(j)
        line_cache[key] = members
        return members
    
    # Compute initial coverage
    coverage = np.zeros(n_points, dtype=np.int32)
    for i in range(n_points):
        for d in range(n_dirs):
            members = get_line_members(i, d)
            has_s = False
            for j in members:
                if j in S:
                    has_s = True
                    break
            if not has_s:
                coverage[i] += 1
    
    # Greedy: try to add points to S
    # For efficiency, only try points that are currently in N (not in S)
    # And only if adding them doesn't break coverage for any point
    
    # When adding point p to S, for each direction d, for each point x on the line through p in direction d (x != p),
    # the coverage of x might decrease by 1 (if direction d was covering x)
    
    # Let's track for each point x and direction d, whether d covers x
    # cover[x][d] = 1 if d covers x, 0 otherwise
    
    # This is 2401 * 50 = 120050 booleans, feasible
    
    # Let me restructure: cover[i][d] = True if direction d covers point i
    
    # Recompute cover array
    cover = np.zeros((n_points, n_dirs), dtype=bool)
    for i in range(n_points):
        for d in range(n_dirs):
            members = get_line_members(i, d)
            has_s = False
            for j in members:
                if j in S:
                    has_s = True
                    break
            cover[i][d] = not has_s
    
    # Now greedy improvement
    # For each point p not in S, check if adding p to S keeps all coverage >= 1
    
    # When we add p to S, for each direction d, for each x on the line through p in direction d (x != p),
    # if cover[x][d] was True, it becomes False (because p is now an S-point on that line)
    # So coverage[x] decreases by 1 for each such d where cover[x][d] was True
    
    # We need to check that for all affected x, coverage[x] - (number of d where cover[x][d] was True and x is on line through p in dir d) >= 1
    
    # Let's try adding points in some order. Start with points that are "easiest" to add.
    
    # Actually, let me just try all points and see which can be added
    
    added = 0
    for p in range(n_points):
        if p in S:
            continue
        if time.time() > deadline - 30:
            break
        
        # Check if adding p to S maintains validity
        # For each direction d, find all x on the line through p in direction d (x != p)
        # For each such x, if cover[x][d] is True, x loses one coverage
        # We need to check that no x loses all its coverage
        
        # Precompute: for direction d, the line through p in direction d
        # The line through p in direction d contains p and 48 other points
        
        # For each d, the other points on the line through p in direction d are get_line_members(p, d)
        # But wait, get_line_members(p, d) gives the other points on the line through p in direction d
        # Actually, I defined it as points p + t*v for t=1..48, which are the other points
        
        # When we add p to S, for each d, for each x in get_line_members(p, d):
        #   if cover[x][d] is True, then x's coverage decreases by 1
        #   We need to ensure x's coverage doesn't drop to 0
        
        # Let's simulate
        valid = True
        for d in range(n_dirs):
            members = get_line_members(p, d)
            for x in members:
                if cover[x][d]:
                    # x would lose one coverage
                    # Check if x has other covering directions
                    other_cover = np.sum(cover[x]) - 1  # minus the one we're removing
                    if other_cover == 0:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            S.add(p)
            added += 1
            # Update cover array
            for d in range(n_dirs):
                members = get_line_members(p, d)
                for x in members:
                    if cover[x][d]:
                        cover[x][d] = False
    
    # Convert S to the Nikodym set N = all points not in S
    N = [i for i in range(n_points) if i not in S]
    
    result = np.array([points[i] for i in N], dtype=np.int64)
    # Reshape to (k, 2, 2)
    result = result.reshape(-1, 2, 2)
    
    return result
