# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy: Use the Hermitian unital U in F_49^2.
    N = F_49^2 \ S where S is a subset of U.
    
    Key insight: For a Hermitian unital U, the tangent line at u in U meets U only at u.
    For x not in U, there exists a unique u in U such that x lies on the tangent at u.
    So if we keep u in S (i.e., u not removed), x is covered by the tangent line minus u.
    
    We want to maximize S, so we keep as many points of U as possible.
    But we need every point x in N to be covered.
    
    Actually, let's think differently. Start with N = all points, S = empty.
    We want to move points from N to S while maintaining the Nikodym property.
    
    A point x in S is "safe" to be in S if removing it from N doesn't break the Nikodym property
    for any point. But this is complex.
    
    Alternative: The complement S must be such that for every point x in N, there's a line
    through x where all other points are in N.
    
    Let's try: S = U (the entire unital). N = F_49^2 \ U.
    For x in N: x is not in U. There exists u in U such that x is on the tangent at u.
    The tangent at u meets U only at u. So on that line, the other 48 points are not in U,
    hence in N. So x is covered.
    For x in U = S: x is in S, so x is in N? No, N = F_49^2 \ S, so x is NOT in N.
    But the Nikodym condition says: every point x of F_49^2 must lie on a line...
    Wait, re-reading: "every point x of F_49^2 (in N or not) must lie on a line...
    whose points other than x all belong to N."
    
    So even points in S must satisfy: there's a line through them where all other 48 points are in N.
    For x in U: the tangent at x meets U only at x. So other 48 points on tangent are in N. Good.
    But also, x could be on other lines. We just need ONE line.
    
    So S = U works! N = F_49^2 \ U.
    |U| = 49*7 + 1 = 344. So |N| = 2401 - 344 = 2057. Complement = 344.
    
    Can we do better? Can we add more points to S?
    If we add a point y not in U to S, then y must still have a line through it with all other 48 in N.
    Also, for every point x that was covered by a line passing through y, if that line had y and now y is in S,
    the line no longer has all other points in N (since y is now in S, not N).
    
    So we need to be careful. Let's start with S = U and try to add points.
    """
    start_time = time.time()
    deadline = start_time + 500.0  # Safety margin
    
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3
    
    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)
    
    def trace(a):
        return (2 * a[0]) % p
    
    def norm(a):
        return (a[0] * a[0] - a[1] * a[1] * k) % p
    
    # Point indexing: idx = ((xa * p + xb) * p + ya) * p + yb
    def point_to_index(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def index_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)
    
    # Identify points on the Hermitian Unital U: Tr(x) + Norm(y) = 0
    unital_indices = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Tr(x) = 2*xa, Norm(y) = ya^2 - 3*yb^2
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        idx = point_to_index(xa, xb, ya, yb)
                        unital_indices.add(idx)
    
    # Build lines: for each direction, generate all lines
    # Direction (1, m) for m in F_49, and (0, 1)
    # A line: {x + t*v : t in F_49}
    
    # Precompute: for each point, which lines pass through it
    # and for each line, which points are on it
    
    lines_points = []  # list of sets of point indices
    point_to_lines = {i: [] for i in range(q * q)}
    
    # Directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)
    
    for vx, vy in directions:
        if vx[0] == 1:
            # y = m*x + c
            m = vy
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = set()
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = point_to_index(xa, xb, y[0], y[1])
                            line_pts.add(idx)
                    line_id = len(lines_points)
                    lines_points.append(line_pts)
                    for idx in line_pts:
                        point_to_lines[idx].append(line_id)
        else:
            # x = c
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = set()
                    for ya in range(p):
                        for yb in range(p):
                            idx = point_to_index(ca, cb, ya, yb)
                            line_pts.add(idx)
                    line_id = len(lines_points)
                    lines_points.append(line_pts)
                    for idx in line_pts:
                        point_to_lines[idx].append(line_id)
    
    # For each point in U, find the tangent line (line meeting U only at that point)
    tangent_line = {}
    for u_idx in unital_indices:
        for line_id in point_to_lines[u_idx]:
            line_set = lines_points[line_id]
            inter = line_set & unital_indices
            if inter == {u_idx}:
                tangent_line[u_idx] = line_id
                break
    
    # Start with S = U
    S = set(unital_indices)
    N_set = set(range(q * q)) - S
    
    # Try to add points to S greedily
    # A point y not in U can be added to S if:
    # 1. y has a line through it where all other 48 points are in N
    # 2. For every point x that was "covered" by a line through y (where y was in N before),
    #    x still has another valid line
    
    # For efficiency, precompute for each point the lines that cover it
    # cover_lines[x] = list of line_ids such that (line \ {x}) ⊆ N
    
    def compute_cover_lines():
        covers = {i: [] for i in range(q * q)}
        for line_id, line_set in enumerate(lines_points):
            # Check if all points in line_set \ {x} are in N for each x in line_set
            line_set_list = list(line_set)
            for x in line_set_list:
                # Check if all other points in line are in N
                valid = True
                for y in line_set_list:
                    if y != x and y not in N_set:
                        valid = False
                        break
                if valid:
                    covers[x].append(line_id)
        return covers
    
    # This is expensive. Let's be smarter.
    # For a line L, it covers all x in L if L \ {x} ⊆ N for each x.
    # Equivalently, L \ S has at most 1 point (which is x itself if x in S, or 0).
    # Actually: for x in L, the condition is (L \ {x}) ⊆ N, i.e., L \ {x} has no points in S.
    # So L \ S ⊆ {x}. Since x is in L, this means |L ∩ S| ≤ 1.
    # If |L ∩ S| = 0, then L covers all its 49 points.
    # If |L ∩ S| = 1, say {s}, then L covers all points in L \ {s}.
    # If |L ∩ S| ≥ 2, L covers no points.
    
    def get_covered_by_line(line_id):
        line_set = lines_points[line_id]
        inter_S = line_set & S
        if len(inter_S) == 0:
            return line_set  # all 49 points covered
        elif len(inter_S) == 1:
            s = next(iter(inter_S))
            return line_set - {s}  # 48 points covered
        else:
            return set()
    
    # Initialize coverage
    covered_by = {i: set() for i in range(q * q)}  # point -> set of line_ids covering it
    for line_id in range(len(lines_points)):
        covered = get_covered_by_line(line_id)
        for x in covered:
            covered_by[x].add(line_id)
    
    # Check initial validity
    all_valid = True
    for x in range(q * q):
        if len(covered_by[x]) == 0:
            all_valid = False
            break
    
    # Try to add points not in U to S
    candidates = [i for i in range(q * q) if i not in unital_indices]
    # Sort by some heuristic: points that are on fewer covering lines first (easier to keep valid)
    # Actually, we want points that, when added to S, don't break many coverages.
    
    # Greedy: try to add each candidate if it maintains validity
    # To maintain validity, after adding y to S:
    # - y must still have a covering line
    # - For lines L containing y that previously covered some points, those points
    #   lose L as a covering line and need another one.
    
    # This is complex. Let's just try a simple greedy and validate at the end.
    
    # Alternative: keep S = U and see if we can also remove some points from U from S
    # (i.e., add them to N) while maintaining validity.
    # If we remove u from S (add to N), then:
    # - u loses its "tangent line" cover? No, u now needs a line through it with others in N.
    # - The tangent line at u: previously, it covered U \ {u} points... wait.
    
    # Let's just go with S = U for now and see the score.
    # |S| = 344, complement = 344.
    
    # Actually, let's try a different approach:
    # Start with S = empty, N = all points. All lines cover all points.
    # Try to add points to S one by one, maintaining validity.
    
    S2 = set()
    N2 = set(range(q * q))
    
    covered_by2 = {i: set(range(len(lines_points))) for i in range(q * q)}
    # Initially all lines cover all points
    
    # For each point, track which lines "support" its coverage
    # When we add a point to S, some lines may stop covering certain points.
    
    # Greedy: try to add points to S
    # Order: try unital points first (they have nice tangent properties)
    
    # For each candidate, check if adding it to S maintains all_valid
    # This is expensive to check each time. Let's use a smarter approach.
    
    # Actually, let's just use S = U which we know works.
    
    # Build N
    N_indices = sorted(range(q * q) - S)
    
    points = []
    for idx in N_indices:
        xa, xb, ya, yb = index_to_point(idx)
        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
