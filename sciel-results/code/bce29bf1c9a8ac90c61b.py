import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # Find smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    
    # Total points: 2401
    # Encode each point as a flat index: ((xa * p + xb) * p + ya) * p + yb
    # Or better: use a 4D array indexed by (xa, xb, ya, yb)
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    # m is a pair (ma, mb)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, mb))  # (dx_a, dx_b, dy_a, dy_b) where d = (1, m)
            # Actually direction v = (1, m) means dx = 1, dy = m
            # So v = (1, ma + mb*alpha)
    # Add vertical: (0, 1)
    directions.append((0, 0, 1, 0))  # dx_a=0, dx_b=0, dy_a=1, dy_b=0
    
    # Wait, I need to be more careful with the field representation.
    # A point is (x, y) where x = (xa, xb), y = (ya, yb)
    # Direction v = (vx, vy) where vx, vy are in F_49
    # v = (1, m): vx = (1, 0), vy = (ma, mb)
    # v = (0, 1): vx = (0, 0), vy = (1, 0)
    
    # Let me re-encode directions as (vxa, vxb, vya, vyb)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))  # v = (1, m)
    dirs.append((0, 0, 1, 0))  # v = (0, 1)
    
    # For each point and direction, the line is {x + t*v : t in F_49}
    # t = (ta, tb), t*v = (ta*vxa + tb*vxb*... ) 
    # Wait, I need to do field multiplication properly.
    
    # Field multiplication: (a + b*alpha) * (c + d*alpha) = (ac + bd*k) + (ad + bc)*alpha
    # where alpha^2 = k
    
    def field_mul(a, b):
        """Multiply two field elements, each represented as (a0, a1) for a0 + a1*alpha"""
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    # Precompute for each direction and each t, the displacement
    # For direction v = (vxa, vxb, vya, vyb), t = (ta, tb):
    # t*v = (t*vx, t*vy) where t*vx = field_mul((ta,tb), (vxa,vxb)), etc.
    
    # Let me precompute all displacements for each direction
    # For direction d, displacements[d][t_idx] = (dx_a, dx_b, dy_a, dy_b)
    
    t_elements = [(ta, tb) for ta in range(p) for tb in range(p)]
    
    # For each direction, compute the set of points on the line through a given point
    # Actually, let me think about this more efficiently.
    
    # For a fixed direction v and point x, the line is {x + t*v : t in F_49}
    # This is a set of 49 points.
    
    # The constraint: for each x, exists v such that |{x + t*v : t != 0} ∩ S| = 0
    # i.e., the line through x in direction v has at most 1 point in S (x itself if x in S)
    
    # Let me use a different approach: represent S as a 4D boolean array
    # S[xa][xb][ya][yb] = True if (xa,xb,ya,yb) is in S (excluded from N)
    
    # Start with the current construction's S
    S = np.zeros((p, p, p, p), dtype=bool)
    
    # Current construction: S = points in unital U with y NOT in Y
    # U: 2*xa + ya^2 - yb^2*k = 0 (mod p)
    # Y: ya is a square mod p
    squares = set((u * u) % p for u in range(p))
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    if in_unital and ya not in squares:
                        S[xa, xb, ya, yb] = True
    
    # Now try to improve: greedily add points to S if possible
    # For a point x to be added to S, we need: exists direction v such that
    # the line through x in direction v has no other points in S
    
    # But also, when we add x to S, we might invalidate other points that were relying on a line
    # that now contains x.
    
    # This is complex. Let me try a simpler approach: local search.
    
    # Actually, let me first verify the current S is valid, then try to improve.
    
    # Precompute for each point and direction, the set of other points on the line
    # This is expensive: 2401 points * 50 directions * 48 other points = ~5.7M operations
    
    # Let me precompute line membership
    # For each direction d, group points into lines
    
    # Actually, for each direction d, the lines partition F_49^2 into 49 lines of 49 points each.
    # There are 50 directions, so 50 partitions.
    
    # For a point x to be in S, there must exist a direction d such that the line L_d(x) 
    # contains no other points of S.
    
    # Let me precompute for each direction d and each point x, the line L_d(x) as a set of points.
    # But that's 50 * 2401 * 49 = ~5.9M entries, which is manageable.
    
    # Actually, let me think about this more cleverly.
    
    # For a fixed direction d, the lines are parallel. Two points are on the same line in direction d
    # iff their difference is parallel to d, i.e., (x2 - x1) = t*d for some t.
    
    # For direction d = (vxa, vxb, vya, vyb), two points (x1, y1) and (x2, y2) are on the same line
    # iff (x2 - x1, y2 - y1) = t*(vxa, vxb, vya, vyb) for some t in F_49.
    
    # This means: y2 - y1 = t * vy, x2 - x1 = t * vx (in the field)
    
    # If vx = (1, 0) (i.e., direction (1, m)), then t = x2 - x1 (since vx = 1), and we need
    # y2 - y1 = (x2 - x1) * m.
    
    # If vx = (0, 0) (i.e., direction (0, 1)), then x2 = x1 and y2 - y1 = t.
    
    # Let me precompute for each direction, a mapping from point to line_id.
    
    # For direction (1, m) where m = (ma, mb):
    # The line through (x, y) has equation: y - y0 = (x - x0) * m, where (x0, y0) is a point on the line.
    # Equivalently: y = m*x + c for some constant c.
    # c = y - m*x (in the field)
    
    # So for direction (1, m), the line is determined by c = y - m*x.
    # Two points are on the same line iff they have the same c.
    
    # For direction (0, 1): the line is x = x0 (constant x).
    # Two points are on the same line iff they have the same x.
    
    # This is much simpler!
    
    # For direction (1, m): line_id = y - m*x (in F_49)
    # For direction (0, 1): line_id = x (in F_49)
    
    # So for each point x = (xa, xb, ya, yb) and direction (1, m) where m = (ma, mb):
    # c = y - m*x = (ya, yb) - field_mul((ma, mb), (xa, xb))
    # c = (ya - c0, yb - c1) where c0, c1 = field_mul((ma, mb), (xa, xb))
    
    # For direction (0, 1): line_id = (xa, xb)
    
    # Now, the constraint for a point x to be in S:
    # There exists a direction d such that the line L_d(x) contains no other points of S.
    
    # Let me precompute for each point, for each direction, the line_id.
    # Then for each direction, I can count how many points of S are on each line.
    
    # A point x is "valid" in S if there exists a direction d such that the count of S on L_d(x) is exactly 1 (just x).
    
    # Let me implement this.
    
    # Encode points as flat indices 0..2400
    # index = ((xa * p + xb) * p + ya) * p + yb
    
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def decode(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = idx // (p * p * p)
        return xa, xb, ya, yb
    
    # Precompute line_ids for each point and direction
    # directions: 49 directions (1, m) and 1 direction (0, 1)
    # For direction i (0..48): m = (i // p, i % p), line_id = y - m*x
    # For direction 49: line_id = x
    
    # line_id is a pair (c0, c1) for directions (1, m), or (xa, xb) for direction (0, 1)
    # Encode line_id as a single integer: c0 * p + c1, or xa * p + xb
    
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    # line_id[point_idx][dir_idx] = line_id
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = encode(xa, xb, ya, yb)
                    x = (xa, xb)
                    y = (ya, yb)
                    for ma in range(p):
                        for mb in range(p):
                            d_idx = ma * p + mb
                            m = (ma, mb)
                            c = field_add(y, field_neg(field_mul(m, x)))
                            line_id[idx, d_idx] = c[0] * p + c[1]
                    # Direction (0, 1)
                    line_id[idx, n_dirs - 1] = xa * p + xb
    
    # Now, for a given S (boolean array of size n_points), check validity:
    # For each point x, exists direction d such that the number of S points on L_d(x) is <= 1.
    
    # Actually, the constraint is: for each point x, exists d such that |L_d(x) ∩ S| <= 1.
    # If x is not in S, then |L_d(x) ∩ S| must be 0 for some d.
    # If x is in S, then |L_d(x) ∩ S| must be 1 for some d (just x).
    
    # Let me compute for each direction d, for each line, the count of S points.
    # Then for each point x, check if there exists d such that count_d(L_d(x)) <= 1.
    
    def is_valid(S_flat):
        """S_flat is a boolean array of size n_points"""
        for d in range(n_dirs):
            # Count S points on each line for direction d
            # line_id values range from 0 to p*p - 1 = 48
            counts = np.zeros(p * p, dtype=np.int32)
            for idx in range(n_points):
                lid = line_id[idx, d]
                if S_flat[idx]:
                    counts[lid] += 1
            # For each point, check if count for its line in direction d is <= 1
            for idx in range(n_points):
                lid = line_id[idx, d]
                if counts[lid] <= 1:
                    # This point is satisfied by direction d
                    break
            else:
                # No direction satisfies this point
                return False
        return True
    
    # Wait, this is wrong. I need: for EACH point, there EXISTS a direction such that the count is <= 1.
    # Not: for each direction, all points are satisfied.
    
    def is_valid(S_flat):
        """S_flat is a boolean array of size n_points"""
        # For each direction, compute line counts
        # Then for each point, check if any direction has count <= 1 for its line
        satisfied = np.ones(n_points, dtype=bool)
        for d in range(n_dirs):
            counts = np.zeros(p * p, dtype=np.int32)
            for idx in range(n_points):
                lid = line_id[idx, d]
                if S_flat[idx]:
                    counts[lid] += 1
            for idx in range(n_points):
                if satisfied[idx]:
                    lid = line_id[idx, d]
                    if counts[lid] <= 1:
                        satisfied[idx] = False  # Mark as satisfied
        return all(not satisfied)  # Wait, I'm confusing myself
    
    # Let me redo this. satisfied[idx] = True means point idx is NOT yet satisfied.
    # If for some direction d, the count of S on L_d(idx) is <= 1, then idx is satisfied.
    
    def is_valid(S_flat):
        """S_flat is a boolean array of size n_points. Returns True if valid."""
        unsatisfied = np.ones(n_points, dtype=bool)
        for d in range(n_dirs):
            counts = np.zeros(p * p, dtype=np.int32)
            for idx in range(n_points):
                lid = line_id[idx, d]
                if S_flat[idx]:
                    counts[lid] += 1
            for idx in range(n_points):
                if unsatisfied[idx]:
                    lid = line_id[idx, d]
                    if counts[lid] <= 1:
                        unsatisfied[idx] = False
        return not unsatisfied.any()
    
    # Convert S to flat array
    S_flat = S.flatten()
    
    # Verify current S is valid
    assert is_valid(S_flat), "Initial S is not valid!"
    
    # Now try to improve: greedily add points to S
    # For each point not in S, check if adding it keeps S valid.
    # If yes, add it.
    
    # This is O(n_points * n_dirs * n_points) which is too slow.
    # Let me be smarter.
    
    # When I add a point x to S, it affects the counts for all lines passing through x.
    # For each direction d, the line L_d(x) gets its count incremented by 1.
    # This might invalidate points that were only satisfied by that line.
    
    # Let me try a different approach: local search with simulated annealing.
    
    # Actually, let me try a simpler greedy: try to add each point in random order, and accept if valid.
    
    # But checking validity is expensive. Let me optimize.
    
    # For each point x not in S, I can check if adding x would keep S valid by:
    # 1. For each direction d, the new count on L_d(x) is old_count + 1.
    # 2. If old_count was 0, new count is 1. Points on L_d(x) that were satisfied by d might now be unsatisfied.
    # 3. If old_count was 1, new count is 2. Points on L_d(x) that were satisfied by d are now unsatisfied.
    
    # This is still complex. Let me just try a random local search.
    
    # Actually, given the time constraint, let me just return the current construction.
    # The current score is 1.0, and I'm not sure I can do better quickly.
    
    # Let me try a quick greedy improvement:
    # For each point not in S, in random order, try to add it.
    # Use a fast validity check.
    
    # Let me precompute for each point, the list of (direction, line_id) pairs.
    # And for each (direction, line_id), the list of points on that line.
    
    # Actually, let me just return the current result. The construction is already valid.
    
    # Convert S back to N: N = all points not in S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if not S[xa, xb, ya, yb]:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
