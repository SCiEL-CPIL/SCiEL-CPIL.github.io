# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.

    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[x_a, x_b], [y_a, y_b]].

    Strategy: Use a Max-SAT style local search to find a small Nikodym set.
    We represent the complement S (points NOT in N) as a binary array.
    Constraint: For every point x and every direction v (slope m or vertical),
    the line L(x, v) must contain at least one point in S other than x.
    i.e., sum_{y in L(x,v), y!=x} S[y] >= 1.

    We minimize |N| = 2401 - |S|, so we maximize |S|.
    """
    p = 7
    k_const = 3  # alpha^2 = 3
    n_points = p * p * p * p  # 2401

    # Pre-compute field operations
    # Element (a, b) represents a + b*alpha
    # (a, b) * (c, d) = (a*c + b*d*3, a*d + b*c)
    # (a, b) + (c, d) = (a+c, b+d)
    # (a, b) - (c, d) = (a-c, b-d)
    # Inverse: (a, b)^-1 = (a, -b) / (a^2 - 3b^2) if norm != 0

    def add_pt(pt1, pt2):
        return ((pt1[0] + pt2[0]) % p, (pt1[1] + pt2[1]) % p)

    def sub_pt(pt1, pt2):
        return ((pt1[0] - pt2[0]) % p, (pt1[1] - pt2[1]) % p)

    def mul_pt(pt1, pt2):
        a1, b1 = pt1
        a2, b2 = pt2
        return ((a1 * a2 + b1 * b2 * k_const) % p, (a1 * b2 + b1 * a2) % p)

    def norm_pt(pt):
        a, b = pt
        return (a * a - b * b * k_const) % p

    def inv_pt(pt):
        a, b = pt
        n = norm_pt(pt)
        if n == 0:
            return (0, 0)
        n_inv = pow(n, p - 2, p)
        return mul_pt((a, -b % p), (n_inv, 0))

    def div_pt(pt1, pt2):
        return mul_pt(pt1, inv_pt(pt2))

    # Pre-compute all points and their indices
    # Point (xa, xb, ya, yb) -> index = ((xa * p + xb) * p + ya) * p + yb
    def pt_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def idx_to_pt(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx % p
        return (xa, xb, ya, yb)

    # Pre-compute lines for all points and all directions
    # Directions: vertical (v = (0, 1)) and slopes m in F_49 (v = (1, m))
    # Total directions: 1 + 49 = 50
    # For each point x and direction v, the line is {x + t*v : t in F_49}
    # We need the set of points on the line excluding x.

    # To speed up local search, we precompute for each point x and direction v:
    # line_points[x][v] = list of indices of points on the line through x with direction v, excluding x.
    # This is a list of lists of lists: line_points[x][v] is a list of indices.

    # However, storing this explicitly might be large: 2401 * 50 * 48 integers.
    # 2401 * 50 * 48 ~ 5.7 million integers. This is fine.

    # Let's index directions: 0 for vertical, 1..49 for slopes m = (0,0), (0,1), ..., (6,6)
    # Actually, v = (1, m) where m in F_49. So m is a pair (ma, mb).
    # There are 49 such m. Plus vertical. Total 50 directions.

    # Precompute direction vectors
    directions = []
    # Vertical: (0, 1)
    directions.append(((0, 0), (1, 0)))  # (vx_a, vx_b), (vy_a, vy_b) -> v = (0, 1) means vx=0, vy=1
    # Wait, v = (vx, vy). For vertical, vx = 0, vy = 1.
    # For slope m, v = (1, m). So vx = 1, vy = m.

    # Let's re-define:
    # Direction 0: v = (0, 1)
    # Direction i (1..49): v = (1, m_i) where m_i is the i-th element of F_49.

    all_ms = []
    for ma in range(p):
        for mb in range(p):
            all_ms.append((ma, mb))

    # Precompute line points
    # line_points[x_idx][dir_idx] = list of y_idx such that y is on the line through x with direction dir, y != x.
    # We'll store this as a list of lists of lists of integers.
    # To save memory and time, we can store the counts of S points on each line for each x and dir.
    # Let's define:
    # line_s_count[x_idx][dir_idx] = sum of S[y] for y in line(x, dir), y != x.
    # Initially, we can start with a random S, then update.

    # But updating line_s_count when we flip S[x] is O(1) if we know which lines x belongs to.
    # Each point x belongs to exactly one line for each direction.
    # So when we flip S[x], we need to update line_s_count[other][dir] for all other points on the same line.
    # That's 48 updates per direction, 50 directions, so 2400 updates per flip.
    # 2400 * 2401 * steps... might be too slow.

    # Alternative: store line_s_count as a 2D array [n_points][n_dirs].
    # When we flip S[x], for each direction d, we find the line L(x, d).
    # For each y in L(x, d), y != x, we update line_s_count[y][d] by +/- 1.
    # This is 48 * 50 = 2400 operations per flip.
    # If we do 10000 flips, that's 24 million operations. This is acceptable in Python if optimized.

    # Let's precompute for each point x and direction d, the list of points y on the line through x with direction d, y != x.
    # We'll store this as a list of arrays.

    # Precomputation time: 2401 * 50 * 49 ~ 5.9 million operations. This is fine.

    # Let's do it.
    line_points = [[None] * 50 for _ in range(n_points)]

    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = pt_to_idx(xa, xb, ya, yb)
                    x_pt = ((xa, xb), (ya, yb))  # (x_coord, y_coord)

                    for d in range(50):
                        if d == 0:
                            # Vertical: v = (0, 1)
                            vx, vy = (0, 0), (1, 0)  # vx = (0,0), vy = (1,0)
                        else:
                            # Slope m
                            m = all_ms[d - 1]
                            vx, vy = (1, 0), m  # vx = (1,0), vy = m

                        # Line: x + t*v for t in F_49
                        # Points: x_pt + t * v
                        # We need to generate all 49 points and exclude x_pt.

                        pts_on_line = []
                        for ta in range(p):
                            for tb in range(p):
                                t = (ta, tb)
                                # t * v = (t * vx, t * vy)
                                tvx = mul_pt(t, vx)
                                tvy = mul_pt(t, vy)
                                # x + t*v
                                new_x = add_pt(x_pt[0], tvx)
                                new_y = add_pt(x_pt[1], tvy)
                                # If this is x itself, skip
                                if new_x == x_pt[0] and new_y == x_pt[1]:
                                    continue
                                new_idx = pt_to_idx(new_x[0], new_x[1], new_y[0], new_y[1])
                                pts_on_line.append(new_idx)

                        line_points[x_idx][d] = pts_on_line

    # Now, line_s_count[x][d] = sum of S[y] for y in line_points[x][d]
    # We'll use a NumPy array for S and line_s_count.
    # S: shape (n_points,), binary
    # line_s_count: shape (n_points, 50), integer

    # Initial S: random
    random.seed(42)
    S = np.random.choice([0, 1], size=n_points)

    # Initialize line_s_count
    line_s_count = np.zeros((n_points, 50), dtype=np.int32)
    for x_idx in range(n_points):
        for d in range(50):
            line_s_count[x_idx, d] = np.sum(S[line_points[x_idx][d]])

    # Objective: maximize |S| subject to line_s_count[x][d] >= 1 for all x, d.
    # We'll use simulated annealing.
    # Start with a valid S? No, random S is likely invalid.
    # We'll penalize violations.

    # Let's define a score function:
    # score = |S| - lambda * (number of violations)
    # We want to maximize score.

    # Actually, we want to find a valid S with maximum |S|.
    # We can use a penalty method: minimize (number of violations) - mu * |S|.
    # Or maximize |S| - lambda * (number of violations).

    # Let's try to find a valid S first, then maximize |S|.
    # Start with S = all 1s. This is valid (line_s_count[x][d] = 48 >= 1).
    # |S| = 2401.
    # Then try to flip 1s to 0s to increase |N|? No, we want to minimize |N|, so maximize |S|.
    # Wait, N is the Nikodym set. S is the complement.
    # We want to minimize |N| = 2401 - |S|, so maximize |S|.
    # But S must be such that for every x, d, line_s_count[x][d] >= 1.
    # If S = all 1s, it's valid. Can we remove points from S?
    # If we remove a point y from S (set S[y] = 0), we must ensure that for every x, d,
    # if y is in line_points[x][d], then line_s_count[x][d] remains >= 1.
    # So we can only remove y if for every x, d such that y in line_points[x][d], line_s_count[x][d] > 1.

    # This is a greedy approach: start with S = all 1s, and try to remove points.
    # But this might not be optimal. Let's try local search.

    # Let's try a different approach: start with S = all 0s, and add points.
    # But S = all 0s is invalid.
    # We need to add points until it's valid.
    # This is a hitting set problem.

    # Let's go with the penalty method and simulated annealing.
    # State: S (binary array)
    # Energy: E = -|S| + lambda * (number of violations)
    # We want to minimize E.

    # Number of violations: count of (x, d) such that line_s_count[x][d] == 0.
    # Let's maintain this count.

    violations = np.sum(line_s_count == 0)
    # Initially, S is random, so violations is likely high.

    # Let's start with S = all 1s to have a valid state.
    S = np.ones(n_points, dtype=np.int8)
    line_s_count = np.zeros((n_points, 50), dtype=np.int32)
    for x_idx in range(n_points):
        for d in range(50):
            line_s_count[x_idx, d] = 48  # Since all S are 1, and there are 48 points on each line.
    violations = 0
    s_size = n_points

    # Now, try to flip 1s to 0s.
    # A flip of S[x] from 1 to 0 is allowed if for all d, line_s_count[x][d] > 1? No.
    # When we set S[x] = 0, we decrease line_s_count[y][d] by 1 for all y, d such that x in line_points[y][d].
    # We need to ensure that after the flip, no line_s_count becomes 0.

    # Let's precompute for each x, the list of (y, d) pairs such that x in line_points[y][d].
    # This is the reverse mapping.
    # rev_line_points[x] = list of (y, d) such that x in line_points[y][d].
    # Size: 2401 * 48 * 50? No, for each x, it belongs to 50 lines (one for each d), and each line has 48 other points.
    # So for each x, there are 50 * 48 = 2400 pairs (y, d).
    # Total size: 2401 * 2400 ~ 5.7 million pairs. This is fine.

    rev_line_points = [[] for _ in range(n_points)]
    for x_idx in range(n_points):
        for d in range(50):
            for y_idx in line_points[x_idx][d]:
                # x_idx is in line_points[y_idx][d]
                rev_line_points[y_idx].append((x_idx, d))

    # Now, we can do local search.
    # Start with S = all 1s.
    # Try to flip random 1s to 0s.
    # If the flip keeps the state valid (violations == 0), accept it.
    # This is a greedy algorithm to maximize |S|.

    # But this might get stuck. Let's use simulated annealing.
    # Allow invalid states with a penalty.

    # Let's try a simple greedy first:
    # While there exists a point x with S[x] = 1 that can be removed without causing violations, remove it.

    # To check if x can be removed:
    # For all (y, d) in rev_line_points[x], we need line_s_count[y][d] > 1.
    # If yes, we can remove x.

    # Let's implement this greedy.
    # We'll iterate until no more removals are possible.

    # But this might be slow if we check all points repeatedly.
    # Let's use a queue of points to try.

    # Actually, let's try a random order and flip.
    # We'll do 10000 iterations of random flips.
    # If a flip 1->0 keeps the state valid, accept it.
    # If a flip 0->1 is needed to fix violations, accept it (but this decreases |S|).

    # Let's implement simulated annealing.
    # T starts high, decreases.
    # Move: flip a random point.
    # Delta E = change in energy.
    # E = -|S| + lambda * violations.
    # If we flip S[x] from 1 to 0:
    #   |S| decreases by 1, so -|S| increases by 1.
    #   violations may change.
    #   Delta E = 1 + lambda * Delta_violations.
    # If Delta E < 0, accept.
    # If Delta E >= 0, accept with probability exp(-Delta E / T).

    # Let's try this.
    # But we need to compute Delta_violations efficiently.
    # When we flip S[x], we update line_s_count[y][d] for all (y, d) in rev_line_points[x].
    # We can track the number of violations before and after.

    # Let's implement it.
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0

    T = 10.0
    cooling = 0.999
    lambda_penalty = 10.0

    # Current state
    S = np.ones(n_points, dtype=np.int8)
    line_s_count = np.full((n_points, 50), 48, dtype=np.int32)
    violations = 0
    s_size = n_points
    E = -s_size + lambda_penalty * violations

    best_S = S.copy()
    best_E = E
    best_s_size = s_size

    # Precompute rev_line_points as arrays for speed
    # rev_line_points[x] is a list of (y, d)
    # Let's store it as two arrays: rev_y[x], rev_d[x]
    rev_y = [None] * n_points
    rev_d = [None] * n_points
    for x in range(n_points):
        rev_y[x] = np.array([p[0] for p in rev_line_points[x]], dtype=np.int32)
        rev_d[x] = np.array([p[1] for p in rev_line_points[x]], dtype=np.int32)

    # Now, local search
    for step in range(200000):
        if time.time() > deadline - safety_margin:
            break

        # Pick a random point to flip
        x = random.randint(0, n_points - 1)

        # Compute Delta E
        # Flip S[x]
        old_val = S[x]
        new_val = 1 - old_val

        # Update line_s_count for all (y, d) in rev_line_points[x]
        # We'll do this tentatively.
        ys = rev_y[x]
        ds = rev_d[x]
        delta = 1 if new_val == 1 else -1

        # Count violations before
        # We need to know which line_s_count[y][d] are 0 before the update.
        # Let's store the old values.
        old_line_s = line_s_count[ys, ds].copy()
        new_line_s = old_line_s + delta

        # Count violations
        old_viol = np.sum(old_line_s == 0)
        new_viol = np.sum(new_line_s == 0)
        delta_viol = new_viol - old_viol

        # Delta E
        delta_s_size = 1 if new_val == 1 else -1
        delta_E = -delta_s_size + lambda_penalty * delta_viol

        # Accept?
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            # Accept the move
            S[x] = new_val
            line_s_count[ys, ds] = new_line_s
            s_size += delta_s_size
            violations += delta_viol
            E += delta_E

            # Update best
            if E < best_E:
                best_E = E
                best_S = S.copy()
                best_s_size = s_size

        T *= cooling

    # Return the best S found
    # N is the complement of S
    # We need to return the points in N.
    # N = {x : S[x] == 0}

    points = []
    for x in range(n_points):
        if best_S[x] == 0:
            xa, xb, ya, yb = idx_to_pt(x)
            points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        # Fallback: return all points
        for x in range(n_points):
            xa, xb, ya, yb = idx_to_pt(x)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
