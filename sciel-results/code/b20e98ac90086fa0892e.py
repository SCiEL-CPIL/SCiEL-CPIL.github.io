import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    deadline = time.time() + 580  # safety margin before 630s kill

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Represent field elements as integers 0..48: element a + b*alpha = a + b*p

    def field_add(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        return (xa + ya) % p + ((xb + yb) % p) * p

    def field_mul(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        ra = (xa * ya + xb * yb * 3) % p
        rb = (xa * yb + xb * ya) % p
        return ra + rb * p

    def field_sub(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        return (xa - ya) % p + ((xb - yb) % p) * p

    def field_neg(x):
        return field_sub(0, x)

    def field_div(x, y):
        ya, yb = y % p, y // p
        norm = (ya * ya - yb * yb * 3) % p
        norm_inv = pow(norm, p - 2, p)
        inv_a = (ya * norm_inv) % p
        inv_b = ((-yb) * norm_inv) % p
        return field_mul(x, inv_a + inv_b * p)

    def point_id(x, y):
        """Encode point (x, y) in F_49^2 as integer 0..2400"""
        return x + y * q

    def decode_point(pid):
        y = pid // q
        x = pid % q
        return x, y

    # Directions in F_49^2:
    # (1, m) for m in 0..48, and (0, 1)
    # For a point x = (x1, x2), the line through x with direction (1, m) contains:
    #   (x1 + t, x2 + m*t) for t in F_49
    # The line through x with direction (0, 1) contains:
    #   (x1, x2 + t) for t in F_49

    # For S to be a valid excluded set (complement of N), we need:
    # For every point x in S, there exists a line L through x such that
    # L \ {x} ⊆ N, i.e., L ∩ S = {x}.
    # This means: for each x in S, there exists a direction d (either (1,m) for some m,
    # or (0,1)) such that no other point of S lies on the line x + t*d.

    # For direction (1, m): the line through (x1, x2) contains points (x1+t, x2+m*t).
    # Another point (x1', x2') is on this line iff x1' - x1 = t and x2' - x2 = m*t,
    # i.e., x2' - x2 = m*(x1' - x1), i.e., x2' - m*x1' = x2 - m*x1.
    # So two points are on the same slope-m line iff they have the same value of (x2 - m*x1).

    # For direction (0, 1): the line through (x1, x2) contains points (x1, x2+t).
    # Another point (x1', x2') is on this line iff x1' = x1.

    # So for S to be valid:
    # For each point x in S, there must exist either:
    #   - Some m in F_49 such that no other point of S has the same (x2 - m*x1) value, OR
    #   - The vertical direction (0,1): no other point of S has the same x1 coordinate.

    # Strategy: Build S greedily. At each step, add a point that still allows all
    # previously added points to have at least one valid direction, AND the new point
    # itself has at least one valid direction.

    # For efficiency, track for each point in S which directions are "blocked" by other points in S.

    # Let's use a smarter approach:
    # A point x = (x1, x2) is "isolated" in direction (1, m) if no other point in S
    # has the same x2 - m*x1 value.
    # A point x is "isolated" vertically if no other point in S has the same x1.

    # For each point in S, we need at least one direction where it's isolated.

    # Greedy: maintain for each candidate point the set of directions that are still open.
    # When we add a point, we close directions for other points that share the same line.

    # Actually, let's think about it differently.
    # For a given set S, point x is valid if:
    #   |{y in S \ {x} : y1 = x1}| < q-1 AND (there exists m such that |{y in S \ {x} : y2 - m*y1 = x2 - m*x1}| < q-1)
    # Wait, no. The condition is that there EXISTS a line through x with no other S points.
    # For direction (1, m): the line through x with direction (1,m) intersects S at exactly
    #   {y in S : y2 - m*y1 = x2 - m*x1}
    # We need this set to be exactly {x}, i.e., |{y in S \ {x} : y2 - m*y1 = x2 - m*x1}| = 0.
    # For direction (0, 1): we need |{y in S \ {x} : y1 = x1}| = 0.

    # So point x is valid iff:
    #   (|{y in S \ {x} : y1 = x1}| == 0) OR
    #   (exists m in F_49: |{y in S \ {x} : y2 - m*y1 = x2 - m*x1}| == 0)

    # This is equivalent to:
    #   x is the only point in S with its x1 coordinate, OR
    #   x is the only point in S with its (x2 - m*x1) value for some m.

    # Let's build S greedily. We want to maximize |S|.

    # Key insight: if S is a set where every pair of points is on a unique line for some
    # direction, we're fine. But we just need each point to have at least one "clean" direction.

    # Simplest valid S: any set where all points have distinct x1 coordinates AND for each point,
    # there's some m where its (x2 - m*x1) is unique. If all x1 are distinct, vertical direction works.

    # So any set of points with distinct x1 coordinates is valid! |S| <= q = 49.
    # Can we do better? If two points share x1, neither can use vertical direction.
    # They'd each need some m where their (x2 - m*x1) values are unique.

    # For points (a, b1) and (a, b2) with same x1=a:
    # For direction (1, m): b1 - m*a vs b2 - m*a. These differ by b1 - b2 ≠ 0.
    # So for any m, they have different (x2 - m*x1) values! So they don't block each other.
    # But they might block other points.

    # Actually, the condition for point (a, b1) is: exists m such that no other point in S
    # has (x2 - m*x1) = b1 - m*a.
    # For point (a, b2): exists m' such that no other point in S has (x2 - m'*x1) = b2 - m'*a.

    # These m and m' can be different! So having same x1 doesn't immediately invalidate.

    # Let's try a greedy approach to build a large S.

    import random
    random.seed(123)

    # Precompute all points
    all_points = []
    for x1 in range(q):
        for x2 in range(q):
            all_points.append((x1, x2))

    # For greedy construction, we need efficient checking.
    # For a candidate set S (as a set of (x1, x2) tuples):
    #   For each point in S, check if it's valid.

    # To speed up, maintain:
    #   count_by_x1[x1] = number of points in S with x1 coordinate = x1
    #   For each m, count_by_slope_m[(m, val)] = number of points in S with x2 - m*x1 = val
    #   But there are 49 * 49 = 2401 slope-value pairs, times 49 slopes = too many to maintain all.

    # Alternative: for each point in S, check validity by iterating over all other points.
    # This is O(|S|^2) per check. For |S| up to ~100, that's 10000 ops per point, 100 points = 1e6.
    # With 580 seconds, we can do many such checks.

    # Let's do a randomized greedy: start with empty S, repeatedly add a random point
    # that keeps S valid.

    def is_point_valid_in_S(pt, S):
        """Check if pt is valid in set S (pt is in S)."""
        x1, x2 = pt
        # Check vertical: any other point in S with same x1?
        if any(other[0] == x1 for other in S if other != pt):
            pass  # vertical is blocked, need to check slopes
        else:
            return True  # vertical works

        # Check all slopes
        for m in range(q):
            val = field_sub(x2, field_mul(m, x1))
            blocked = False
            for other in S:
                if other == pt:
                    continue
                ox1, ox2 = other
                oval = field_sub(ox2, field_mul(m, ox1))
                if oval == val:
                    blocked = True
                    break
            if not blocked:
                return True
        return False

    def is_set_valid(S):
        """Check if all points in S are valid."""
        S_set = set(S)
        for pt in S_set:
            if not is_point_valid_in_S(pt, S_set):
                return False
        return True

    # Greedy construction
    S = set()
    remaining = list(all_points)
    random.shuffle(remaining)

    for pt in remaining:
        if time.time() > deadline - 50:
            break
        # Try adding pt
        S.add(pt)
        if is_set_valid(S):
            pass  # keep it
        else:
            S.remove(pt)

    # Try to add more points from a different random order
    # Do multiple passes with different random seeds
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 30:
            break
        random.seed(seed)
        remaining2 = list(all_points - S)
        random.shuffle(remaining2)
        for pt in remaining2:
            if time.time() > deadline - 30:
                break
            S.add(pt)
            if is_set_valid(S):
                pass
            else:
                S.remove(pt)

    # Convert S to N (complement)
    S_point_ids = set()
    for x1, x2 in S:
        S_point_ids.add(point_id(x1, x2))

    all_point_ids = set(range(q * q))
    N_point_ids = all_point_ids - S_point_ids

    points = []
    for pid in sorted(N_point_ids):
        x, y = decode_point(pid)
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return all points (valid but score 0)
        for pid in range(q * q):
            x, y = decode_point(pid)
            xa, xb = x % p, x // p
            ya, yb = y % p, y // p
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
