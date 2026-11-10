# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using greedy removal with efficient
    clean-line tracking.
    
    Strategy:
    1. Start with N = all 2401 points (S = empty).
    2. For each point, track how many "clean" lines pass through it.
       A line through point x with direction v is clean if all other points
       on that line are in N (not in S).
    3. A point p can be removed from N (added to S) iff for every point x != p,
       x still has at least one clean line after p is removed.
       Equivalently: no point x has its last clean line destroyed by removing p.
    4. Greedily remove the point that "costs" the least (fewest clean lines destroyed).
    
    Efficient tracking:
    - For each point x, maintain clean_count[x] = number of clean lines through x.
    - When removing point p from N:
      * For each direction v, let L be the line through p with direction v.
      * For each other point x on L: clean_count[x] -= 1 (line L is no longer clean for x)
      * If clean_count[x] becomes 0 for any x, we can't remove p.
    - To make this fast, we precompute for each point p, the list of (x, v) pairs
      where x is on the same line as p in direction v, and track which lines
      are currently clean.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3 in F_7[alpha]

    # Field operations
    idx_to_ab = np.zeros((q, 2), dtype=np.int64)
    ab_to_idx = np.full((p, p), -1, dtype=np.int64)
    for i in range(p):
        for j in range(p):
            idx = i * p + j
            idx_to_ab[idx] = [i, j]
            ab_to_idx[i, j] = idx

    def field_add(a, b):
        aa, ab = idx_to_ab[a]
        ba, bb = idx_to_ab[b]
        return ab_to_idx[(aa + ba) % p, (ab + bb) % p]

    def field_neg(a):
        aa, ab = idx_to_ab[a]
        return ab_to_idx[(-aa) % p, (-ab) % p]

    def field_mul(a, b):
        aa, ab = idx_to_ab[a]
        ba, bb = idx_to_ab[b]
        prod_a = (aa * ba + ab * bb * k) % p
        prod_b = (aa * bb + ab * ba) % p
        return ab_to_idx[prod_a, prod_b]

    # Point indexing: point (x, y) where x, y in F_49
    # Point index = x * q + y
    # x = xa * p + xb, y = ya * p + yb

    def point_to_idx(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)

    def idx_to_point(idx):
        y_idx = idx % q
        x_idx = idx // q
        ya = y_idx // p
        yb = y_idx % p
        xa = x_idx // p
        xb = x_idx % p
        return (xa, xb, ya, yb)

    total_points = q * q  # 2401

    # Directions: 50 total
    # 49 directions of type (1, m) for m in F_49 (m_idx = 0..48)
    # 1 direction of type (0, 1)
    # Direction index: 0..48 for (1, m_idx), 49 for (0, 1)

    num_directions = q + 1  # 50

    # For each direction d and each point x, compute the line points.
    # line_points[d][x] = list of point indices on the line through x with direction d
    # There are 50 * 2401 = 120050 lines, each with 49 points.

    # Actually, for each direction, the lines partition the space.
    # For direction (1, m): line through (x0, y0) = {(x0 + t, y0 + m*t) : t in F_49}
    # For direction (0, 1): line through (x0, y0) = {(x0, y0 + t) : t in F_49}

    # Precompute for each direction d and each point x, the set of other points on the line.
    # line_others[d][x] = list of point indices on the line through x with direction d, excluding x.

    # For efficiency, store as: for each direction d, for each point x, 
    # the 48 other points on the line through x in direction d.

    # Let's precompute line membership:
    # line_id[d][x] = the canonical line ID for the line through x in direction d
    # For direction (1, m): canonical representative is the point with x=0, i.e., (0, y0 - m*x0)
    # For direction (0, 1): canonical representative is (x0, 0)

    # Actually, let's just precompute for each (d, x) the list of other points.
    # This is 50 * 2401 * 48 entries.

    # For speed, precompute field multiplication: m * t for all m, t in F_49
    mul_table = np.zeros((q, q), dtype=np.int64)
    for a in range(q):
        for b in range(q):
            mul_table[a, b] = field_mul(a, b)

    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int64)
    for a in range(q):
        for b in range(q):
            add_table[a, b] = field_add(a, b)

    # Precompute negation
    neg_table = np.zeros(q, dtype=np.int64)
    for a in range(q):
        neg_table[a] = field_neg(a)

    # Precompute for each direction d and each point (x_idx, y_idx), 
    # the line points (excluding the point itself).
    # We'll store: line_others[d_idx][x_idx][y_idx] = array of 48 point indices

    # Actually, let's store it differently for cache efficiency.
    # For each direction d, for each point x, store the 48 other points.
    # Total: 50 * 2401 * 48 = 5,762,400 integers.

    # Let's compute line points on the fly but cache the structure.
    # For direction (1, m_idx):
    #   Line through (x0, y0): points are (x0 + t, y0 + m*t) for t in F_49
    #   t ranges over all 49 field elements.
    #   When t=0, we get (x0, y0) itself.
    #   Other 48 points: t != 0.

    # For direction (0, 1):
    #   Line through (x0, y0): points are (x0, y0 + t) for t in F_49
    #   Other 48 points: t != 0.

    # Precompute: for each direction d and each point, the list of other 48 points.
    # Store as a flat array: shape (num_directions, total_points, 48)
    # That's 50 * 2401 * 48 = 5,762,400 entries. Each is an int64. ~46 MB. OK.

    line_others = np.zeros((num_directions, total_points, q - 1), dtype=np.int64)

    for d in range(num_directions):
        if d < q:
            # Direction (1, m_idx=d)
            m_idx = d
            for x_idx in range(q):
                for y_idx in range(q):
                    point_idx = x_idx * q + y_idx
                    others = []
                    for t_idx in range(q):
                        if t_idx == 0:
                            continue
                        # new_x = x + t, new_y = y + m*t
                        nx = add_table[x_idx, t_idx]
                        mt = mul_table[m_idx, t_idx]
                        ny = add_table[y_idx, mt]
                        others.append(nx * q + ny)
                    line_others[d, point_idx, :] = others
        else:
            # Direction (0, 1)
            for x_idx in range(q):
                for y_idx in range(q):
                    point_idx = x_idx * q + y_idx
                    others = []
                    for t_idx in range(q):
                        if t_idx == 0:
                            continue
                        nx = x_idx
                        ny = add_table[y_idx, t_idx]
                        others.append(nx * q + ny)
                    line_others[d, point_idx, :] = others

    # Now, greedy removal.
    # State:
    #   in_N[x] = True if x is in N (not in S)
    #   clean_count[x] = number of clean lines through x
    #   A line through x in direction d is clean if all other points on that line are in N.
    #   Initially, all lines are clean, so clean_count[x] = 50 for all x.

    in_N = np.ones(total_points, dtype=bool)
    clean_count = np.full(total_points, num_directions, dtype=np.int64)

    # For each point p, we want to compute the "cost" of removing it:
    # cost[p] = number of (x, d) pairs where x != p, x is on the line through p in direction d,
    #           and the line through x in direction d would become unclean.
    # Actually, it's simpler: removing p makes the line through p in direction d unclean for
    # all other points on that line. So for each direction d, for each other point x on the
    # line through p in direction d, clean_count[x] decreases by 1.
    # The removal is valid iff no clean_count[x] drops below 1 (i.e., no point loses its last clean line).

    # To efficiently find good candidates, we can use a heuristic:
    # A point p is "safe" to remove if for all x on lines through p, clean_count[x] > 1
    # (so they still have at least one clean line after losing this one).
    # More precisely, we need clean_count[x] >= 2 for all affected x, OR clean_count[x] >= 1
    # and the line being lost is not their only clean line.

    # Let's use a priority queue. For each point, compute a "safety score":
    # safety[p] = min over all (d, x) where x is on line through p in direction d, x != p,
    #             of clean_count[x]
    # If safety[p] >= 2, then removing p is safe (all affected points still have >= 1 clean line).
    # If safety[p] == 1, removing p might be unsafe (some point might lose its last clean line).
    # If safety[p] == 0, removing p is definitely unsafe.

    # Actually, we need to be more careful. Let me just do a simple greedy:
    # 1. For each point p, check if removing it is valid.
    # 2. If valid, remove it.
    # 3. Update clean_count for all affected points.
    # 4. Repeat.

    # To speed up, we'll process points in a random order and use a heap of candidates.

    # Let's use a different approach: iterative greedy with random restarts.
    # But first, let's do a simple sequential greedy.

    # Precompute for each point p, the list of (d, x) pairs affected by removing p.
    # affected[p] = list of (d, x) where x is on the line through p in direction d, x != p.
    # This is 50 * 48 = 2400 entries per point. Total: 2401 * 2400 = 5,762,400.

    # Actually, we already have line_others[d][p] which gives the 48 other points.
    # So affected pairs for p: for d in 0..49, for x in line_others[d][p], the pair (d, x).

    # Let's do the greedy:
    # Use a heap keyed by some heuristic. For now, just try all points in order.

    # To make it faster, let's batch the checks.
    # For a point p to be removable:
    #   For all d in 0..49, for all x in line_others[d][p]:
    #     clean_count[x] >= 2  (so that after decrement, it's still >= 1)
    #   OR more precisely:
    #     After decrementing clean_count[x] by 1 for all affected (d, x),
    #     no clean_count[x] should be 0.

    # Let's just do it: try to remove each point, check validity, update if valid.

    # For speed, we'll use numpy vectorization where possible.

    # Let's precompute for each point p, the flat list of affected (x) indices.
    # affected_points[p] = flat array of all x that are on some line through p (x != p).
    # Note: the same x might appear multiple times (for different directions).

    # Actually, for the validity check, we need to check that min(clean_count[x]) >= 2
    # for all affected x (counting multiplicity doesn't matter for the min).
    # Wait, no. If x is affected by 2 different directions, clean_count[x] decreases by 2.
    # So we need: for all affected x, clean_count[x] - (number of times x is affected) >= 1.

    # Let's precompute for each p, the set of affected x and their multiplicity.

    # This is getting complex. Let's just do a simple O(2400) check per candidate.
    # With 2401 points, that's 2401 * 2400 = 5.76M operations per full pass.
    # We can do multiple passes.

    # Let's do iterative passes until no more points can be removed.

    S = set()  # points removed from N
    removed = np.zeros(total_points, dtype=bool)

    # For efficiency, let's precompute for each p, the list of affected x indices (with multiplicity).
    # affected_list[p] = list of x indices (length 2400, with duplicates)

    # Actually, let's just compute on the fly.

    max_passes = 100
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        any_removed = False
        # Try to remove points in a shuffled order
        order = np.random.permutation(total_points)
        for p in order:
            if time.time() > deadline:
                break
            if removed[p]:
                continue
            # Check if removing p is valid
            valid = True
            # For each direction d, for each x in line_others[d][p]:
            #   if clean_count[x] <= 1: valid = False; break
            # But we need to account for the fact that multiple directions might affect the same x.
            # Let's count how many times each x is affected.
            
            # Simple check: for all affected (d, x), clean_count[x] >= 2
            # This is sufficient but not necessary. If clean_count[x] == 1 and x is affected
            # by 2 directions, it would go to -1, which is bad. But if clean_count[x] == 1
            # and x is affected by 1 direction, it goes to 0, which is also bad.
            # So the condition "clean_count[x] >= 2 for all affected x" is sufficient.
            # It might miss some valid removals, but it's safe.
            
            for d in range(num_directions):
                for x in line_others[d, p]:
                    if clean_count[x] < 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Remove p from N
                removed[p] = True
                in_N[p] = False
                any_removed = True
                # Update clean_count for all affected points
                for d in range(num_directions):
                    for x in line_others[d, p]:
                        clean_count[x] -= 1
        
        if not any_removed:
            break

    # Build the result: N = points not removed
    points = []
    for idx in range(total_points):
        if not removed[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
