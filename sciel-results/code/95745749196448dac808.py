# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np

_START_TIME = time.time()
_DEADLINE = 630.0 - 45.0  # safety margin

def _remaining():
    return _DEADLINE - time.time()


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 via greedy local search on the
    complement. A set N is a Nikodym set iff for every point x in F_49^2 there
    exists a direction v=(1,m) (m in F_49) or v=(0,1) such that the punctured
    line {x + t*v : t in F_49, t != 0} is contained in N. Equivalently, for the
    complement C = F_49^2 \ N, every point x must have at least one direction m
    for which the line through x in direction m contains no point of C other
    than possibly x itself.

    We maximize |C| (the metric) via a greedy algorithm:
      - coverage[x] = number of directions m such that line(x,m) \ {x} has no C point
      - A point is "covered" iff coverage[x] > 0
      - Adding p to C may uncover points that had p on their only covering line
      - We use a priority queue to always add the "cheapest" point (fewest
        newly-uncovered points), restarting with the zero-cost point when available
    """
    p = 7
    q = p * p  # 49

    # Build field F_49 = F_7[alpha], alpha^2 = 3
    # Element encoded as int 0..48: a + b*alpha -> a + b*p
    # Multiplication: (a,b)*(c,d) = (ac + bd*3, ad + bc)
    mult_table = np.zeros((q, q), dtype=np.int32)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    e = (a * c + b * d * 3) % p
                    f = (a * d + b * c) % p
                    mult_table[a + b * p, c + d * p] = e + f * p
    # Precompute inverse (for direction m, need 1/m)
    inv_table = np.zeros(q, dtype=np.int32)
    for m in range(1, q):
        # find inv: m * inv = 1
        for x in range(1, q):
            if mult_table[m, x] == 1:
                inv_table[m] = x
                break
    inv_table[0] = 0  # unused

    # Directions: 49 total. Direction d encodes (dx, dy):
    #   d = 0: vertical, (0, 1)
    #   d = 1..49: (1, m) where m = d-1 in F_49
    # For point x, the other points on line(x, d) are x + t*v for t in F_49\{0}
    # Precompute for each point x and direction d, the list of other points on the line
    # Actually, we need: for each point x, for each direction d, the set of other points
    # on that line. But storing all is 2401*49*48 = 5.6M entries - too much.
    # Instead: for each point x and direction d, we need to know which points y (y != x)
    # are on the same line. We can compute this on the fly or precompute a lookup.
    #
    # Better approach: precompute for each point x, a list of (direction, other_points)
    # But that's the same memory. Let's think differently.
    #
    # For the greedy algorithm, when we add point p to C, we need to find all points x
    # such that p is on the line through x in some direction d, AND d was the only
    # direction keeping x covered. This is complex.
    #
    # Simpler: precompute for each point x and direction d, the 48 other points on the line.
    # Store as a flat array: line_points[x * 49 + d] = array of 48 point indices
    # Total: 2401 * 49 * 48 = 5,631,432 entries. As int32, ~22 MB. Acceptable.
    line_points = np.zeros((q * q, 48), dtype=np.int32)
    for x in range(q * q):
        # x = xa + xb*p + (ya + yb*p)*q
        xa = x % p
        xb = (x // p) % p
        ya = (x // q) % p
        yb = (x // (q * p)) % p
        x_enc = xa + xb * p  # F_49 encoding of x-coordinate
        y_enc = ya + yb * p  # F_49 encoding of y-coordinate
        for d in range(49):
            if d == 0:
                # vertical: (0, 1), points are (x, y + t) for t in F_49
                for ti in range(1, q):
                    new_y = mult_table[y_enc, ti] + y_enc  # wait, addition is component-wise
                    # Actually addition: (a,b)+(c,d) = (a+c, b+d)
                    # y_enc = ya + yb*p, t_enc = ta + tb*p
                    # sum = (ya+ta) + (yb+tb)*p
                    ta = ti % p
                    tb = (ti // p) % p
                    nys = (ya + ta) % p
                    nby = (yb + tb) % p
                    new_y_enc = nys + nby * p
                    # point index: x_enc + new_y_enc * q
                    pt_idx = x_enc + new_y_enc * q
                    # find position ti in the 48
                    # t goes 1..48 (ti from 1 to q-1)
                    line_points[x * 49 + d, ti - 1] = pt_idx
            else:
                # direction (1, m) where m = d-1
                m_enc = d - 1
                for ti in range(1, q):
                    # x + t*(1, m) = (x + t, y + t*m)
                    # t_enc = ta + tb*p
                    ta = ti % p
                    tb = (ti // p) % p
                    t_enc = ta + tb * p
                    # new_x = x_enc + t_enc (component-wise addition)
                    nxa = (xa + ta) % p
                    nxb = (xb + tb) % p
                    new_x_enc = nxa + nxb * p
                    # new_y = y_enc + t_enc * m_enc
                    tm = mult_table[t_enc, m_enc]
                    tma = tm % p
                    tmb = (tm // p) % p
                    nya = (ya + tma) % p
                    nyb = (yb + tmb) % p
                    new_y_enc = nya + nyb * p
                    pt_idx = new_x_enc + new_y_enc * q
                    line_points[x * 49 + d, ti - 1] = pt_idx

    # Now run greedy local search
    # C = complement set, initially empty
    in_C = np.zeros(q * q, dtype=bool)
    # coverage[x] = number of directions d such that no point on line(x,d)\{x} is in C
    coverage = np.full(q * q, 49, dtype=np.int32)  # initially all 49 directions work

    # For efficiency, track for each point x and direction d, how many C-points are on the line
    # line_c_count[x, d] = number of C-points on line(x,d) excluding x
    # But that's 2401*49 = 117649 entries, fine
    line_c_count = np.zeros((q * q, 49), dtype=np.int32)

    # Priority queue: (cost, point_index) where cost = number of points that would be uncovered
    # when adding this point to C. Initially all points have cost = number of points x
    # for which this point is the only C-point blocker... actually initially C is empty,
    # so adding any point p: for each x, if p is on line(x,d), then line_c_count[x,d] becomes 1,
    # so coverage[x] decreases by 1. x becomes uncovered iff coverage[x] was 1.
    # cost(p) = number of x such that p is on some line(x,d) AND coverage[x] == 1 (before adding p)
    # But coverage is dynamic. Let's use a different approach.

    # Simpler greedy: repeatedly find a point p that can be added to C without uncovering any point.
    # Such a p exists iff there's a p such that for all x with p on line(x,d), coverage[x] >= 2.
    # We can check this by: for candidate p, compute the set of (x, d) pairs where p is on line(x,d),
    # and check if any has coverage[x] == 1.
    #
    # Even simpler: use a priority queue with cost = max(0, number of x that would become uncovered).
    # When we add p, update coverage and line_c_count, then recompute costs for affected points.

    # Let's use a simpler iterative approach:
    # 1. Compute cost for all points: cost[p] = number of x such that p is on line(x,d) for some d
    #    AND coverage[x] == 1 (i.e., x would become uncovered if p is added)
    # 2. Pick the point with minimum cost (preferably 0)
    # 3. Add it to C, update coverage and line_c_count
    # 4. Update costs for points whose cost changed
    # 5. Repeat until no zero-cost point or time runs out

    # For step 1, we need for each point p, the set of (x, d) such that p is on line(x,d)\{x}.
    # Precompute: for each p, list of (x, d) pairs where p is on line(x,d) and x != p.
    # This is the "reverse" of line_points. For each (x, d, i), p = line_points[x*49+d, i],
    # add (x, d) to reverse_list[p].
    # Total entries: 2401 * 49 * 48 = 5.6M, same as before.
    # Store as: for each p, a list of (x, d) pairs.
    # Memory: 5.6M * 8 bytes (two int32) = 45 MB. OK.

    # Actually, let's just compute cost on the fly for the top candidates.
    # Use a max-heap on "safety" = min coverage[x] over x that would be affected, or just count.

    # Let me use a simpler approach:
    # Maintain a list of "candidate" points that might have zero cost.
    # When C changes, only points p that are on lines of newly-uncovered points need cost updates.

    # Given time constraints, let's do a simpler greedy:
    # - Start with C = empty
    # - In each iteration, find a point p that can be added without uncovering any point
    # - To find such p: for each direction d, consider lines in that direction.
    #   A point p on line L (in direction d) can be added if for all x on L, coverage[x] >= 2
    #   (since adding p only affects coverage of x on the same line in direction d, and other directions)
    #   Wait, adding p affects coverage of x for ALL directions d' where p is on line(x, d').
    #   So p can be added safely iff for all x and all d' with p on line(x,d'), coverage[x] >= 2.
    #
    # This is still complex. Let me just do a simple random greedy with validation:
    # - Try to add random points to C
    # - After each addition, check if all points are still covered
    # - If not, revert
    # - This is slow but correct. With 2401 points, checking coverage is O(2401*49*48) = 5.6M per check.
    # Too slow for many iterations.

    # Let me use the coverage-based approach with a priority queue.
    # Key insight: when we add point p to C, the only points whose coverage changes are
    # those x for which p lies on line(x, d) for some d. For each such (x, d), coverage[x] decreases by 1.
    # x becomes uncovered iff coverage[x] reaches 0.
    # So cost(p) = number of x such that p is on line(x,d) for some d AND coverage[x] == 1.
    # (Because if coverage[x] == 1 and p is on one of x's lines, adding p makes coverage[x] == 0.)

    # Precompute reverse: for each p, list of (x, d) where p is on line(x,d)\{x}
    # Use a dictionary or list of lists
    reverse_list = [[] for _ in range(q * q)]
    for x in range(q * q):
        for d in range(49):
            base = x * 49 + d
            for i in range(48):
                p = line_points[base, i]
                reverse_list[p].append((x, d))

    # Now run greedy
    # cost[p] = number of (x, d) in reverse_list[p] such that coverage[x] == 1
    # But coverage changes as we add points. We need to maintain cost dynamically.
    # When we add point p0 to C:
    #   For each (x, d) in reverse_list[p0]:
    #     line_c_count[x, d] += 1
    #     if line_c_count[x, d] == 1:  # this was the first C-point on this line for x
    #       coverage[x] -= 1
    #       # Now x may be uncovered if coverage[x] == 0
    #       # Also, for all points p1 in reverse_list[x]... no wait.
    #       # The cost of other points p1 changes if p1 is on line(x, d') for some d'
    #       # and coverage[x] just changed.
    #       # Specifically, if coverage[x] went from 1 to 0, then x is now uncovered -> INVALID.
    #       # If coverage[x] went from 2 to 1, then x is still covered but now "fragile".
    #       # For any point p1 that is on line(x, d') for some d', the cost of p1 increases by 1
    #       # (because if p1 is added, it would uncover x).
    #
    # This is getting complex. Let me simplify:
    # - Only add a point p if cost[p] == 0 (no points would be uncovered)
    # - After adding p, update coverage and line_c_count
    # - Recompute cost for all points (O(2401 * avg_reverse_list_size))
    #   avg_reverse_list_size = 49*48/49 = 48, so 2401*48 = 115k operations per recompute.
    #   With ~1000 iterations, that's 115M operations. Might be too slow in Python.
    #
    # Let me optimize: only recompute cost for points whose cost might have changed.
    # When coverage[x] changes, only points p that are on line(x, d') for some d' have their cost change.
    # Those are the points in reverse_list[x]... no, reverse_list[x] gives (y, d) where x is on line(y,d).
    # I need the forward: for point x, which points p are on lines through x? That's all points
    # except x itself, but only those on the 49 lines through x.
    # Actually, for a given x, the points on lines through x (excluding x) are exactly the
    # line_points[x*49+d, :] for d in 0..48, which is 49*48 = 2352 points.
    # So when coverage[x] changes, the cost of those 2352 points may change.
    #
    # Let me just do the simple approach with numpy vectorization where possible.

    # Initialize cost
    cost = np.zeros(q * q, dtype=np.int32)
    # For each p, cost[p] = sum over (x, d) in reverse_list[p] of [coverage[x] == 1]
    # But initially coverage[x] = 49 for all x, so cost[p] = 0 for all p.

    # Add points greedily
    # Use a set of "available" points with cost 0
    available = set(range(q * q))
    # Initially all points have cost 0, so all are available
    # But we should add them in some order. Let's just pick arbitrarily.

    added_count = 0
    # To make it deterministic and fast, process in a fixed order but skip if cost > 0
    # Actually, let's use a priority queue with cost
    # Since initially all costs are 0, we can just iterate through points and add those with cost 0

    # Simpler: iterate through points in order, try to add each if cost is 0
    # After adding, update coverage and recompute costs for affected points
    # This is O(n * update_cost) where update_cost is ~2352 per added point
    # Total: 2401 * 2352 = 5.6M operations. Fast!

    for p_idx in range(q * q):
        if time.time() > _DEADLINE:
            break
        if p_idx in in_C:
            continue
        # Check if cost[p_idx] == 0
        # cost[p_idx] = number of (x, d) in reverse_list[p_idx] with coverage[x] == 1
        c = 0
        for (x, d) in reverse_list[p_idx]:
            if coverage[x] == 1:
                c += 1
                break  # early exit
        if c > 0:
            continue
        # Add p_idx to C
        in_C[p_idx] = True
        added_count += 1
        # Update coverage and line_c_count
        for (x, d) in reverse_list[p_idx]:
            line_c_count[x, d] += 1
            if line_c_count[x, d] == 1:
                coverage[x] -= 1
                # If coverage[x] == 0, we have an invalid state! But we checked cost==0,
                # so this shouldn't happen. Let's assert.
                if coverage[x] == 0:
                    # This shouldn't happen if cost was correctly 0
                    # Revert and skip
                    in_C[p_idx] = False
                    added_count -= 1
                    # Undo the updates
                    for (x2, d2) in reverse_list[p_idx]:
                        line_c_count[x2, d2] -= 1
                        if line_c_count[x2, d2] == 0:
                            coverage[x2] += 1
                    break
        else:
            # Successfully added, continue
            continue
        # If we broke out of the for-else, the addition failed
        continue

    # Build the Nikodym set N = all points not in C
    N = []
    for pt in range(q * q):
        if not in_C[pt]:
            xa = pt % p
            xb = (pt // p) % p
            ya = (pt // q) % p
            yb = (pt // (q * p)) % p
            N.append([[xa, xb], [ya, yb]])

    return np.array(N, dtype=np.int64)


# EVOLVE-BLOCK-END
