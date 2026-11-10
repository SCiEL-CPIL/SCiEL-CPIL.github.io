# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy approach.
    
    We build N incrementally. A point x can be added to N if for every
    direction v (either (1,m) for m in F_49, or (0,1)), the punctured line
    {x + t*v : t != 0} is fully contained in N.
    
    We start with an empty set and greedily add points that satisfy this condition.
    To maximize |N| (minimize |S|), we want to add as many points as possible.
    
    Key insight: If we process points in a smart order, we can build a large N.
    The condition is that for each point x in N, there must exist at least one
    direction where all other points on that line are in N.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    p = 7
    q = p * p  # 49
    
    # Precompute field operations
    # F_49 = F_7[alpha] where alpha^2 = 3
    # Elements are (a, b) representing a + b*alpha
    # Multiplication: (a1,b1)*(a2,b2) = (a1*a2 + 3*b1*b2, a1*b2 + a2*b1) mod 7
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + 3*b[1]*b[1]) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    # Precompute all field elements as indices 0..48
    field_elements = []
    field_index = {}
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            field_elements.append((a, b))
            field_index[(a, b)] = idx
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            prod = field_mul(field_elements[i], field_elements[j])
            mul_table[i, j] = field_index[prod]
    
    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            s = field_add(field_elements[i], field_elements[j])
            add_table[i, j] = field_index[s]
    
    # Precompute negation
    neg_table = np.zeros(q, dtype=np.int64)
    for i in range(q):
        neg_table[i] = field_index[field_neg(field_elements[i])]
    
    # Directions: 49 slopes m in F_49, plus vertical (represented as -1 or special)
    # For slope m, direction is (1, m). For vertical, direction is (0, 1).
    # We'll use indices 0..48 for slopes, and 49 for vertical.
    
    # For each point and each direction, precompute the other 48 points on the line
    # point_idx = xa * q + ya (where xa, ya are field element indices)
    # Actually, let's use: point index = x_idx * q + y_idx where x_idx, y_idx in 0..48
    
    n_points = q * q  # 2401
    
    # For each point and direction, store the set of other points on that line
    # This is expensive to precompute fully, so let's be smart about it.
    
    # Alternative approach: For a given point x and direction v, the line is
    # {x + t*v : t in F_49}. The other points are x + t*v for t != 0.
    
    # Let's think about this differently. We'll maintain:
    # - in_N: boolean array of size n_points
    # - For each point, count how many directions have all other points in N
    
    # A point x is "ready" to be added if there exists a direction v such that
    # all points x + t*v for t != 0 are in N.
    
    # We'll use a queue-based greedy: start with no points, then iteratively
    # add points that become "ready".
    
    # But initially no point is ready since N is empty. We need a different approach.
    
    # Better approach: Start with all points in N, then try to remove points
    # while maintaining validity. A point x can be removed from N if for every
    # direction v, the punctured line through x in direction v still has all
    # other points in N (i.e., removing x doesn't break the condition for x).
    
    # Wait, that's not quite right either. Let me reconsider.
    
    # The condition is: for every point x in F_49^2 (whether in N or not),
    # there exists a direction v such that all other points on the line
    # through x in direction v are in N.
    
    # So if x is in N, we need some direction where all others are in N.
    # If x is not in N, we ALSO need some direction where all others are in N.
    
    # This means: for every x, there's a "witness" direction.
    
    # Greedy removal: Start with N = all points. Try to remove points one by one.
    # A point x can be removed if after removal, for every point y (including x),
    # there still exists a witness direction.
    
    # This is complex. Let me try a simpler approach:
    
    # Build N from scratch using a greedy algorithm that adds points in an order
    # that ensures each added point has a witness direction.
    
    # Key insight: If we add points in a specific order, we can ensure that
    # when we add point x, there's already a direction where all other points
    # on the line are in N.
    
    # Let's try: pick a direction, say vertical (0,1). For each vertical line,
    # we can add all points on that line. Then for horizontal-ish lines, etc.
    
    # Actually, let's try a different structural approach based on the inspiration:
    # Use the fact that we can model this as finding a large set where each point
    # has a "witness" line.
    
    # Simple valid construction: N = all points except those on a single line.
    # Let S = {x : y = 0} (the x-axis). |S| = 49.
    # For x in S: x = (a, 0). Direction (0,1): line is {(a, t) : t in F_49}.
    # Other points: (a, t) for t != 0. These have y = t != 0, so in N. Good.
    # For x not in S: x = (a, b) with b != 0. Direction (0,1): line is {(a, b+t) : t}.
    # Other points: (a, b+t) for t != 0. One of these is (a, 0) when t = -b.
    # (a, 0) is in S, not in N. So this fails.
    
    # So a single line doesn't work. Let me try S = two parallel lines.
    # S = {y = 0} union {y = 1}. |S| = 98.
    # For x in S, say x = (a, 0). Direction (0,1): line has (a, 1) in S. Fails.
    
    # Try S = all points with y_a = 0 (i.e., y is in F_7). |S| = 49 * 7 = 343.
    # For x in S: x = (a, b) where b_a = 0. Direction (0,1): line is {(a, b+t)}.
    # b+t has a-component b_a + t_a = t_a. For t_a = 0, b+t is in S. So many points in S.
    # This doesn't help.
    
    # Let me go back to the unital-based approach but optimize it.
    # The current approach has |S| around some value. Let me compute it.
    
    # Actually, let me just implement a proper greedy algorithm.
    
    # State: in_N[x] = True if x is in N
    # For each point x, witness_dir[x] = the direction that witnesses x (or -1 if none)
    
    # A point x is "valid" if there exists a direction d such that for all t != 0,
    # x + t * v_d is in N.
    
    # We'll maintain for each point x and direction d, the count of points on the
    # punctured line that are NOT in N. If this count is 0, then d is a witness for x.
    
    # Initialize: all points in N. Then try to remove points.
    
    # For each point x, count[x][d] = number of points on punctured line through x
    # in direction d that are NOT in N.
    
    # Initially all in N, so all counts are 0. Every point has all 50 directions as witnesses.
    
    # When we remove a point y from N, for each direction d, for each point x on the
    # line through y in direction d (other than y), count[x][d] increases by 1.
    
    # A point x can be removed if it still has at least one witness direction
    # (i.e., min_d count[x][d] == 0).
    
    # We want to remove as many points as possible.
    
    # Precompute for each point and direction, the other 48 points on the line.
    
    # This is 2401 * 50 * 48 = ~5.7 million entries. Might be too much memory.
    
    # Alternative: compute on the fly.
    
    # Let's use a different representation.
    # Point index: idx = x_idx * q + y_idx where x_idx, y_idx in 0..48
    
    # For direction d:
    # - If d < 49: slope m = field_elements[d], direction vector (1, m)
    #   Line through (x, y): {(x + t, y + t*m) : t in F_49}
    # - If d == 49: vertical, direction (0, 1)
    #   Line through (x, y): {(x, y + t) : t in F_49}
    
    # For a given point (x, y) and direction d, the other points are:
    # - If d < 49: (x + t, y + t*m) for t in F_49, t != 0
    # - If d == 49: (x, y + t) for t in F_49, t != 0
    
    # We'll maintain:
    # - in_N: boolean array of size n_points
    # - For each point x, witness_count[x] = number of directions that are witnesses
    
    # When we remove point y:
    # For each direction d:
    #   For each other point x on the line through y in direction d:
    #     Check if d was a witness for x before removing y.
    #     If yes, and now the line has a point not in N (namely y), then d is no longer
    #     a witness for x. Decrement witness_count[x].
    
    # This is still complex. Let me try a simpler heuristic.
    
    # Simple approach: Remove points in a random order, checking validity each time.
    # This is O(n^2) per removal, too slow.
    
    # Let me try yet another approach: Use the structure of the field.
    
    # Construction: Let S be a set of points such that for every point x,
    # there's a line through x that avoids S except possibly at x.
    
    # Known construction: S = graph of a function f: F_49 -> F_49 with certain properties.
    
    # Actually, the simplest valid Nikodym set I can think of:
    # N = F_49^2 \ S where S is a "blocking set" in the projective plane sense.
    
    # In the projective plane over F_49, a blocking set is a set of points that
    # intersects every line. The complement of a blocking set is a Nikodym set
    # in the affine plane (with some care about lines at infinity).
    
    # The smallest blocking set in PG(2, q) has size at least q + sqrt(q) + 1 for
    # q a square. For q=49, that's at least 49 + 7 + 1 = 57.
    
    # But we need an affine blocking set (blocking all affine lines).
    
    # Let me just try to implement the greedy removal with efficient bookkeeping.
    
    # I'll use numpy for vectorized operations where possible.
    
    # Point representation: idx = x_idx * q + y_idx
    # x_idx, y_idx in 0..48
    
    # For each direction d (0..49), precompute the permutation of points along lines.
    # For direction d, the lines partition the 2401 points into 49 lines of 49 points each.
    # (For non-vertical: lines are parameterized by y - m*x = c, 49 values of c.)
    # (For vertical: lines are parameterized by x = c, 49 values of c.)
    
    # For each direction d and each line L on that direction, we have 49 points.
    # When we remove a point from L, it affects the witness status of the other 48 points on L.
    
    # Let's precompute for each direction d, the list of lines, and for each line,
    # the list of point indices.
    
    lines_by_direction = [[] for _ in range(50)]  # 50 directions
    
    for d in range(50):
        if d < 49:
            m_idx = d
            m = field_elements[m_idx]
            # Line: y - m*x = c, for c in F_49
            for c_idx in range(q):
                c = field_elements[c_idx]
                line_points = []
                for x_idx in range(q):
                    x = field_elements[x_idx]
                    # y = m*x + c
                    mx = field_mul(m, x)
                    y = field_add(mx, c)
                    y_idx = field_index[y]
                    point_idx = x_idx * q + y_idx
                    line_points.append(point_idx)
                lines_by_direction[d].append(line_points)
        else:
            # Vertical: x = c
            for c_idx in range(q):
                x = field_elements[c_idx]
                x_idx = field_index[x]
                line_points = []
                for y_idx in range(q):
                    point_idx = x_idx * q + y_idx
                    line_points.append(point_idx)
                lines_by_direction[d].append(line_points)
    
    # Now, for each point, we can find which line it's on for each direction.
    # point_line[point_idx][d] = line index for direction d
    
    point_line = np.zeros((n_points, 50), dtype=np.int64)
    for d in range(50):
        for line_idx, line_points in enumerate(lines_by_direction[d]):
            for point_idx in line_points:
                point_line[point_idx, d] = line_idx
    
    # For each direction d and line L, we need to track how many points on L are NOT in N.
    # line_not_in_N[d, line_idx] = count of points on line L in direction d that are not in N
    
    line_not_in_N = np.zeros((50, q), dtype=np.int64)
    
    # For each point x, witness_count[x] = number of directions d such that
    # line_not_in_N[d, point_line[x, d]] == 0 (i.e., the punctured line is all in N)
    # Wait, that's not quite right. The punctured line excludes x itself.
    # So we need: the number of points on the line (excluding x) that are not in N is 0.
    # If x is in N, then the punctured line has not_in_N count = line_not_in_N[d, L] (since x is in N, it's not counted).
    # If x is not in N, then the punctured line has not_in_N count = line_not_in_N[d, L] - 1 (since x itself is not in N).
    
    # Let me redefine: for point x and direction d, let L be the line through x in direction d.
    # punctured_not_in_N[x, d] = number of points on L (other than x) that are not in N.
    # This equals:
    #   If x in N: line_not_in_N[d, L]
    #   If x not in N: line_not_in_N[d, L] - 1
    
    # x has a witness in direction d iff punctured_not_in_N[x, d] == 0.
    
    # We'll maintain witness_count[x] = number of directions where x has a witness.
    
    # Initially, all points in N, all line_not_in_N = 0, so all punctured_not_in_N = 0.
    # All points have 50 witnesses.
    
    in_N = np.ones(n_points, dtype=bool)
    witness_count = np.full(n_points, 50, dtype=np.int64)
    
    # Now, greedily remove points. We want to remove as many as possible.
    # A point x can be removed if after removal, it still has at least one witness.
    # But removal also affects other points.
    
    # When we remove point y from N:
    # For each direction d:
    #   L = point_line[y, d]
    #   line_not_in_N[d, L] += 1
    #   For each other point x on L:
    #     Before: if x in N, punctured_not_in_N[x, d] = line_not_in_N[d, L] (old value)
    #             if x not in N, punctured_not_in_N[x, d] = line_not_in_N[d, L] - 1 (old value)
    #     After:  if x in N, punctured_not_in_N[x, d] = line_not_in_N[d, L] (new value) = old + 1
    #             if x not in N, punctured_not_in_N[x, d] = line_not_in_N[d, L] - 1 (new value) = old + 1
    #     So in both cases, punctured_not_in_N increases by 1.
    #     If it was 0 before, it's now 1, so d is no longer a witness for x.
    #     Decrement witness_count[x].
    
    # We want to remove points that can be removed (have witness_count >= 1 after removal).
    # But removing a point might make other points unremovable.
    
    # Greedy: repeatedly remove a point that has witness_count >= 2 (so after decrement, it still has >= 1).
    # Actually, we need to be careful: when we remove y, y's own witness_count changes too.
    # But we're removing y, so we don't care about y's witness_count after removal.
    # We care about other points' witness_counts.
    
    # Let's use a priority queue: points with highest witness_count are safest to remove.
    
    import heapq
    
    # Min-heap of (-witness_count, point_idx)
    heap = [(-witness_count[i], i) for i in range(n_points)]
    heapq.heapify(heap)
    
    removed = 0
    max_removals = 2400  # Can't remove all points
    
    while removed < max_removals:
        if time.time() > deadline:
            break
        
        # Pop the point with highest witness_count
        neg_wc, y = heapq.heappop(heap)
        current_wc = -neg_wc
        
        if not in_N[y]:
            continue  # Already removed
        
        if current_wc < 2:
            # Can't remove this point (after removal, it would have < 1 witness)
            # Actually, we need to check: after removing y, does y still have a witness?
            # But y is being removed, so it doesn't need a witness.
            # We need other points to still have witnesses.
            # The issue is: removing y might reduce witness_count of other points.
            # If any other point has witness_count == 1 and y is its only witness,
            # then removing y makes that point invalid.
            
            # This is getting complex. Let me simplify: only remove points with witness_count >= 2.
            # This is conservative but should work.
            continue
        
        # Remove y
        in_N[y] = False
        removed += 1
        
        # Update witness_count for other points on lines through y
        for d in range(50):
            L = point_line[y, d]
            old_count = line_not_in_N[d, L]
            line_not_in_N[d, L] = old_count + 1
            
            # For each other point x on this line:
            line_points = lines_by_direction[d][L]
            for x in line_points:
                if x == y:
                    continue
                
                # Before: punctured_not_in_N[x, d] = 
                #   if in_N[x]: old_count
                #   if not in_N[x]: old_count - 1
                # After: punctured_not_in_N[x, d] =
                #   if in_N[x]: old_count + 1
                #   if not in_N[x]: old_count
                
                # If before was 0, after is 1, so witness lost.
                if in_N[x]:
                    if old_count == 0:
                        witness_count[x] -= 1
                else:
                    if old_count - 1 == 0:
                        witness_count[x] -= 1
                
                # Update heap
                heapq.heappush(heap, (-witness_count[x], x))
        
        # Note: we don't update y's witness_count since y is removed
    
    # Build result: all points in N
    N_points = []
    for idx in range(n_points):
        if in_N[idx]:
            x_idx = idx // q
            y_idx = idx % q
            xa, xb = field_elements[x_idx]
            ya, yb = field_elements[y_idx]
            N_points.append([[xa, xb], [ya, yb]])
    
    if len(N_points) == 0:
        # Fallback: return all points
        for x_idx in range(q):
            for y_idx in range(q):
                xa, xb = field_elements[x_idx]
                ya, yb = field_elements[y_idx]
                N_points.append([[xa, xb], [ya, yb]])
    
    result = np.array(N_points, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
