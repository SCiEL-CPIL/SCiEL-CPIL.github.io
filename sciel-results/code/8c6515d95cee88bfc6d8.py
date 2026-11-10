# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy independent set approach on the line graph.
    
    The Nikodym condition requires that for every point x in F_49^2, there exists a line L through x
    (direction (1,m) or (0,1)) such that L \ {x} is disjoint from S = F_49^2 \ N.
    
    Equivalently, S must be an independent set in the graph where vertices are points of F_49^2
    and edges connect pairs of points that share a line with some specific slope.
    
    We use a greedy algorithm to build S, always picking the point with the highest safety margin
    (minimum number of neighbors already in S among unselected points).
    """
    p_prime = 7
    q = 49
    N = q * q  # 2401 points
    
    # Index mapping: index = ((xa * 7 + xb) * 7 + ya) * 7 + yb
    def idx_to_coords(i):
        yb = i % 7
        ya = (i // 7) % 7
        xb = (i // 49) % 7
        xa = (i // 343) % 7
        return (xa, xb, ya, yb)

    def coords_to_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    # Precompute all 2401 points
    points = []
    for i in range(N):
        xa, xb, ya, yb = idx_to_coords(i)
        points.append((xa, xb, ya, yb))

    # Precompute lines: for each slope m in F_49 and vertical, generate all lines
    # A line with slope m through point (x0, y0) contains points (x0 + t, y0 + m*t) for t in F_49
    # Vertical lines: x = x0, all y

    # We need to track, for each point, how many lines through it contain another point in S
    # Actually, let's think differently.
    
    # The condition is: for every x, there exists a line L through x such that L \ {x} is disjoint from S.
    # This means: for every x, not all 50 lines through x intersect S (in a point other than x).
    
    # Let's define: for point x, blocked_lines[x] = number of lines through x that contain at least one point of S (other than x).
    # x is satisfied iff blocked_lines[x] < 50.
    
    # When we add point p to S:
    # - For every other point x, the line through x and p now contains p (in S).
    #   So if this line previously didn't contain any point of S, blocked_lines[x] increases by 1.
    #   But we need to know if the line already had a point in S.
    
    # Let's maintain: for each line, how many points of S it contains.
    # And for each point x, the number of lines through x that have >= 1 point in S (other than x).
    
    # Lines: there are 50 * 49 = 2450 lines total (49 slopes * 49 intercepts + 49 vertical).
    # Actually: for each slope m (49 values), there are 49 parallel lines. Plus 49 vertical lines. Total 2450.
    
    # Let's index lines:
    # Lines 0..2448: slope m in 0..48, intercept c in 0..48. Line: y = m*x + c
    # Lines 2449..2497: vertical, x = c for c in 0..48.
    
    # Wait, we're working in F_49, so m and c are elements of F_49, represented as pairs (a, b).
    # Let me re-index properly.
    
    # Element of F_49: (a, b) representing a + b*alpha, a,b in 0..6.
    # Index of element: a * 7 + b, so 0..48.
    
    # Line with slope m_idx (0..48) and intercept c_idx (0..48):
    #   Contains points (x_idx, y_idx) where y_idx = m_idx * x_idx + c_idx in F_49.
    
    # Vertical line with x_idx = c_idx:
    #   Contains points (c_idx, y_idx) for all y_idx.
    
    # Total lines: 49 * 49 + 49 = 2450.
    
    # Precompute: for each point, which 50 lines pass through it.
    # point_lines[point_idx] = list of 50 line indices.
    
    # Also: for each line, the list of 49 points on it.
    # line_points[line_idx] = list of 49 point indices.
    
    # And: line_count[line_idx] = number of points in S on this line.
    
    # For each point x, we track:
    #   satisfied_lines[x] = number of lines through x that have line_count == 0 (disjoint from S)
    #   OR, equivalently, blocked_lines[x] = 50 - satisfied_lines[x]
    #   x is valid iff satisfied_lines[x] > 0.
    
    # When we add point p to S:
    #   For each of the 50 lines L through p:
    #     line_count[L] increases by 1.
    #     For each other point x on L:
    #       If line_count[L] was 0 before (now 1), then L becomes blocked for x.
    #         So satisfied_lines[x] decreases by 1.
    #       If line_count[L] was > 0 before, L was already blocked for x, no change.
    
    # Let's precompute the line structure.
    
    # First, build a mapping from (slope_idx, intercept_idx) to line_idx for non-vertical lines.
    # And from x_idx to line_idx for vertical lines.
    
    # Let's just directly build line_points and point_lines.
    
    # line_points: array of shape (2450, 49) of point indices
    # point_lines: array of shape (2401, 50) of line indices
    
    # We need field arithmetic in F_49.
    # F_49 = F_7[alpha] / (alpha^2 - 3)
    # Elements are (a, b) = a + b*alpha.
    # Addition: (a1+b1*alpha) + (a2+b2*alpha) = (a1+a2) + (b1+b2)*alpha
    # Multiplication: (a1+b1*alpha)(a2+b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
    #                = a1*a2 + 3*b1*b2 + (a1*b2 + a2*b1)*alpha
    #                = (a1*a2 + 3*b1*b2) + (a1*b2 + a2*b1)*alpha
    
    def field_add(idx1, idx2):
        a1, b1 = divmod(idx1, 7)
        a2, b2 = divmod(idx2, 7)
        a = (a1 + a2) % 7
        b = (b1 + b2) % 7
        return a * 7 + b
    
    def field_sub(idx1, idx2):
        a1, b1 = divmod(idx1, 7)
        a2, b2 = divmod(idx2, 7)
        a = (a1 - a2) % 7
        b = (b1 - b2) % 7
        return a * 7 + b
    
    def field_mul(idx1, idx2):
        a1, b1 = divmod(idx1, 7)
        a2, b2 = divmod(idx2, 7)
        a = (a1 * a2 + 3 * b1 * b2) % 7
        b = (a1 * b2 + a2 * b1) % 7
        return a * 7 + b
    
    # Precompute field multiplication table
    mul_table = np.zeros((49, 49), dtype=np.int32)
    for i in range(49):
        for j in range(49):
            mul_table[i, j] = field_mul(i, j)
    
    # Precompute field addition table
    add_table = np.zeros((49, 49), dtype=np.int32)
    for i in range(49):
        for j in range(49):
            add_table[i, j] = field_add(i, j)
    
    # Now build line_points and point_lines.
    # Line index for slope m, intercept c: m * 49 + c
    # Line index for vertical line at x = c: 49 * 49 + c = 2401 + c
    
    line_points = np.zeros((2450, 49), dtype=np.int32)
    point_lines = np.zeros((2401, 50), dtype=np.int32)
    
    # Fill in non-vertical lines
    for m in range(49):
        for c in range(49):
            line_idx = m * 49 + c
            for x_idx in range(49):
                y_idx = field_add(mul_table[m, x_idx], c)
                # Point is (x_idx, y_idx)
                point_idx = (x_idx * 7 + y_idx) * 49  # Wait, this isn't right.
                # Point index: ((xa * 7 + xb) * 7 + ya) * 7 + yb
                # x_idx represents the field element for x, which is (xa, xb) -> x_idx = xa * 7 + xb
                # y_idx represents the field element for y, which is (ya, yb) -> y_idx = ya * 7 + yb
                # So point_idx = (x_idx * 7 + y_idx) * 49? No.
                # x_idx is already xa*7+xb. y_idx is ya*7+yb.
                # point_idx = ((xa*7+xb) * 7 + ya) * 7 + yb = (x_idx * 7 + ya) * 7 + yb
                # But we don't have ya and yb separately from y_idx.
                # y_idx = ya * 7 + yb.
                # So point_idx = (x_idx * 7 + ya) * 7 + yb = x_idx * 49 + ya * 7 + yb = x_idx * 49 + y_idx.
                # Yes! point_idx = x_idx * 49 + y_idx.
                
                point_idx = x_idx * 49 + y_idx
                line_points[line_idx, x_idx] = point_idx
                # Also record which lines pass through this point
                # We'll fill point_lines later by iterating over all lines
    
    # Fill in vertical lines
    for c in range(49):
        line_idx = 2401 + c
        for y_idx in range(49):
            point_idx = c * 49 + y_idx
            line_points[line_idx, y_idx] = point_idx
    
    # Now fill point_lines by iterating over all lines and points on them
    for line_idx in range(2450):
        for pos in range(49):
            point_idx = line_points[line_idx, pos]
            # Find where to put this line in point_lines[point_idx]
            # We need to know which of the 50 slots this line occupies.
            # For non-vertical lines (line_idx < 2401): slot = line_idx // 49 = m
            # For vertical lines (line_idx >= 2401): slot = 49
            if line_idx < 2401:
                slot = line_idx // 49
            else:
                slot = 49
            point_lines[point_idx, slot] = line_idx
    
    # Now implement the greedy algorithm.
    # We want to maximize |S|.
    # S is the excluded set. N = F_49^2 \ S is the Nikodym set we return.
    
    # State:
    #   in_S: boolean array of size 2401
    #   line_count: integer array of size 2450, number of points in S on each line
    #   satisfied_lines: integer array of size 2401, number of lines through x disjoint from S
    
    in_S = np.zeros(2401, dtype=bool)
    line_count = np.zeros(2450, dtype=np.int32)
    satisfied_lines = np.full(2401, 50, dtype=np.int32)
    
    # Greedy: repeatedly pick the point with the highest safety margin.
    # Safety margin for point p: how many lines through p are still disjoint from S.
    # But we also want to avoid picking points that would block too many other points.
    
    # Actually, the standard greedy for independent set: pick the vertex with minimum degree
    # in the remaining graph. Here, the "degree" of point p is the number of other points
    # that would be affected if we add p to S.
    
    # Let's use a different heuristic: pick the point p that minimizes the number of
    # other points x for which the line xp would become blocked (i.e., line xp has count 0).
    
    # For point p, the cost of adding it to S is:
    #   For each of the 50 lines L through p:
    #     If line_count[L] == 0, then adding p makes line_count[L] = 1,
    #     and for each of the other 48 points x on L, satisfied_lines[x] decreases by 1.
    #     So the cost is 48 * (number of lines through p with line_count == 0).
    
    # We want to minimize this cost.
    
    # Let's precompute for each point p, the list of 50 lines through it.
    # That's point_lines[p].
    
    # We'll use a priority queue or just scan all points each iteration.
    # Scanning all 2401 points each iteration, for up to 2401 iterations, is 2401^2 = 5.7M,
    # and each scan is O(50) to compute the cost. Total 5.7M * 50 = 285M operations.
    # This might be slow in Python. Let's try to optimize.
    
    # Alternative: use a simpler greedy. Just pick points in a fixed order,
    # and only add if it doesn't make any point unsatisfied.
    
    # Let's try the simple approach first with a time limit.
    
    start_time = time.time()
    deadline = start_time + 550  # Leave 80s for safety
    
    # Try multiple random orderings and keep the best S.
    rng = np.random.RandomState(12345)
    
    best_S_size = 0
    best_in_S = None
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        in_S_trial = np.zeros(2401, dtype=bool)
        line_count_trial = np.zeros(2450, dtype=np.int32)
        satisfied_lines_trial = np.full(2401, 50, dtype=np.int32)
        
        candidates = np.arange(2401)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if time.time() > deadline:
                break
            
            # Compute cost of adding p_idx to S
            # For each of the 50 lines through p_idx, check if line_count == 0
            # Cost = sum over lines L through p_idx: if line_count[L] == 0, then 48 else 0
            
            lines_through_p = point_lines[p_idx]
            # lines_through_p is an array of 50 line indices
            
            # Check if adding p_idx would make any point unsatisfied
            # We need to check: for each line L through p_idx with line_count[L] == 0,
            #   for each other point x on L, satisfied_lines[x] - 1 >= 0
            #   i.e., satisfied_lines[x] > 0
            
            can_add = True
            for L in lines_through_p:
                if line_count_trial[L] == 0:
                    # This line will become blocked
                    # Check all other points on this line
                    for pos in range(49):
                        x = line_points[L, pos]
                        if x == p_idx:
                            continue
                        if satisfied_lines_trial[x] <= 1:
                            can_add = False
                            break
                    if not can_add:
                        break
            
            if can_add:
                # Add p_idx to S
                in_S_trial[p_idx] = True
                for L in lines_through_p:
                    if line_count_trial[L] == 0:
                        # This line becomes blocked for all other points on it
                        for pos in range(49):
                            x = line_points[L, pos]
                            if x != p_idx:
                                satisfied_lines_trial[x] -= 1
                    line_count_trial[L] += 1
                # Note: satisfied_lines for p_idx itself doesn't change
                # (lines through p_idx that were disjoint remain disjoint in the sense
                #  that L \ {p_idx} is still disjoint from S)
        
        S_size = np.sum(in_S_trial)
        if S_size > best_S_size:
            best_S_size = S_size
            best_in_S = in_S_trial.copy()
    
    # Use the best S found
    in_S = best_in_S if best_in_S is not None else np.zeros(2401, dtype=bool)
    
    # Construct N = F_49^2 \ S
    result_list = []
    for i in range(2401):
        if not in_S[i]:
            xa, xb, ya, yb = idx_to_coords(i)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
    
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
    
    result = np.array(result_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
