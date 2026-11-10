# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy maximization algorithm.
    We want to maximize the size of S = F_49^2 \ N.
    N is valid if for every x, there is a line L through x such that L \ {x} is subset of N.
    This is equivalent to: for every x, there is a line L through x such that L contains no points of S \ {x}.
    """
    p = 7
    q = p * p
    k = 3
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600
    
    # Point representation: index = x_a * 49 + x_b * 7 + y_a * 1 + y_b ? 
    # Let's use index = (x_a * 7 + x_b) * 49 + (y_a * 7 + y_b)
    # Or simply map (xa, xb, ya, yb) to 0..2400.
    # Let's stick to the task's format [[xa, xb], [ya, yb]] for output.
    
    # Precompute field elements
    # We need to define lines.
    # Lines are of form y = m*x + c or x = c.
    # In F_49, m, c in F_49.
    # There are 49 slopes m + 1 vertical = 50 directions.
    # For each direction, there are 49 parallel lines.
    # Total 2450 lines.
    
    # To check validity efficiently, we need to know for each point x, which lines pass through it.
    # And for each line, which points are on it.
    
    # Let's represent points by their 4-tuple (xa, xb, ya, yb).
    # Index: idx = xa * 343 + xb * 49 + ya * 7 + yb.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append((xa, xb, ya, yb))
                    
    # Map point tuple to index
    point_to_idx = {}
    for i, pt in enumerate(points):
        point_to_idx[pt] = i
        
    # Precompute lines
    # A line is defined by a direction and an offset.
    # Directions:
    # 1. Vertical: x = c. c in F_49.
    # 2. Slope m: y = m*x + c. m in F_49, c in F_49.
    
    # We need to add points in F_49.
    # Let's implement field arithmetic helpers for F_49 = F_7[alpha]/(alpha^2-3).
    # Element is (a, b) representing a + b*alpha.
    # Add: (a1+b1, a2+b2) mod 7
    # Mul: (a1*a2 + 3*b1*b2, a1*b2 + a2*b1) mod 7
    
    def f_add(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
        
    def f_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
        
    def f_sub(p1, p2):
        return ((p1[0] - p2[0]) % p, (p1[1] - p2[1]) % p)
        
    # Generate all field elements
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
            
    # Precompute lines as lists of point indices
    # lines[i] is a list of point indices.
    # Also need line_through_point[x][l] to know which line index l corresponds to which line through x.
    # Actually, for the greedy check, when we add y to S:
    # We iterate over all 50 lines L passing through y.
    # For each L, we iterate over all x in L \ {y}.
    # We need to know if adding y blocks the last available line for x.
    # This means we need to track for each x, how many lines through x are "blocked" by S.
    # A line L through x is "blocked" if L contains any point z in S, z != x.
    
    # Let's precompute:
    # 1. For each point x, the list of line indices passing through it. (50 lines)
    # 2. For each line L, the list of point indices on it. (49 points)
    
    # Construct lines
    lines = []
    point_lines = [[] for _ in range(q*q)] # 2401 points
    
    # Vertical lines: x = c
    # x is a field element.
    # Point is (x, y).
    # In our tuple representation (xa, xb, ya, yb), x is (xa, xb).
    for c in field_elements:
        line_pts = []
        for y in field_elements:
            # Point (c, y)
            pt = (c[0], c[1], y[0], y[1])
            idx = point_to_idx[pt]
            line_pts.append(idx)
        line_idx = len(lines)
        lines.append(line_pts)
        for pt_idx in line_pts:
            point_lines[pt_idx].append(line_idx)
            
    # Sloped lines: y = m*x + c
    for m in field_elements:
        for c in field_elements:
            line_pts = []
            for x in field_elements:
                # y = m*x + c
                mx = f_mul(m, x)
                y = f_add(mx, c)
                pt = (x[0], x[1], y[0], y[1])
                idx = point_to_idx[pt]
                line_pts.append(idx)
            line_idx = len(lines)
            lines.append(line_pts)
            for pt_idx in line_pts:
                point_lines[pt_idx].append(line_idx)
                
    # Now, greedy search.
    # S is a set of points.
    # We want to add points to S one by one.
    # State: 
    #   in_S: boolean array of size 2401.
    #   blocked_count[x][l]: number of points in S \ {x} on line l (where l is a line through x).
    #   Actually, we just need to know if ALL lines through x are blocked.
    #   Let available_lines[x] be the number of lines through x that are NOT blocked.
    #   Initially, available_lines[x] = 50 for all x.
    #   When we add y to S:
    #     For each line L through y:
    #       For each x in L \ {y}:
    #         Check if L was previously "free" for x (i.e., blocked_count[x][L] == 0).
    #         If yes, decrement available_lines[x].
    #         Increment blocked_count[x][L].
    #     If any available_lines[x] becomes 0, we have a conflict (unless x==y? No, if x==y, we don't care about lines through x being blocked by y? 
    #     Wait. The condition is: for every x, there is a line L through x such that L \ {x} subset N.
    #     If x in S, L \ {x} subset N means L contains no other points of S.
    #     So if x in S, we need a line L through x with blocked_count[x][L] == 0.
    #     If x in N, we need a line L through x with blocked_count[x][L] == 0.
    #     So in both cases, we need available_lines[x] > 0.
    
    in_S = np.zeros(q*q, dtype=bool)
    # blocked_count[x] is an array of 50 integers, counting points in S on each line through x.
    # We only care if it's 0 or >0.
    # Let's use an array of size 2401 x 50.
    blocked = np.zeros((q*q, 50), dtype=np.int8)
    available = np.full(q*q, 50, dtype=np.int8)
    
    # We need to map the global line index to the local index in point_lines[x].
    # point_lines[x] is a list of 50 global line indices.
    # We need a quick lookup: given x and global line L, what is the local index?
    # Let's create a dict or array for this.
    # local_idx_map[x][global_line] = local_index
    # This is 2401 * 2450, which is ~6M. Manageable.
    local_idx_map = [{} for _ in range(q*q)]
    for x in range(q*q):
        for i, gl in enumerate(point_lines[x]):
            local_idx_map[x][gl] = i
            
    # Candidates: all points.
    # We can try to add them in a specific order. 
    # Random order might be good to avoid local minima, but deterministic is better for reproducibility.
    # Let's try all points.
    candidates = list(range(q*q))
    
    # To improve score, we want to maximize |S|.
    # Greedy: iterate through candidates. If we can add, add.
    # We can restart with different orders if time permits.
    
    best_S_size = 0
    best_S_mask = in_S.copy()
    
    # We will try a few different orderings or just one good one.
    # Given time constraints, one pass might not be enough if order matters.
    # But let's start with one pass.
    
    for y in candidates:
        if time.time() > deadline - 60:
            break
            
        # Check if y is already in S (shouldn't be, as we only add)
        if in_S[y]:
            continue
            
        # Tentatively add y
        in_S[y] = True
        
        # Update blocked counts
        # Lines through y
        y_lines = point_lines[y]
        for gl in y_lines:
            # Get points on this line
            pts_on_line = lines[gl]
            for x in pts_on_line:
                if x == y:
                    continue
                # Find local index of gl in point_lines[x]
                li = local_idx_map[x][gl]
                if blocked[x, li] == 0:
                    # This line was previously free for x
                    available[x] -= 1
                    if available[x] == 0:
                        # Conflict! x has no free lines.
                        # Revert and break
                        in_S[y] = False
                        # Revert blocked counts for this y
                        for gl2 in y_lines:
                            pts2 = lines[gl2]
                            for x2 in pts2:
                                if x2 == y:
                                    continue
                                li2 = local_idx_map[x2][gl2]
                                if blocked[x2, li2] > 0:
                                    blocked[x2, li2] -= 1
                                    if blocked[x2, li2] == 0:
                                        available[x2] += 1
                        break
                else:
                    blocked[x, li] += 1
            else:
                # Inner loop finished without break
                continue
            # If we broke out of inner loop, y was rejected
            break
        else:
            # y was accepted
            if np.sum(in_S) > best_S_size:
                best_S_size = np.sum(in_S)
                best_S_mask = in_S.copy()
                
    # Construct N from best_S_mask
    # N is the complement of S.
    N_indices = [i for i in range(q*q) if not best_S_mask[i]]
    
    # Convert indices to [[xa, xb], [ya, yb]]
    result = []
    for idx in N_indices:
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        result.append([[xa, xb], [ya, yb]])
        
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
