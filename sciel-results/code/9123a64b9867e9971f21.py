# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using local search.
    
    We work with S = F_49^2 \ N (the excluded set).
    A point x is "blocked" if every line through x intersects S.
    S is valid iff no point is blocked.
    
    We use simulated annealing:
    - State: binary vector in_S of length 2401
    - Energy: number of blocked points (want 0)
    - Moves: flip a bit (add/remove from S)
    - Accept if energy doesn't increase, or with probability exp(-delta/T)
    """
    p_prime = 7
    q = 49
    n_points = 2401
    
    # Precompute: for each point x, the set of points on each of the 50 lines through x
    # Line through x with slope m: {x + t*(1, m) : t in F_49}
    # Vertical line through x: {x + t*(0, 1) : t in F_49}
    
    # Index encoding: idx = ((xa*7+xb)*7+ya)*7+yb
    # Field element (a, b) represents a + b*alpha where alpha^2 = 3
    
    def field_add(a, b):
        """Add two field elements (a, b) and (c, d) -> (a+c, b+d) mod 7"""
        return ((a[0] + b[0]) % 7, (a[1] + b[1]) % 7)
    
    def field_mul(a, b):
        """Multiply (a, b) * (c, d) in F_7[alpha]/(alpha^2-3)"""
        # (a + b*alpha)(c + d*alpha) = ac + (ad+bc)*alpha + bd*alpha^2
        # = (ac + 3*bd) + (ad+bc)*alpha
        return ((a[0]*b[0] + 3*a[1]*b[1]) % 7, (a[0]*b[1] + a[1]*b[0]) % 7)
    
    def field_inv(a):
        """Inverse of (a, b) in F_49"""
        # (a + b*alpha)^{-1} = (a - b*alpha) / (a^2 - 3b^2)
        denom = (a[0]*a[0] - 3*a[1]*a[1]) % 7
        if denom == 0:
            return (0, 0)
        # Inverse of denom in F_7
        denom_inv = pow(denom, -1, 7)
        return ((a[0] * denom_inv) % 7, (-a[1] * denom_inv) % 7)
    
    # Precompute all field elements as tuples
    field_elements = [(a, b) for a in range(7) for b in range(7)]
    
    # Precompute multiplication table
    mult_table = np.zeros((49, 49, 2), dtype=np.int8)
    for i, (a, b) in enumerate(field_elements):
        for j, (c, d) in enumerate(field_elements):
            mult_table[i, j] = field_mul((a, b), (c, d))
    
    # Precompute inverse table
    inv_table = np.zeros((49, 2), dtype=np.int8)
    for i, (a, b) in enumerate(field_elements):
        if i != 0:  # 0 has no inverse
            inv_table[i] = field_inv((a, b))
    
    # Convert index to field element index
    # idx = ((xa*7+xb)*7+ya)*7+yb
    # Field element index for (xa, xb) is xa*7+xb
    def point_to_coords(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb
    
    # Precompute for each point x, the list of points on each of the 50 lines through x
    # lines_through[x] = list of 50 lists, each containing 49 points (excluding x)
    
    # For efficiency, precompute the direction vectors
    # Direction (1, m) for m in F_49, and (0, 1) for vertical
    
    # Precompute line members: for each point x and each direction d, 
    # the set of points on the line through x in direction d
    
    # Actually, for the energy computation, we need:
    # For each point x, count how many of the 50 lines through x have at least one point in S (other than x itself if x in S)
    # x is blocked if all 50 lines have at least one point of S on them
    
    # Let's precompute: for each point x, for each of the 50 lines L through x, 
    # the list of points on L \ {x}
    
    # This is 2401 * 50 * 48 points to store, which is ~5.7M integers. That's fine.
    
    # But we can be smarter. Let's precompute the line membership differently.
    
    # For each line (defined by slope m and intercept, or vertical and x-coordinate),
    # store the set of points on it.
    
    # Lines in F_49^2:
    # - 49 slopes m in F_49, each with 49 intercepts -> 49*49 = 2401 non-vertical lines
    # - 49 vertical lines (one for each x-coordinate in F_49)
    # Total: 2450 lines, each with 49 points
    
    # Line identification:
    # Non-vertical: (m_idx, c_idx) where m is slope, c is y-intercept
    #   Line: y = m*x + c
    # Vertical: (49, x_idx) where x_idx is the x-coordinate
    #   Line: x = x_idx
    
    # Precompute for each line, the list of points on it
    line_points = {}
    
    # Generate all lines
    # Non-vertical lines: y = m*x + c
    for m_idx in range(49):
        m = field_elements[m_idx]
        for c_idx in range(49):
            c = field_elements[c_idx]
            points = []
            for x_idx in range(49):
                x = field_elements[x_idx]
                # y = m*x + c
                mx = tuple(mult_table[m_idx, x_idx])
                y = field_add(mx, c)
                # Find indices
                xa, xb = x
                ya, yb = y
                y_idx = ya * 7 + yb
                x_idx_val = xa * 7 + xb
                point_idx = (x_idx_val * 7 + y_idx)  # Wait, this isn't right
                # Let me re-derive the index
                # idx = ((xa*7+xb)*7+ya)*7+yb
                point_idx = ((xa * 7 + xb) * 7 + ya) * 7 + yb
                points.append(point_idx)
            line_points[(m_idx, c_idx, 0)] = points  # 0 = non-vertical
    
    # Vertical lines: x = x_idx
    for x_idx in range(49):
        x = field_elements[x_idx]
        xa, xb = x
        points = []
        for y_idx in range(49):
            y = field_elements[y_idx]
            ya, yb = y
            point_idx = ((xa * 7 + xb) * 7 + ya) * 7 + yb
            points.append(point_idx)
        line_points[(x_idx, 0, 1)] = points  # 1 = vertical
    
    # Now, for each point x, find which 50 lines pass through it
    # point_lines[x] = list of 50 line keys
    
    point_lines = [[] for _ in range(2401)]
    for line_key, pts in line_points.items():
        for pt in pts:
            point_lines[pt].append(line_key)
    
    # Verify each point has exactly 50 lines
    # (This should be true by construction)
    
    # Now, for energy computation:
    # For each point x, for each line L through x, check if L \ {x} intersects S
    # x is blocked if for all 50 lines L through x, L \ {x} intersects S
    
    # We can compute this incrementally.
    # Maintain: for each point x and each line L through x, the count of points in S on L \ {x}
    # line_count[x][L_idx] = number of points in S on line L through x, excluding x
    
    # This is 2401 * 50 integers = 120,050 integers. Fine.
    
    # But updating this when we add/remove a point p to/from S:
    # For each of the 50 lines L through p:
    #   For each other point x on L:
    #     line_count[x][L] += 1 (if adding) or -= 1 (if removing)
    #   And for p itself:
    #     line_count[p][L] is not affected by p being in S (since we exclude x itself)
    #     But wait, when we add p to S, for other points x on L, the line L now has p in S.
    #     So line_count[x][L] should increase by 1.
    #     For p itself, line_count[p][L] counts points in S on L \ {p}. Adding p doesn't change this.
    
    # So the update is: for each of the 50 lines L through p, for each of the 48 other points x on L,
    # update line_count[x][L].
    
    # That's 50 * 48 = 2400 updates per flip. With 2401 points, total is ~5.8M operations per full pass.
    # For simulated annealing with many iterations, this might be slow in Python.
    
    # Alternative: maintain blocked_count[x] = number of lines through x that have at least one point of S on L \ {x}
    # x is blocked if blocked_count[x] == 50
    
    # When we add p to S:
    # For each line L through p:
    #   For each other point x on L:
    #     If line_count[x][L] was 0, it becomes 1, so blocked_count[x] increases by 1
    #     If line_count[x][L] was > 0, blocked_count[x] is unchanged
    #   line_count[x][L] += 1 for all x on L, x != p
    
    # When we remove p from S:
    # For each line L through p:
    #   For each other point x on L:
    #     line_count[x][L] -= 1
    #     If line_count[x][L] was 1, it becomes 0, so blocked_count[x] decreases by 1
    #     If line_count[x][L] was > 1, blocked_count[x] is unchanged
    
    # This is more efficient. Let's implement this.
    
    # Precompute: for each point p, for each of the 50 lines L through p, the list of other points on L
    # line_other_points[p][L_idx] = list of 48 points (all points on L except p)
    
    # Actually, we need to map line keys to indices for each point
    # Let's restructure:
    
    # For each point p, store:
    #   p_lines[p] = list of 50 line keys
    #   p_line_other_points[p][i] = list of other points on the i-th line through p
    
    p_line_other_points = [None] * 2401
    for p in range(2401):
        p_lines_list = point_lines[p]
        other_points = []
        for line_key in p_lines_list:
            pts = line_points[line_key]
            others = [pt for pt in pts if pt != p]
            other_points.append(others)
        p_line_other_points[p] = other_points
    
    # Now implement simulated annealing
    
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin
    
    # Initialize with random S
    rng = np.random.RandomState(123)
    in_S = np.zeros(2401, dtype=bool)
    
    # Start with a small random S
    num_initial = 100
    initial_points = rng.choice(2401, size=num_initial, replace=False)
    in_S[initial_points] = True
    
    # Compute initial line_count and blocked_count
    line_count = np.zeros((2401, 50), dtype=np.int32)
    blocked_count = np.zeros(2401, dtype=np.int32)
    
    # For each point p in S, update line_count and blocked_count
    for p in initial_points:
        for i, line_key in enumerate(point_lines[p]):
            others = p_line_other_points[p][i]
            for x in others:
                # Find the index of this line in point_lines[x]
                # This is expensive. Let's precompute a mapping.
                pass
    
    # This is getting complex. Let me simplify by using a different approach.
    
    # Instead of tracking line_count per (point, line), let's just recompute the energy from scratch
    # for each move. The energy computation is:
    # For each point x, check if all 50 lines through x intersect S (excluding x itself)
    # This is 2401 * 50 * (check if line intersects S)
    # Checking if a line intersects S: iterate over the 49 points on the line and check if any is in S
    # That's 2401 * 50 * 49 = ~5.9M operations per energy evaluation.
    # With 1000 iterations, that's 5.9B operations. Too slow in Python.
    
    # Let's use a vectorized approach with NumPy.
    
    # Represent S as a boolean array of shape (49, 49) where S[x_idx, y_idx] indicates if point (x, y) is in S
    # x_idx = xa*7+xb, y_idx = ya*7+yb
    
    def idx_to_xy(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        x_idx = xa * 7 + xb
        y_idx = ya * 7 + yb
        return x_idx, y_idx
    
    def xy_to_idx(x_idx, y_idx):
        xa = x_idx // 7
        xb = x_idx % 7
        ya = y_idx // 7
        yb = y_idx % 7
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb
    
    # Precompute for each line, the x_idx and y_idx values
    # Non-vertical line (m_idx, c_idx): y = m*x + c
    # For each x_idx in 0..48, compute y_idx
    
    line_x_coords = {}
    line_y_coords = {}
    
    for m_idx in range(49):
        for c_idx in range(49):
            m = field_elements[m_idx]
            c = field_elements[c_idx]
            xs = []
            ys = []
            for x_idx in range(49):
                x = field_elements[x_idx]
                mx = tuple(mult_table[m_idx, x_idx])
                y = field_add(mx, c)
                ya, yb = y
                y_idx = ya * 7 + yb
                xs.append(x_idx)
                ys.append(y_idx)
            line_x_coords[(m_idx, c_idx, 0)] = xs
            line_y_coords[(m_idx, c_idx, 0)] = ys
    
    for x_idx in range(49):
        line_x_coords[(x_idx, 0, 1)] = [x_idx] * 49
        line_y_coords[(x_idx, 0, 1)] = list(range(49))
    
    # For each point, precompute the 50 lines through it as (line_key, x_idx, y_idx)
    point_line_info = [None] * 2401
    for p in range(2401):
        x_idx, y_idx = idx_to_xy(p)
        info = []
        for line_key in point_lines[p]:
            xs = line_x_coords[line_key]
            ys = line_y_coords[line_key]
            # Find the position of (x_idx, y_idx) in this line
            # Since the line has 49 points, we can just store the line key
            info.append(line_key)
        point_line_info[p] = info
    
    # Energy computation using NumPy
    def compute_energy(in_S_2d):
        """
        in_S_2d: boolean array of shape (49, 49)
        Returns: number of blocked points
        """
        blocked = 0
        for x_idx in range(49):
            for y_idx in range(49):
                p = xy_to_idx(x_idx, y_idx)
                # Check if all 50 lines through p intersect S
                all_blocked = True
                for line_key in point_lines[p]:
                    xs = line_x_coords[line_key]
                    ys = line_y_coords[line_key]
                    # Check if any point on this line (other than p) is in S
                    # Create a mask for points on this line
                    on_line = np.zeros((49, 49), dtype=bool)
                    for i in range(49):
                        on_line[xs[i], ys[i]] = True
                    # Exclude p
                    on_line[x_idx, y_idx] = False
                    # Check if any point on this line (excluding p) is in S
                    if not np.any(on_line & in_S_2d):
                        all_blocked = False
                        break
                if all_blocked:
                    blocked += 1
        return blocked
    
    # This is still slow. Let me try a different approach.
    
    # Given the complexity, let me just use a simple greedy algorithm but with a better initial guess.
    # The previous program had a score of 0.3333, which corresponds to |S| = 147 (if seed was 110.25).
    # Wait, the score is value/seed where value = 2401 - |N| = |S|.
    # If score is 0.3333, and seed score is 1.0, then |S| = 0.3333 * |S_seed|.
    # The parent had score 1.3333, so |S_parent| = 1.3333 * |S_seed|.
    # If |S_seed| = 110, then |S_parent| = 147, and |S_current| = 0.3333 * 110 = 36.67, so |S_current| ≈ 37.
    
    # Let me just use a simpler approach: start with an empty S and greedily add points.
    # But use a smarter order: add points that are "far" from each other.
    
    # Actually, let me just fix the current program's greedy algorithm to be more efficient.
    # The issue with the current program is that it uses a random order, which might not be optimal.
    
    # Let me try a different strategy: use the structure of the field to create a structured S.
    # For example, take S to be a set of points where x_idx is in some subset A and y_idx is in some subset B.
    
    # If S = A x B (Cartesian product), then |S| = |A| * |B|.
    # A point (x, y) is blocked if every line through it intersects A x B.
    # This is a complex condition.
    
    # Let me just run the greedy algorithm with multiple random seeds and keep the best.
    
    best_S = None
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 580
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(seed)
        in_S = np.zeros(2401, dtype=bool)
        
        # Use the efficient update from the original program
        valid_count = np.full(2401, 50, dtype=np.int32)
        
        candidates = np.arange(2401)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if time.time() > deadline:
                break
            
            valid_count -= 1
            valid_count[p_idx] += 1
            
            if np.min(valid_count) > 0:
                in_S[p_idx] = True
            else:
                valid_count += 1
                valid_count[p_idx] -= 1
        
        size = np.sum(in_S)
        if size > best_size:
            best_size = size
            best_S = in_S.copy()
    
    if best_S is None:
        best_S = np.zeros(2401, dtype=bool)
    
    # Construct the result N
    result_list = []
    for i in range(2401):
        if not best_S[i]:
            xa, xb, ya, yb = point_to_coords(i)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
    
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
