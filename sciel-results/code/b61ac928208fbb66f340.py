import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses baseline construction followed by fast greedy pruning with
    precomputed membership matrix for O(1) line-membership checks.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0
    
    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_7

    # Element representation: (a, b) represents a + b*alpha
    # idx = a * p + b

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

    # Precompute field operations
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    
    elems = [idx_to_elem(i) for i in range(q)]
    
    for i in range(q):
        a_i, b_i = elems[i]
        for j in range(q):
            a_j, b_j = elems[j]
            add_table[i, j] = elem_to_idx((a_i + a_j) % p, (b_i + b_j) % p)
            prod_a = (a_i * a_j + b_i * b_j * alpha_sq) % p
            prod_b = (a_i * b_j + b_i * a_j) % p
            mul_table[i, j] = elem_to_idx(prod_a, prod_b)

    num_dirs = q + 1  # 50 directions
    total_points = q * q  # 2401

    # Build baseline Nikodym set
    in_N = np.ones(total_points, dtype=bool)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    for x_idx in range(q):
        xa, xb = elems[x_idx]
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = elems[y_idx]
            norm_y = (ya * ya - yb * yb * alpha_sq) % p
            p_idx = x_idx * q + y_idx
            if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                in_N[p_idx] = False

    # Precompute line points
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)
    
    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx
        
        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    # Precompute membership: mem[pt, d] = set of points on line (pt, d)
    # For fast lookup: is_on_line[point, line_pt, dir] = bool
    # But that's 2401*2401*50 = 288M entries - too big.
    # Instead, use: for each point y, precompute which (x, d) pairs have y on line (x, d)
    # i.e., reverse_index[y] = list of (x, d) such that y is on line (x, d)
    
    # Build reverse index: for each point y, find all (x, d) where y is on line through x with direction d
    # y is on line (x, d) iff y = x + t*v_d for some t in F_49, t != 0
    # This means (y - x) is parallel to v_d, i.e., (y-x) = t * v_d
    
    # For each y, for each d, the set of x such that y is on line (x, d) is:
    # x = y - t * v_d for t in F_49 (all 49 values including t=0 which gives x=y)
    # But line (x, d) contains x and x + t*v_d for t=1..48, so y is on line (x,d) 
    # iff y = x + t*v_d for some t in {1,...,48}, i.e., x = y - t*v_d for t=1..48
    
    # Precompute: for each y and d, the list of x such that y is on line (x, d)
    # That's 48 x-values for each (y, d) pair
    # Total: 2401 * 50 * 48 = 5.76M entries
    
    # Actually, let's just build a 3D boolean array:
    # on_line[y, x, d] = True if y is on line (x, d)
    # Size: 2401 * 2401 * 50 = 288M bools = 288 MB - too much
    
    # Better: for each y, build a set of (x, d) pairs
    # Or: use the fact that y is on line (x, d) iff (y - x) is a scalar multiple of v_d
    
    # Let's precompute for each direction d, the "parallel" relation:
    # two vectors are parallel if one is a scalar multiple of the other
    # v_d = (1, m) for d < 49, or (0, 1) for d = 49
    
    # For direction d < 49: v = (1, m)
    # y - x = (dx, dy) is parallel to (1, m) iff dy = m * dx (when dx != 0)
    # or dx = 0 and dy = 0 (same point, but t != 0 so this doesn't count)
    
    # For direction d = 49: v = (0, 1)
    # y - x = (dx, dy) is parallel to (0, 1) iff dx = 0
    
    # Let's build: for each point y, for each direction d, the list of points x 
    # such that y is on the line through x in direction d (excluding x itself)
    
    # Actually, the simplest approach: precompute a 2D array for each direction
    # parallel_mask_d[x, y] = True if (y - x) is parallel to v_d
    # Size per direction: 2401 * 2401 = 5.76M, times 50 = 288M - still too big
    
    # Alternative: use the line_points array we already have.
    # For fast checking, build a dictionary: for each (x, d), the set of points on that line
    # Then to check if y is on line (x, d), just check membership in the set.
    
    # But Python set membership is O(1) on average, so we can use that.
    # Build: line_set[x, d] = frozenset of points on line (x, d)
    # That's 2401 * 50 = 120050 sets, each with 48 elements.
    
    # Actually, even simpler: we already have line_points[pt, d, :] which is the 48 other points.
    # To check if y is on line (x, d), we need to check if y == x or y is in line_points[x, d, :].
    # The `y in line_points[x, d, :]` is the slow part.
    
    # Let's precompute a 2D boolean array for each direction:
    # on_line_d[y, x] = True if y is on line (x, d) [including x itself]
    # We only need this for the validity check.
    
    # For efficiency, let's precompute for each direction d:
    # a 2401x2401 boolean array, but that's 5.76M per direction, 50 directions = 288M bools
    # In numpy, that's 288 MB which is too much.
    
    # Compromise: precompute for each direction d, a list of arrays or use sparse representation.
    # Or: just accept the O(48) search but vectorize it.
    
    # Better approach: for each point y, precompute which (x, d) pairs have y on line (x, d).
    # Build: reverse_lines[y] = list of (x, d) tuples
    # Total entries: each point y is on exactly 50 lines (one for each direction), 
    # and each line has 49 points, so each point is on 50 * 49 / 49 = 50 lines? No.
    # Each point y is on exactly one line for each direction d, so 50 lines total.
    # So reverse_lines[y] has exactly 50 entries.
    
    # Let's build it:
    reverse_lines = [[] for _ in range(total_points)]
    for x in range(total_points):
        for d in range(num_dirs):
            # x is on line (x, d) trivially, but we want to find lines that pass through x
            # Line (x, d) contains x and line_points[x, d, :]
            # So for each point y on this line, add (x, d) to reverse_lines[y]
            reverse_lines[x].append((x, d))
            for y in line_points[x, d, :]:
                reverse_lines[y].append((x, d))
    
    # Now to check if y is on line (x, d), we can check if (x, d) is in reverse_lines[y]
    # But that's O(50) per check. Still better than O(48) with array search? Maybe similar.
    
    # Even better: build a set for each y
    reverse_sets = [set() for _ in range(total_points)]
    for y in range(total_points):
        for (x, d) in reverse_lines[y]:
            reverse_sets[y].add((x, d))
    
    # Now checking if y is on line (x, d) is O(1) via reverse_sets[y].contains((x, d))
    
    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    
    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()

    def try_prune(order, max_rounds=3):
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir
        
        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()
        removed = 0
        
        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break
            
            progress = False
            for y in order:
                if not work[y]:
                    continue
                
                # Check removability using precomputed reverse sets
                removable = True
                # For each (x, d) such that y is on line (x, d), check if removing y
                # would invalidate line (x, d) and if x has other valid lines
                for (x, d) in reverse_lines[y]:
                    if ird[x, d]:
                        # y is on line (x, d) and this line is currently valid for x
                        # If we remove y, line (x, d) becomes invalid
                        # Check if x has other valid lines
                        if wc[x] <= 1:
                            removable = False
                            break
                
                if removable:
                    work[y] = False
                    removed += 1
                    progress = True
                    # Update validity: for each (x, d) where y is on line (x, d),
                    # if line (x, d) was valid, it becomes invalid
                    for (x, d) in reverse_lines[y]:
                        if ird[x, d]:
                            ird[x, d] = False
                            wc[x] -= 1
            
            if not progress:
                break
        
        comp = np.sum(~work)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work.copy()
            best_valid_count = wc.copy()
            best_is_valid_dir = ird.copy()
        
        in_N = best_in_N.copy()
        valid_count = best_valid_count.copy()
        is_valid_dir = best_is_valid_dir.copy()

    # Strategy 1: Sequential order
    order1 = list(range(total_points))
    try_prune(order1, max_rounds=3)

    # Strategy 2: Random order
    if time.time() < deadline - safety_margin:
        rng = random.Random(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune(order2, max_rounds=3)

    # Strategy 3: Prioritize points with low "coverage"
    if time.time() < deadline - safety_margin:
        coverage = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not in_N[y]:
                continue
            for (x, d) in reverse_lines[y]:
                if is_valid_dir[x, d]:
                    coverage[y] += 1
        
        order3 = sorted(range(total_points), key=lambda y: 0 if not in_N[y] else coverage[y])
        try_prune(order3, max_rounds=2)

    # Strategy 4: Multiple random restarts
    if time.time() < deadline - safety_margin:
        for seed in [123, 456, 789, 1000, 2000, 3000]:
            if time.time() > deadline - safety_margin:
                break
            rng = random.Random(seed)
            order4 = list(range(total_points))
            rng.shuffle(order4)
            try_prune(order4, max_rounds=2)

    # Final validity check
    if np.any(valid_count == 0):
        in_N = np.ones(total_points, dtype=bool)
        for x_idx in range(q):
            xa, xb = elems[x_idx]
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = elems[y_idx]
                norm_y = (ya * ya - yb * yb * alpha_sq) % p
                p_idx = x_idx * q + y_idx
                if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                    in_N[p_idx] = False

    # Extract points
    points = []
    for p_idx in range(total_points):
        if in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
