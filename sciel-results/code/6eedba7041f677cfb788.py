import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Uses a MODEL-AND-OPTIMIZE approach: parameterize the set N and optimize
    the complement size directly via greedy removal and local search.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 600s
    safety_margin = 5.0

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
    # N = {(x, y) : Trace(x) + Norm(y) = 0} minus Y_special
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

    # Precompute line points: for each point and direction, the other q-1 points on the line
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)

    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx  # element (0,1) = 1

        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    # Precompute: for each point y and direction d, which points x have y on their line in direction d
    # i.e., y is on the line through x with direction d
    # This is: for each x, d, the points on line through x in dir d include x itself and q-1 others
    # y is on line through x in dir d iff x is on line through y in dir d (symmetric for affine lines)
    # Actually, lines are undirected in the sense that if y = x + t*v, then x = y + (-t)*v
    # So y is on line through x in dir d iff x is on line through y in dir d
    # We can build: for each y, d, the set of x such that y is on line through x in dir d
    # This includes x=y itself and the q-1 other points on line through y in dir d
    # So: neighbors_on_line[y, d] = {y} union line_points[y, d, :]
    # And x has y on its line in dir d iff y is in neighbors_on_line[x, d] iff x is in neighbors_on_line[y, d]
    
    # Precompute membership: mem[y, d, x] = True if y is on line through x in direction d
    # Equivalent: mem[y, d, x] = True iff x is on line through y in direction d (including y itself)
    # So mem[y, d, :] has True at position y and at positions line_points[y, d, :]
    mem = np.zeros((total_points, num_dirs, total_points), dtype=bool)
    for y in range(total_points):
        mem[y, :, y] = True  # y is on its own line
        for d in range(num_dirs):
            mem[y, d, line_points[y, d, :]] = True

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    # Vectorized initialization
    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    # Greedy pruning with multiple strategies
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

                # Check removability: for all x such that y is on a valid line of x in direction d,
                # y must not be the ONLY valid line point for x in direction d
                # More precisely: removing y invalidates (x,d) if ird[x,d] and y is on line through x in dir d
                # After removal, x still needs at least one valid direction
                # So y is removable iff for every (x,d) with ird[x,d] and y on line(x,d), wc[x] > 1
                # (because removing y only affects direction d for x, so x still has wc[x]-1 valid dirs if wc[x]>1)
                # Wait, that's not right. Removing y invalidates direction d for x. 
                # x remains valid if it has at least one OTHER valid direction.
                # So the condition is: for every (x,d) with ird[x,d] and y on line(x,d), wc[x] >= 2
                # Because after removing y, direction d is gone, but x has wc[x]-1 other valid directions.
                # x is still valid iff wc[x]-1 >= 1, i.e., wc[x] >= 2.
                
                # Find all (x,d) pairs where ird[x,d] is True and y is on line through x in direction d
                # y is on line through x in dir d iff mem[x, d, y] is True
                # We need: for all such (x,d), wc[x] >= 2
                
                # Get all directions d where there exists x with ird[x,d] and mem[x,d,y]
                # For each d, find x's where ird[x,d] and mem[x,d,y]
                removable = True
                for d in range(num_dirs):
                    # x's that have y on their line in direction d and have valid direction d
                    xs = np.where(ird[:, d] & mem[:, d, y])[0]
                    if len(xs) > 0:
                        # Exclude y itself, as removing y does not affect y's own line validity
                        # (validity depends on OTHER points)
                        if y in xs:
                            xs = xs[xs != y]
                        if len(xs) > 0:
                            # Check if any of these x's have wc[x] == 1
                            if np.any(wc[xs] == 1):
                                removable = False
                                break

                if removable:
                    work[y] = False
                    removed += 1
                    progress = True
                    # Update: for each d, for each x where ird[x,d] and mem[x,d,y], invalidate
                    for d in range(num_dirs):
                        xs = np.where(ird[:, d] & mem[:, d, y])[0]
                        if len(xs) > 0:
                            ird[xs, d] = False
                            wc[xs] -= 1

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
        rng = np.random.RandomState(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune(order2, max_rounds=3)

    # Strategy 3: Prioritize points with low "coverage"
    if time.time() < deadline - safety_margin:
        # Points that are on fewer valid lines for others are safer to remove
        coverage = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not in_N[y]:
                continue
            for d in range(num_dirs):
                xs = np.where(is_valid_dir[:, d] & mem[:, d, y])[0]
                coverage[y] += len(xs)

        order3 = sorted(range(total_points), key=lambda y: 0 if not in_N[y] else coverage[y])
        try_prune(order3, max_rounds=2)

    # Strategy 4: Prioritize points with high "coverage" (more redundant)
    if time.time() < deadline - safety_margin:
        coverage2 = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not in_N[y]:
                continue
            for d in range(num_dirs):
                xs = np.where(is_valid_dir[:, d] & mem[:, d, y])[0]
                coverage2[y] += len(xs)

        order4 = sorted(range(total_points), key=lambda y: 0 if not in_N[y] else -coverage2[y])
        try_prune(order4, max_rounds=2)

    # Strategy 5: More random restarts
    if time.time() < deadline - safety_margin:
        for seed in [123, 456, 789, 101, 202, 303, 404, 505, 606, 707]:
            if time.time() > deadline - safety_margin:
                break
            rng = np.random.RandomState(seed)
            order5 = list(range(total_points))
            rng.shuffle(order5)
            try_prune(order5, max_rounds=2)

    # Strategy 6: Iterative random with increasing rounds
    if time.time() < deadline - safety_margin:
        for seed in [111, 222, 333, 444, 555]:
            if time.time() > deadline - safety_margin:
                break
            rng = np.random.RandomState(seed)
            order6 = list(range(total_points))
            rng.shuffle(order6)
            try_prune(order6, max_rounds=4)

    # Final validity check
    if np.any(best_valid_count == 0):
        # Invalid, revert to baseline
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
        valid_count = np.zeros(total_points, dtype=np.int32)
        is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
        for pt in range(total_points):
            for d in range(num_dirs):
                if np.all(in_N[line_points[pt, d, :]]):
                    is_valid_dir[pt, d] = True
                    valid_count[pt] += 1
        best_in_N = in_N.copy()
        best_valid_count = valid_count.copy()
        best_is_valid_dir = is_valid_dir.copy()
        best_complement = np.sum(~in_N)
        in_N = best_in_N.copy()
        valid_count = best_valid_count.copy()
        is_valid_dir = best_is_valid_dir.copy()

    # Extract points
    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
