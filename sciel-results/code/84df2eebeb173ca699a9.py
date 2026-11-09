import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0
    safety_margin = 10.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

    elems = [idx_to_elem(i) for i in range(q)]

    # Precompute field operations
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)

    for i in range(q):
        a_i, b_i = elems[i]
        for j in range(q):
            a_j, b_j = elems[j]
            add_table[i, j] = elem_to_idx((a_i + a_j) % p, (b_i + b_j) % p)
            prod_a = (a_i * a_j + b_i * b_j * alpha_sq) % p
            prod_b = (a_i * b_j + b_i * a_j) % p
            mul_table[i, j] = elem_to_idx(prod_a, prod_b)

    num_dirs = q + 1  # 50
    total_points = q * q  # 2401

    # Build baseline: N = all points except those with trace(x)+norm(y)=0 and y in y_special
    in_N = np.ones(total_points, dtype=bool)

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    trace_arr = np.zeros(q, dtype=np.int64)
    norm_arr = np.zeros(q, dtype=np.int64)
    for i in range(q):
        a, b = elems[i]
        trace_arr[i] = (2 * a) % p
        norm_arr[i] = (a * a - b * b * alpha_sq) % p

    y_special_mask = np.zeros(q, dtype=bool)
    for ys in y_special:
        y_special_mask[elem_to_idx(ys[0], ys[1])] = True

    sum_arr = (trace_arr[:, None] + norm_arr[None, :]) % p
    condition = (sum_arr == 0) & y_special_mask[None, :]
    condition_flat = condition.flatten()
    in_N[condition_flat] = False

    # Precompute line points: line_points[pt, d, t] = point at position t on line through pt in direction d (t=1..48)
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)

    # For each direction, precompute the offset points
    dir_offsets = np.zeros((num_dirs, q - 1, 2), dtype=np.int64)  # [d, t, 0]=dx_idx, [d,t,1]=dy_idx
    for d in range(num_dirs):
        if d < q:
            vx = one_idx
            vy = d
        else:
            vx = zero_idx
            vy = one_idx
        for t in range(1, q):
            dir_offsets[d, t - 1, 0] = mul_table[t, vx]
            dir_offsets[d, t - 1, 1] = mul_table[t, vy]

    # line_points[pt, d, t] for t=0..47
    x_indices = np.arange(total_points) // q
    y_indices = np.arange(total_points) % q

    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    for d in range(num_dirs):
        dx = dir_offsets[d, :, 0]  # (48,)
        dy = dir_offsets[d, :, 1]  # (48,)
        nx = add_table[x_indices[:, None], dx[None, :]]  # (2401, 48)
        ny = add_table[y_indices[:, None], dy[None, :]]  # (2401, 48)
        line_points[:, d, :] = nx * q + ny

    # Compute initial validity
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    valid_count = np.zeros(total_points, dtype=np.int32)

    for d in range(num_dirs):
        mask = in_N[line_points[:, d, :]]  # (2401, 48)
        valid_pts = np.all(mask, axis=1)
        is_valid_dir[:, d] = valid_pts
        valid_count += valid_pts.astype(np.int32)

    # covers[y, d, :] = points x such that y is on line x-d = line_points[y, d, :]
    # (by symmetry)

    best_complement = int(np.sum(~in_N))
    best_in_N = in_N.copy()
    best_is_valid_dir = is_valid_dir.copy()
    best_valid_count = valid_count.copy()

    def try_prune(order, max_rounds=30):
        nonlocal best_complement, best_in_N, best_is_valid_dir, best_valid_count

        work_in_N = best_in_N.copy()
        work_is_valid_dir = best_is_valid_dir.copy()
        work_valid_count = best_valid_count.copy()

        for rnd in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break

            progress = False
            for y in order:
                if not work_in_N[y]:
                    continue

                # Check if y is removable
                removable = True
                for d in range(num_dirs):
                    xs = line_points[y, d, :]  # points x such that y is on line x-d
                    vm = work_is_valid_dir[xs, d]
                    if np.any(vm):
                        if np.any(work_valid_count[xs][vm] == 1):
                            removable = False
                            break

                if removable:
                    work_in_N[y] = False
                    progress = True
                    for d in range(num_dirs):
                        xs = line_points[y, d, :]
                        vm = work_is_valid_dir[xs, d]
                        if np.any(vm):
                            work_valid_count[xs] -= vm.astype(np.int32)
                            work_is_valid_dir[xs, d] = False

            if not progress:
                break

        comp = int(np.sum(~work_in_N))
        if comp > best_complement:
            best_complement = comp
            best_in_N = work_in_N.copy()
            best_is_valid_dir = work_is_valid_dir.copy()
            best_valid_count = work_valid_count.copy()

    # Strategy 1: Sequential
    if time.time() < deadline - safety_margin:
        try_prune(list(range(total_points)), max_rounds=50)

    # Strategy 2: Multiple random orders
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        if time.time() > deadline - safety_margin:
            break
        rng = random.Random(seed)
        order = list(range(total_points))
        rng.shuffle(order)
        try_prune(order, max_rounds=50)

    # Strategy 3: Criticality-based ordering
    if time.time() < deadline - safety_margin:
        criticality = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not best_in_N[y]:
                continue
            for d in range(num_dirs):
                xs = line_points[y, d, :]
                vm = best_is_valid_dir[xs, d]
                if np.any(vm):
                    criticality[y] += np.sum(best_valid_count[xs][vm] == 1)
        order3 = sorted(range(total_points), key=lambda y: (criticality[y], y))
        try_prune(order3, max_rounds=50)

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
