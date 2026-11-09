import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses baseline construction followed by aggressive greedy pruning.
    """
    start_time = time.time()
    deadline = start_time + 600.0
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

    # Precompute line points
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

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    
    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    # Greedy pruning with conflict-count-based criticality check
    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()

    # Precompute: for each point y, which points x have y on their line in direction d
    # conflict_count[x, y] = number of valid lines of x that pass through y
    # We build this incrementally: start with all lines, then subtract as we remove

    # For fast lookup: given y and d, the 48 other points on that line
    # We need: for a candidate removal of y, check if any x has all its valid lines passing through y
    # i.e., conflict_count[x, y] == valid_count[x] for some x where y is on a valid line of x

    # Precompute: for each y, list of (x, d) pairs where y is on the line through x in direction d
    # and that line is currently valid (ird[x,d] is True)
    # This is expensive to maintain, so we'll use a different approach:
    # For each y, precompute all (x, d) such that y is on line(x, d)
    # Then at removal time, check if any such (x,d) with ird[x,d] has wc[x] <= 1
    # or count valid lines of x through y == wc[x]

    # Precompute for each y: list of (x, d) where y is on line through x in direction d
    # line_points[pt, d, t] gives the t-th other point on line through pt in direction d
    # So y is on line through x in direction d iff x is on line through y in direction d
    # (lines are undirected in the sense that if y = x + t*v then x = y + (q-t)*v)
    # Actually line_points[y, d, :] gives all other points on the line through y in direction d
    # So x is on line through y in direction d iff x is in line_points[y, d, :]
    # AND y is on line through x in direction d (which is true by symmetry of lines)

    # So for each y, the set of (x, d) where y is on line(x,d) is:
    # all (x, d) where x is in line_points[y, d, :]
    # But we also need to handle the vertical direction properly

    # Precompute neighbor lists: for each y, for each d, the 48 x values on the line through y in direction d
    # This is just line_points[y, d, :] which we already have

    def try_prune_fast(order, max_rounds=3):
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()

        # conflict_count[x, y] = number of valid lines of x that pass through y
        # We'll maintain this as a 2D array. Initially compute from ird.
        # conflict_count[x, y] = sum over d of [ird[x,d] and y is on line(x,d)]
        # For each x, for each valid d, for each y in line_points[x,d,:], increment conflict_count[x,y]

        # This is O(total_points * num_dirs * (q-1)) = 2401 * 50 * 48 ~ 5.7M operations
        # But we can do it more efficiently using numpy

        # Actually, let's just do the direct check but with precomputed membership
        # For each y, precompute a boolean array of which (x,d) pairs have y on their line
        # Then the check for removability of y is:
        #   for all (x,d) in pairs[y]: if ird[x,d] and (wc[x] <= 1 or count_valid_lines_of_x_through_y == wc[x])
        #   then y is not removable

        # Precompute: for each y, a list of (x, d) such that y is on the line through x in direction d
        # From line_points: y is on line through x in direction d iff x is on line through y in direction d
        # So for a given y, for each d, x ranges over line_points[y, d, :]
        # But we need to be careful: line_points[y, d, :] gives the OTHER points on the line
        # So x in line_points[y, d, :] means x is on the same line as y in direction d
        # And since lines are symmetric, y is also on the line through x in direction d

        # Build pairs_y: for each y, list of (x, d) where y is on line(x, d)
        # This has 2401 * 50 * 48 = 5,762,400 entries total
        # Too much to store explicitly. Instead, we'll compute on the fly but faster.

        # Alternative: for each y, for each d, we know the 48 x's.
        # Check: for each such x, if ird[x, d] is True, then y is on a valid line of x.
        # We need to know if ALL valid lines of x pass through y.
        # i.e., for each x with ird[x,d], count how many valid lines of x pass through y.
        # If that count equals wc[x], then y is critical for x.

        # To speed this up: precompute for each x, the set of valid directions
        # And for each (x, y), the number of valid lines of x passing through y

        # Let's precompute pass_count[x, y] = number of valid lines of x that pass through y
        # This is a 2401 x 2401 matrix = 5.7M entries, feasible

        pass_count = np.zeros((total_points, total_points), dtype=np.int32)
        for x in range(total_points):
            for d in range(num_dirs):
                if ird[x, d]:
                    for t in range(q - 1):
                        y = line_points[x, d, t]
                        pass_count[x, y] += 1

        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break

            progress = False
            for y in order:
                if not work[y]:
                    continue

                # Check removability using pass_count
                removable = True
                for d in range(num_dirs):
                    for t in range(q - 1):
                        x = line_points[y, d, t]
                        if ird[x, d]:
                            if wc[x] <= 1:
                                removable = False
                                break
                            if pass_count[x, y] >= wc[x]:
                                removable = False
                                break
                    if not removable:
                        break

                if removable:
                    work[y] = False
                    progress = True
                    # Update: for all x where y was on a valid line, invalidate that line
                    for d in range(num_dirs):
                        for t in range(q - 1):
                            x = line_points[y, d, t]
                            if ird[x, d]:
                                ird[x, d] = False
                                wc[x] -= 1
                                # Update pass_count: for all z on line(x,d), pass_count[x,z] -= 1
                                for t2 in range(q - 1):
                                    z = line_points[x, d, t2]
                                    pass_count[x, z] -= 1

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

    def try_prune_pq(max_rounds=3):
        """Priority-queue based greedy: always remove the point with lowest damage."""
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir

        import heapq

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()

        # Precompute pass_count
        pass_count = np.zeros((total_points, total_points), dtype=np.int32)
        for x in range(total_points):
            for d in range(num_dirs):
                if ird[x, d]:
                    for t in range(q - 1):
                        y = line_points[x, d, t]
                        pass_count[x, y] += 1

        # damage[y] = number of valid lines that would be invalidated if y is removed
        # = sum over (x,d) where y is on line(x,d) and ird[x,d] of 1
        damage = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not work[y]:
                continue
            for d in range(num_dirs):
                for t in range(q - 1):
                    x = line_points[y, d, t]
                    if ird[x, d]:
                        damage[y] += 1

        # Priority queue: (damage[y], y)
        pq = []
        for y in range(total_points):
            if work[y]:
                heapq.heappush(pq, (int(damage[y]), y))

        # Track which entries are stale
        pq_version = np.zeros(total_points, dtype=np.int64)
        version = np.zeros(total_points, dtype=np.int64)
        for y in range(total_points):
            if work[y]:
                pq_version[y] = 1

        removed_count = 0
        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break
            if not pq:
                break

            progress = False
            while pq:
                if time.time() > deadline - safety_margin:
                    break

                dm, y = heapq.heappop(pq)
                if not work[y]:
                    continue
                if pq_version[y] != version[y]:
                    continue
                if dm != damage[y]:
                    continue

                # Check if y is removable
                removable = True
                for d in range(num_dirs):
                    for t in range(q - 1):
                        x = line_points[y, d, t]
                        if ird[x, d]:
                            if wc[x] <= 1:
                                removable = False
                                break
                            if pass_count[x, y] >= wc[x]:
                                removable = False
                                break
                    if not removable:
                        break

                if removable:
                    work[y] = False
                    removed_count += 1
                    progress = True
                    # Invalidate all valid lines through y
                    for d in range(num_dirs):
                        for t in range(q - 1):
                            x = line_points[y, d, t]
                            if ird[x, d]:
                                ird[x, d] = False
                                wc[x] -= 1
                                # Update pass_count and damage
                                for t2 in range(q - 1):
                                    z = line_points[x, d, t2]
                                    pass_count[x, z] -= 1
                                    if work[z]:
                                        damage[z] -= 1
                                        version[z] += 1
                                        pq_version[z] = version[z]
                                        heapq.heappush(pq, (int(damage[z]), z))
                    # Also update damage for points on lines that were invalidated
                    # (already done above for z in line(x,d))
                    # But we also need to update damage for y's neighbors on other lines
                    # Actually damage[z] counts valid lines through z. When we invalidate line(x,d),
                    # for each z on that line, damage[z] decreases by 1. Done above.
                    # But we also need to check: are there other lines through z that are still valid?
                    # damage[z] is maintained correctly.

                    # Push updated y's neighbors? No, y is removed.
                    # Push any z whose damage changed: already done above.
                    break  # Process one removal per inner iteration

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

    # Strategy 1: Priority queue (best first)
    if time.time() < deadline - safety_margin:
        try_prune_pq(max_rounds=2)

    # Strategy 2: Sequential order with fast check
    if time.time() < deadline - safety_margin:
        order1 = list(range(total_points))
        try_prune_fast(order1, max_rounds=2)

    # Strategy 3: Random order
    if time.time() < deadline - safety_margin:
        rng = random.Random(42)
        order2 = list(range(total_points))
        rng.shuffle(order2)
        try_prune_fast(order2, max_rounds=2)

    # Strategy 4: Coverage-based order
    if time.time() < deadline - safety_margin:
        coverage = np.zeros(total_points, dtype=np.int32)
        for y in range(total_points):
            if not in_N[y]:
                continue
            for d in range(num_dirs):
                for t in range(q - 1):
                    x = line_points[y, d, t]
                    if is_valid_dir[x, d]:
                        coverage[y] += 1
        order3 = sorted(range(total_points), key=lambda y: 0 if not in_N[y] else coverage[y])
        try_prune_fast(order3, max_rounds=1)

    # Strategy 5: More PQ rounds
    if time.time() < deadline - safety_margin:
        try_prune_pq(max_rounds=3)

    # Strategy 6: Random restarts with fast check
    if time.time() < deadline - safety_margin:
        for seed in [123, 456, 789, 1000, 2000]:
            if time.time() > deadline - safety_margin:
                break
            rng = random.Random(seed)
            order4 = list(range(total_points))
            rng.shuffle(order4)
            try_prune_fast(order4, max_rounds=1)

    # Final validity check
    if np.any(valid_count == 0):
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
