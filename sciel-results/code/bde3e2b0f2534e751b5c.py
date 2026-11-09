# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    deadline = time.time() + 590  # safety margin before 630s kill

    # Field F_49 = F_7[alpha], alpha^2 = 3
    k = 3

    def to_int(a, b):
        return a + b * p

    def to_pair(z):
        return (z % p, z // p)

    # Precompute field operation tables
    def add(z1, z2):
        a1, b1 = to_pair(z1)
        a2, b2 = to_pair(z2)
        return to_int((a1 + a2) % p, (b1 + b2) % p)

    def sub(z1, z2):
        a1, b1 = to_pair(z1)
        a2, b2 = to_pair(z2)
        return to_int((a1 - a2) % p, (b1 - b2) % p)

    def mul(z1, z2):
        a1, b1 = to_pair(z1)
        a2, b2 = to_pair(z2)
        return to_int((a1 * a2 + b1 * b2 * k) % p, (a1 * b2 + a2 * b1) % p)

    add_table = np.zeros((q, q), dtype=np.int64)
    sub_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for z1 in range(q):
        for z2 in range(q):
            add_table[z1, z2] = add(z1, z2)
            sub_table[z1, z2] = sub(z1, z2)
            mul_table[z1, z2] = mul(z1, z2)

    def point_idx(x, y):
        return x * q + y

    # Directions: (1, m) for m in 0..48, and (0, 1)
    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    num_dirs = len(directions)  # 50

    # Precompute affected_x: for each point z and direction v, the 49 x indices
    # such that z lies on line x + t*v for t in 1..48
    affected_x = np.zeros((q * q, num_dirs, q - 1), dtype=np.int64)
    for x in range(q):
        for y in range(q):
            z_idx = point_idx(x, y)
            for vi, (vx, vy) in enumerate(directions):
                for t in range(1, q):
                    tvx = mul_table[t, vx]
                    tvy = mul_table[t, vy]
                    px = sub_table[x, tvx]
                    py = sub_table[y, tvy]
                    affected_x[z_idx, vi, t - 1] = point_idx(px, py)

    # Initialize: all points in N
    in_N = np.ones(q * q, dtype=bool)

    # Structured initial exclusion: exclude points on curve tr(x) + N(y) = 0
    # where tr(x) = 2*xa (trace in F_7) and N(y) = ya^2 - 3*yb^2 (norm)
    # Keep only "special" y values (squares times arbitrary) to avoid over-excluding
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k) % p
            if (trace_x + norm_y) % p == 0:
                if y_idx not in y_special:
                    pt = point_idx(x_idx, y_idx)
                    in_N[pt] = False

    # Build not_in_N_count[point, dir] = number of excluded points on line through point in direction dir
    not_in_N_count = np.zeros((q * q, num_dirs), dtype=np.int32)
    for d_idx, (vx, vy) in enumerate(directions):
        for x in range(q):
            for y in range(q):
                pt = point_idx(x, y)
                count = 0
                for t in range(1, q):
                    tx = add_table[x, mul_table[t, vx]]
                    ty = add_table[y, mul_table[t, vy]]
                    t_pt = point_idx(tx, ty)
                    if not in_N[t_pt]:
                        count += 1
                not_in_N_count[pt, d_idx] = count

    # cover_count[point] = number of directions where all other points on line are in N
    cover_count = np.array([np.sum(not_in_N_count[i] == 0) for i in range(q * q)], dtype=np.int32)

    # Greedy exclusion using priority queue
    # Points with higher cover_count are "safer" to exclude (more redundant coverage)
    # Build priority queue: (-cover_count, point_idx)
    pq = []
    for i in range(q * q):
        if in_N[i]:
            heapq.heappush(pq, (-cover_count[i], i))

    excluded_count = 0
    while pq and time.time() < deadline - 30:
        neg_cc, p_idx = heapq.heappop(pq)
        if not in_N[p_idx]:
            continue
        if cover_count[p_idx] == 0:
            continue  # No valid directions, can't exclude

        # Try to exclude p_idx
        # Check if safe: for each (x, d) affected by p_idx, if not_in_N_count[x,d] was 0,
        # then x loses a valid direction. x becomes uncovered if cover_count[x] was 1.
        affected = []
        safe = True
        for d in range(num_dirs):
            for t in range(q - 1):
                x_idx = affected_x[p_idx, d, t]
                if not_in_N_count[x_idx, d] == 0:
                    # This direction becomes invalid for x_idx
                    if cover_count[x_idx] <= 1:
                        safe = False
                        break
                    affected.append((x_idx, d))
            if not safe:
                break

        if not safe:
            continue

        # Exclude p_idx
        in_N[p_idx] = False
        excluded_count += 1

        # Update not_in_N_count and cover_count
        for d in range(num_dirs):
            for t in range(q - 1):
                x_idx = affected_x[p_idx, d, t]
                old_val = not_in_N_count[x_idx, d]
                not_in_N_count[x_idx, d] = old_val + 1
                if old_val == 0:
                    cover_count[x_idx] -= 1
                    if cover_count[x_idx] > 0:
                        heapq.heappush(pq, (-cover_count[x_idx], x_idx))

    # Build result
    points = []
    for x in range(q):
        for y in range(q):
            idx = point_idx(x, y)
            if in_N[idx]:
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])

    result = np.array(points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
