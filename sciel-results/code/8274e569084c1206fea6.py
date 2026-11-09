# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using greedy removal with
    multiple strategies and a final polish pass.
    """
    deadline = time.time() + 580  # Safety margin before 630s limit

    p = 7
    q = p * p  # 49
    k_field = 3  # alpha^2 = 3

    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_neg = np.zeros(q, dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        f_neg[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)

    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dir = len(directions)

    # All points
    point_index = {}
    all_points = []
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            all_points.append((x_idx, y_idx))

    idx_to_point = [None] * (q * q)
    for pt, idx in point_index.items():
        idx_to_point[idx] = pt

    # Build initial N using Hermitian unital construction
    def build_initial_in_N():
        in_N = np.ones(q * q, dtype=bool)
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(to_int((u * u) % p, v))

        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                norm_y = (ya * ya - yb * yb * k_field) % p
                if (trace_x + norm_y) % p == 0:
                    if y_idx not in y_special:
                        in_N[point_index[(x_idx, y_idx)]] = False
        return in_N

    # Precompute affects mapping: for each point r_idx, list of (p_idx, d_idx)
    # where p_idx is a point on the line through r_idx in direction d_idx
    # (excluding r_idx itself)
    affects = [[] for _ in range(q * q)]
    for r_idx in range(q * q):
        rx, ry = idx_to_point[r_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tv_x = f_mul[t, vx]
                tv_y = f_mul[t, vy]
                px = f_add[rx, tv_x]
                py = f_add[ry, tv_y]
                if (px, py) in point_index:
                    p_idx = point_index[(px, py)]
                    affects[r_idx].append((p_idx, d_idx))

    # Precompute reverse affects: for each (p_idx, d_idx), list of r_idx that affect it
    rev_affects = [dict() for _ in range(q * q)]
    for r_idx in range(q * q):
        for p_idx, d_idx in affects[r_idx]:
            if d_idx not in rev_affects[p_idx]:
                rev_affects[p_idx][d_idx] = []
            rev_affects[p_idx][d_idx].append(r_idx)

    def compute_not_in_N_count(in_N):
        """Compute not_in_N_count[p_idx, d_idx] for given in_N."""
        not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)
        for d_idx, (vx, vy) in enumerate(directions):
            t_v = []
            for t in range(1, q):
                tx = f_mul[t, vx]
                ty = f_mul[t, vy]
                t_v.append((tx, ty))

            for p_idx in range(q * q):
                px, py = idx_to_point[p_idx]
                count = 0
                for tx, ty in t_v:
                    nx = f_add[px, tx]
                    ny = f_add[py, ty]
                    n_idx = point_index[(nx, ny)]
                    if not in_N[n_idx]:
                        count += 1
                not_in_N_count[p_idx, d_idx] = count
        return not_in_N_count

    def greedy_removal(in_N, not_in_N_count, cover_count, order='random'):
        """
        Greedy removal with given order.
        Returns (in_N, not_in_N_count, cover_count, removed_count).
        """
        in_N = in_N.copy()
        not_in_N_count = not_in_N_count.copy()
        cover_count = cover_count.copy()

        # Create removal order
        indices = list(range(q * q))
        if order == 'random':
            random.shuffle(indices)
        elif order == 'cost':
            # Sort by initial removal cost (number of points whose cover would be reduced)
            costs = np.zeros(q * q, dtype=np.int32)
            for r_idx in indices:
                if in_N[r_idx]:
                    cost = 0
                    for p_idx, d_idx in affects[r_idx]:
                        if not_in_N_count[p_idx, d_idx] == 0:
                            cost += 1
                    costs[r_idx] = cost
            indices.sort(key=lambda i: costs[i] if in_N[i] else 10**9)
        elif order == 'reverse_cost':
            costs = np.zeros(q * q, dtype=np.int32)
            for r_idx in indices:
                if in_N[r_idx]:
                    cost = 0
                    for p_idx, d_idx in affects[r_idx]:
                        if not_in_N_count[p_idx, d_idx] == 0:
                            cost += 1
                    costs[r_idx] = cost
            indices.sort(key=lambda i: -costs[i] if in_N[i] else -10**9)

        removed_count = 0
        start_time = time.time()

        for r_idx in indices:
            if time.time() - start_time > 120:
                break
            if not in_N[r_idx]:
                continue

            # Check if removal is valid
            invalid = False
            affected = []

            for p_idx, d_idx in affects[r_idx]:
                old_count = not_in_N_count[p_idx, d_idx]
                if old_count == 0:
                    if cover_count[p_idx] == 1:
                        invalid = True
                        break
                affected.append((p_idx, d_idx, old_count))

            if invalid:
                continue

            # Commit removal
            for p_idx, d_idx, old_count in affected:
                not_in_N_count[p_idx, d_idx] = old_count + 1
                if old_count == 0:
                    cover_count[p_idx] -= 1

            in_N[r_idx] = False
            removed_count += 1

        return in_N, not_in_N_count, cover_count, removed_count

    def polish_pass(in_N, not_in_N_count, cover_count, max_rounds=10):
        """
        Polish pass: repeatedly try to remove points in random order.
        """
        in_N = in_N.copy()
        not_in_N_count = not_in_N_count.copy()
        cover_count = cover_count.copy()

        for _ in range(max_rounds):
            if time.time() > deadline - 30:
                break

            progress = False
            indices = list(range(q * q))
            random.shuffle(indices)

            for r_idx in indices:
                if not in_N[r_idx]:
                    continue

                invalid = False
                affected = []

                for p_idx, d_idx in affects[r_idx]:
                    old_count = not_in_N_count[p_idx, d_idx]
                    if old_count == 0:
                        if cover_count[p_idx] == 1:
                            invalid = True
                            break
                    affected.append((p_idx, d_idx, old_count))

                if invalid:
                    continue

                for p_idx, d_idx, old_count in affected:
                    not_in_N_count[p_idx, d_idx] = old_count + 1
                    if old_count == 0:
                        cover_count[p_idx] -= 1

                in_N[r_idx] = False
                progress = True

            if not progress:
                break

        return in_N, not_in_N_count, cover_count

    # Run multiple strategies
    best_in_N = None
    best_size = q * q + 1

    # Strategy 1: Random order greedy
    in_N_init = build_initial_in_N()
    not_in_N_count_init = compute_not_in_N_count(in_N_init)
    cover_count_init = np.sum(not_in_N_count_init == 0, axis=1).astype(np.int32)

    in_N_r, _, _, _ = greedy_removal(in_N_init, not_in_N_count_init, cover_count_init, order='random')
    in_N_r, _, _ = polish_pass(in_N_r, compute_not_in_N_count(in_N_r),
                               np.sum(compute_not_in_N_count(in_N_r) == 0, axis=1).astype(np.int32))
    size_r = np.sum(in_N_r)
    if size_r < best_size:
        best_size = size_r
        best_in_N = in_N_r.copy()

    # Strategy 2: Cost order greedy
    in_N_init = build_initial_in_N()
    not_in_N_count_init = compute_not_in_N_count(in_N_init)
    cover_count_init = np.sum(not_in_N_count_init == 0, axis=1).astype(np.int32)

    in_N_c, _, _, _ = greedy_removal(in_N_init, not_in_N_count_init, cover_count_init, order='cost')
    in_N_c, _, _ = polish_pass(in_N_c, compute_not_in_N_count(in_N_c),
                               np.sum(compute_not_in_N_count(in_N_c) == 0, axis=1).astype(np.int32))
    size_c = np.sum(in_N_c)
    if size_c < best_size:
        best_size = size_c
        best_in_N = in_N_c.copy()

    # Strategy 3: Reverse cost order greedy
    in_N_init = build_initial_in_N()
    not_in_N_count_init = compute_not_in_N_count(in_N_init)
    cover_count_init = np.sum(not_in_N_count_init == 0, axis=1).astype(np.int32)

    in_N_rc, _, _, _ = greedy_removal(in_N_init, not_in_N_count_init, cover_count_init, order='reverse_cost')
    in_N_rc, _, _ = polish_pass(in_N_rc, compute_not_in_N_count(in_N_rc),
                                np.sum(compute_not_in_N_count(in_N_rc) == 0, axis=1).astype(np.int32))
    size_rc = np.sum(in_N_rc)
    if size_rc < best_size:
        best_size = size_rc
        best_in_N = in_N_rc.copy()

    # Strategy 4: Multiple random restarts
    for restart in range(3):
        if time.time() > deadline - 60:
            break

        in_N_init = build_initial_in_N()
        not_in_N_count_init = compute_not_in_N_count(in_N_init)
        cover_count_init = np.sum(not_in_N_count_init == 0, axis=1).astype(np.int32)

        in_N_res, _, _, _ = greedy_removal(in_N_init, not_in_N_count_init, cover_count_init, order='random')
        in_N_res, _, _ = polish_pass(in_N_res, compute_not_in_N_count(in_N_res),
                                     np.sum(compute_not_in_N_count(in_N_res) == 0, axis=1).astype(np.int32))
        size_res = np.sum(in_N_res)
        if size_res < best_size:
            best_size = size_res
            best_in_N = in_N_res.copy()

    # Final polish on best result
    if best_in_N is not None:
        not_in_N_count_final = compute_not_in_N_count(best_in_N)
        cover_count_final = np.sum(not_in_N_count_final == 0, axis=1).astype(np.int32)
        best_in_N, _, _ = polish_pass(best_in_N, not_in_N_count_final, cover_count_final, max_rounds=20)

    # Build result from best_in_N
    if best_in_N is None:
        best_in_N = build_initial_in_N()

    points = []
    for i in range(q * q):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        # Fallback
        best_in_N = build_initial_in_N()
        points = []
        for i in range(q * q):
            if best_in_N[i]:
                x, y = idx_to_point[i]
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
