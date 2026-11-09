# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 620
    p = 7
    q = p * p

    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    def f_add(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa + ya) % p, (xb + yb) % p)

    def f_sub(x, y):
        return f_add(x, f_neg(y))

    def f_neg(x):
        xa, xb = to_pair(x)
        return to_int((-xa) % p, (-xb) % p)

    def f_mul(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        real = (xa * ya + xb * yb * 3) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    def f_inv(x):
        if x == 0:
            return 0
        xa, xb = to_pair(x)
        denom = (xa * xa - xb * xb * 3) % p
        if denom == 0:
            return 0
        denom_inv = pow(denom, p - 2, p)
        real = (xa * denom_inv) % p
        imag = (-xb * denom_inv) % p
        return to_int(real, imag)

    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)

    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}
    idx_to_point = {v: k for k, v in point_index.items()}

    in_N = np.ones(q * q, dtype=bool)

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    excluded_count = 0
    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * 3) % p
            if (trace_x + norm_y) % p == 0:
                if y_idx not in y_special:
                    pt = point_index[(x_idx, y_idx)]
                    in_N[pt] = False
                    excluded_count += 1

    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(q):
            for y_idx in range(q):
                pt = point_index[(x_idx, y_idx)]
                count = 0
                for t in range(1, q):
                    tx = f_add(x_idx, f_mul(t, vx))
                    ty = f_add(y_idx, f_mul(t, vy))
                    t_pt = point_index[(tx, ty)]
                    if not in_N[t_pt]:
                        count += 1
                not_in_N_count[pt, d_idx] = count

    cover_count = np.array([np.sum(not_in_N_count[i] == 0) for i in range(q * q)], dtype=np.int32)

    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tx = f_sub(px, f_mul(t, vx))
                ty = f_sub(py, f_mul(t, vy))
                x_idx = point_index[(tx, ty)]
                affects[p_idx].append((x_idx, d_idx))

    # Multi-pass greedy pruning with improved heap management
    max_passes = 10
    for pass_num in range(max_passes):
        if time.time() > deadline - 60:
            break

        n_indices = [i for i in range(q * q) if in_N[i]]
        if not n_indices:
            break

        # Build priority queue: sort by cover_count descending (most redundant first)
        heap = []
        heap_set = set()
        for i in n_indices:
            heap.append((-cover_count[i], i))
            heap_set.add(i)
        heapq.heapify(heap)

        removed_this_pass = 0
        while heap:
            if time.time() > deadline - 30:
                break

            neg_cc, p_idx = heapq.heappop(heap)
            if p_idx not in heap_set:
                continue
            if not in_N[p_idx]:
                heap_set.discard(p_idx)
                continue

            # Try to remove this point
            in_N[p_idx] = False
            affected_counts = []
            for (x, d) in affects[p_idx]:
                old_val = not_in_N_count[x, d]
                not_in_N_count[x, d] = old_val + 1
                affected_counts.append((x, d, old_val))

            invalid = False
            new_cover_counts = {}
            for (x, d, old_val) in affected_counts:
                if old_val == 0:
                    if x not in new_cover_counts:
                        new_cover_counts[x] = cover_count[x]
                    new_cover_counts[x] -= 1
                    if new_cover_counts[x] <= 0:
                        invalid = True
                        break

            if invalid:
                for (x, d, old_val) in affected_counts:
                    not_in_N_count[x, d] = old_val
                in_N[p_idx] = True
                # Don't add back to heap - it's not removable right now
            else:
                removed_this_pass += 1
                heap_set.discard(p_idx)
                for (x, d, old_val) in affected_counts:
                    if old_val == 0:
                        cover_count[x] -= 1
                        # Add affected points back to heap if they're still in N
                        if in_N[x] and x in heap_set:
                            heapq.heappush(heap, (-cover_count[x], x))

        if removed_this_pass == 0:
            break

    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                if in_N[point_index[(x_idx, y_idx)]]:
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
