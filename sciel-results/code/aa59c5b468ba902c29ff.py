# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7).
    Strategy: Priority-queue greedy with multiple restarts.
    We maintain a set E (excluded points). N = F_49^2 \ E.
    Constraint: For every point x in F_49^2, there exists a direction v
    such that line(x, v) \ {x} is disjoint from E.
    We want to maximize |E|.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_dirs = q + 1  # 50

    deadline = time.time() + 570.0

    # Field arithmetic setup
    k_const = 3
    elem_to_idx = {}
    idx_to_elem = []
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            elem_to_idx[(a, b)] = idx
            idx_to_elem.append((a, b))

    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def f_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        aa = (a1 * a2 + b1 * b2 * k_const) % p
        bb = (a1 * b2 + b1 * a2) % p
        return (aa, bb)

    def point_to_indices(idx):
        return idx // q, idx % q

    def indices_to_point(ix, iy):
        return ix * q + iy

    # Directions
    directions = []
    directions.append(((0, 0), (1, 0)))  # vertical
    for m_idx in range(q):
        m_elem = idx_to_elem[m_idx]
        directions.append(((1, 0), m_elem))

    # Precompute line_other
    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    for pidx in range(total_points):
        xi, yi = point_to_indices(pidx)
        x_elem = idx_to_elem[xi]
        y_elem = idx_to_elem[yi]
        for didx, (dx, dy) in enumerate(directions):
            count = 0
            for t_idx in range(1, q):  # t != 0
                t_elem = idx_to_elem[t_idx]
                tdx = f_mul(t_elem, dx)
                tdy = f_mul(t_elem, dy)
                nx = f_add(x_elem, tdx)
                ny = f_add(y_elem, tdy)
                nxi = elem_to_idx[nx]
                nyi = elem_to_idx[ny]
                npidx = indices_to_point(nxi, nyi)
                line_other[pidx, didx, count] = npidx
                count += 1

    # Precompute affected_by
    affected_by = [[] for _ in range(total_points)]
    for x in range(total_points):
        for v in range(num_dirs):
            for z in line_other[x, v]:
                affected_by[z].append((x, v))

    # State management for local search
    in_E = np.zeros(total_points, dtype=bool)
    c = np.zeros((total_points, num_dirs), dtype=np.int32)
    min_c = np.zeros(total_points, dtype=np.int32)
    count_min = np.ones(total_points, dtype=np.int32)

    def update_min_c(x):
        """Recalculate min_c[x] and count_min[x] from c[x, :]"""
        row = c[x]
        m = np.min(row)
        cnt = np.sum(row == m)
        min_c[x] = m
        count_min[x] = cnt

    def try_add(z):
        """
        Try to add point z to E.
        Returns True if successful, False otherwise.
        Updates state if successful.
        """
        if in_E[z]:
            return False

        # Collect the set of x that might be affected
        affected_x = set()
        for (x, v) in affected_by[z]:
            if c[x, v] == min_c[x]:
                affected_x.add(x)

        # Tentatively apply changes
        changes = []
        for (x, v) in affected_by[z]:
            changes.append((x, v, c[x, v]))
            c[x, v] += 1

        # Check validity for affected x
        valid = True
        for x in affected_x:
            update_min_c(x)
            if min_c[x] > 0:
                valid = False
                break

        if valid:
            in_E[z] = True
            return True
        else:
            # Revert
            for (x, v, old_val) in changes:
                c[x, v] = old_val
                if x in affected_x:
                    update_min_c(x)
            return False

    # Search Loop with priority queue
    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0

    num_restarts = 150  # Adjust based on time

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Reset state
        in_E[:] = False
        c[:] = 0
        min_c[:] = 0
        count_min[:] = 1

        size = 0

        # Priority queue: (cost, z) where cost is the number of affected x with c[x,v] == min_c[x]
        # Lower cost = safer to add
        pq = []
        for z in range(total_points):
            cost = 0
            for (x, v) in affected_by[z]:
                if c[x, v] == min_c[x]:
                    cost += 1
            heapq.heappush(pq, (cost, z))

        # Greedy add using priority queue
        while pq:
            if time.time() > deadline:
                break
            cost, z = heapq.heappop(pq)
            if in_E[z]:
                continue
            if try_add(z):
                size += 1
                # Update priority queue for affected points
                # This is expensive, so we'll just do sequential passes instead
                break

        # Sequential follow-up pass
        order = list(range(total_points))
        random.Random(1000 + restart).shuffle(order)
        for z in order:
            if time.time() > deadline:
                break
            if try_add(z):
                size += 1

        if size > best_size:
            best_size = size
            best_E[:] = in_E

    # Build result
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi, yi = point_to_indices(pidx)
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        for pidx in range(total_points):
            xi, yi = point_to_indices(pidx)
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)

# EVOLVE-BLOCK-END
