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

    # Precompute field multiplication table for speed
    mul_table = np.zeros((q, q), dtype=np.int32)
    for a in range(q):
        for b in range(q):
            mul_table[a, b] = f_mul(a, b)

    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)

    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}
    idx_to_point = {v: k for k, v in point_index.items()}

    in_N = np.ones(q * q, dtype=bool)

    # Initial exclusion: remove points where trace(x) + norm(y) == 0 and y not special
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

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

    # Precompute line membership: for each point and direction, which points are on the line
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(q):
            for y_idx in range(q):
                pt = point_index[(x_idx, y_idx)]
                count = 0
                for t in range(1, q):
                    tx = f_add(x_idx, mul_table[t, vx])
                    ty = f_add(y_idx, mul_table[t, vy])
                    t_pt = point_index[(tx, ty)]
                    if not in_N[t_pt]:
                        count += 1
                not_in_N_count[pt, d_idx] = count

    # cover_count[pt] = number of directions for which pt has a valid line (all other points in N)
    cover_count = np.array([np.sum(not_in_N_count[i] == 0) for i in range(q * q)], dtype=np.int32)

    # affects[pt] = list of (other_pt, d_idx) pairs: when pt is removed, these (other_pt, d) pairs
    # gain one more "not in N" point
    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tx = f_sub(px, mul_table[t, vx])
                ty = f_sub(py, mul_table[t, vy])
                x_idx = point_index[(tx, ty)]
                affects[p_idx].append((x_idx, d_idx))

    def try_remove_point(p_idx):
        """Try to remove point p_idx. Returns True if removal was successful."""
        if not in_N[p_idx]:
            return False

        in_N[p_idx] = False
        affected_counts = []
        for (x, d) in affects[p_idx]:
            old_val = not_in_N_count[x, d]
            not_in_N_count[x, d] = old_val + 1
            affected_counts.append((x, d, old_val))

        # Check if any point loses all its cover
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
            # Revert
            for (x, d, old_val) in affected_counts:
                not_in_N_count[x, d] = old_val
            in_N[p_idx] = True
            return False
        else:
            # Commit removal
            for (x, d, old_val) in affected_counts:
                if old_val == 0:
                    cover_count[x] -= 1
            return True

    # Priority queue greedy: sort by cover_count descending (most redundant first)
    # Use a heap with negative cover_count for max-heap behavior
    n_indices = [i for i in range(q * q) if in_N[i]]
    
    # Multi-pass with priority queue
    max_passes = 12
    for pass_num in range(max_passes):
        if time.time() > deadline - 60:
            break

        # Build priority queue: lower cover_count first (cheapest to remove)
        # Actually we want to remove points with HIGHEST cover_count first (most redundant)
        # But the lesson says "fewest cover_count reductions" - let me re-read
        # "Priority-queue greedy always removes the 'cheapest' point (fewest cover_count reductions)"
        # This means points that, when removed, cause the fewest other points to lose coverage
        # That's different from highest cover_count. Let me think...
        
        # Actually, a point with high cover_count is more "protected" - removing it is safer
        # because it has more alternative lines. So we should remove high cover_count first.
        
        heap = []
        for i in n_indices:
            if in_N[i]:
                heapq.heappush(heap, (-cover_count[i], i))
        
        if not heap:
            break

        removed_this_pass = 0
        while heap:
            if time.time() > deadline - 30:
                break
            
            neg_cc, p_idx = heapq.heappop(heap)
            if not in_N[p_idx]:
                continue
            if cover_count[p_idx] == 0:
                continue

            if try_remove_point(p_idx):
                removed_this_pass += 1
                # Update heap for affected points
                for (x, d, old_val) in affects[p_idx]:
                    if old_val == 0 and cover_count[x] > 0:
                        heapq.heappush(heap, (-cover_count[x], x))

        if removed_this_pass == 0:
            break

    # Build result
    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return all points
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
