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

    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)

    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}
    idx_to_point = {v: k for k, v in point_index.items()}

    in_N = np.ones(q * q, dtype=bool)

    # Initial exclusion based on algebraic property
    # This reduces the initial set size, making the problem easier
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

    # Precompute not_in_N_count
    # not_in_N_count[pt, d] = number of points y != pt on line (pt, d) such that y not in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    # Efficient precomputation of line counts
    # Instead of iterating all points for each direction, we can iterate lines
    # But q=49, n_dir=50. q*q*n_dir*q = 49*49*50*49 ~ 5.9M ops. This is fast enough.
    
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

    cover_count = np.zeros(q * q, dtype=np.int32)
    for i in range(q * q):
        cover_count[i] = np.sum(not_in_N_count[i] == 0)

    # Precompute affects
    # affects[z] = list of (pt, d) such that z is on line (pt, d) and z != pt
    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tx = f_sub(px, f_mul(t, vx))
                ty = f_sub(py, f_mul(t, vy))
                x_idx = point_index[(tx, ty)]
                affects[p_idx].append((x_idx, d_idx))

    # Dynamic Priority Queue (Max-Heap via Min-Heap of negatives)
    heap = []
    for i in range(q * q):
        if in_N[i]:
            heap.append((-cover_count[i], i))
    heapq.heapify(heap)
    
    # To handle stale entries, we keep track of current cover_count
    # and check if the popped value matches the current state.
    # Actually, since cover_count only decreases, if we push a new value,
    # the old value (which is higher) will be popped first.
    # We can just check if in_N[i] is True and if the popped priority matches current cover_count?
    # No, if we push (-new_cc, i), and later pop (-old_cc, i) where old_cc > new_cc,
    # we should ignore the stale one?
    # Wait, we want the MAX cover_count.
    # Heap contains (-cc, i). Smallest -cc is largest cc.
    # If we remove point z, cover_count[pt] decreases.
    # We push (-new_cc, pt).
    # The old entry (-old_cc, pt) is still in the heap.
    # Since new_cc < old_cc, -new_cc > -old_cc.
    # So (-old_cc, pt) will be popped BEFORE (-new_cc, pt).
    # When we pop (-old_cc, pt), we check: is cover_count[pt] == -(-old_cc)? i.e., cover_count[pt] == old_cc?
    # If not, it's stale, skip.
    # If yes, it's the latest? No, if it's stale, it means we already updated it to a lower value.
    # So we should skip if cover_count[pt] != popped_cc.
    
    removed = 0
    while heap:
        if time.time() > deadline - 30:
            break
            
        neg_cc, p_idx = heapq.heappop(heap)
        current_cc = -neg_cc
        
        # Check if this entry is stale
        if not in_N[p_idx]:
            continue
        if cover_count[p_idx] != current_cc:
            continue
            
        # Try to remove p_idx
        in_N[p_idx] = False
        
        affected_pts = set()
        invalid = False
        
        for (pt, d) in affects[p_idx]:
            old_val = not_in_N_count[pt, d]
            not_in_N_count[pt, d] = old_val + 1
            
            if old_val == 0:
                # This direction is no longer good for pt
                cover_count[pt] -= 1
                affected_pts.add(pt)
                
                if cover_count[pt] <= 0:
                    invalid = True
                    break
        
        if invalid:
            # Roll back
            in_N[p_idx] = True
            for (pt, d) in affects[p_idx]:
                not_in_N_count[pt, d] -= 1
                # Note: we don't need to restore cover_count here if we broke early?
                # We modified cover_count for some pts. We need to restore them.
                # Let's track which ones were modified.
                pass
            # Better rollback strategy:
            # We modified not_in_N_count for all (pt, d) in affects[p_idx].
            # We modified cover_count for pts where old_val was 0.
            # Let's re-do the rollback properly.
            for (pt, d) in affects[p_idx]:
                not_in_N_count[pt, d] -= 1
                # We can't easily know if cover_count was decremented for this specific (pt,d)
                # unless we check if not_in_N_count[pt,d] is now 0.
                # If not_in_N_count[pt,d] is 0 after decrement, it was 1 before, so cover_count was decremented.
                if not_in_N_count[pt, d] == 0:
                    cover_count[pt] += 1
        else:
            # Commit
            removed += 1
            # Push updated points to heap
            for pt in affected_pts:
                if in_N[pt]:
                    heapq.heappush(heap, (-cover_count[pt], pt))

    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return all points (valid but score 0? No, score is proportional to complement size.
        # If N is all points, complement is 0. Score 0.
        # But if we failed to remove anything, we should still return a valid set.
        # The initial set is valid.
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                if in_N[point_index[(x_idx, y_idx)]]:
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
