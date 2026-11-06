import numpy as np
import random
import time
import sys

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 290  # Safety margin before 300s limit

# Pre-compute coordinates
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

# Pre-compute full squared distance matrix
def _precompute_distances():
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=2).astype(np.int32)

D_grid = _precompute_distances()

def _build_conflict_bitsets():
    """
    Build conflict bitsets: neighbors[p] is a Python int bitmask of points that conflict with p.
    Two points p,q conflict if there exists r such that {p,q,r} is isosceles.
    """
    # For each point p, group other points by squared distance
    dist_groups = [{} for _ in range(NUM_POINTS)]
    for p in range(NUM_POINTS):
        d_col = D_grid[p]
        for q in range(NUM_POINTS):
            if p == q:
                continue
            d = int(d_col[q])
            if d not in dist_groups[p]:
                dist_groups[p][d] = 1 << q
            else:
                dist_groups[p][d] |= (1 << q)
    
    neighbors = [0] * NUM_POINTS
    for p in range(NUM_POINTS):
        for d, mask in dist_groups[p].items():
            if mask.bit_count() >= 2:
                m = mask
                while m:
                    lsb = m & (-m)
                    q = lsb.bit_length() - 1
                    neighbors[q] |= (mask & ~(1 << q))
                    m ^= lsb
    
    for p in range(NUM_POINTS):
        neighbors[p] &= ~(1 << p)
    
    return neighbors

print("Building conflict graph...", file=sys.stderr)
t0 = time.time()
neighbors = _build_conflict_bitsets()
print(f"Done in {time.time()-t0:.2f}s", file=sys.stderr)

def _greedy_iset(order, neighbors):
    selected = 0
    for v in order:
        if (selected & neighbors[v]) == 0:
            selected |= (1 << v)
    return selected

def _local_search(initial_mask, neighbors, time_limit):
    """
    Local search: try to add points, and do 1-for-1 and 1-for-2 swaps.
    """
    selected = initial_mask
    selected_set = set()
    m = selected
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        selected_set.add(v)
        m ^= lsb
    
    all_verts = set(range(NUM_POINTS))
    pool = all_verts - selected_set
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Try to add a point from pool
        for v in list(pool):
            if (selected & neighbors[v]) == 0:
                selected |= (1 << v)
                selected_set.add(v)
                pool.discard(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swap
        for rem in list(selected_set):
            new_sel = selected & ~(1 << rem)
            new_pool = pool | {rem}
            for add in new_pool:
                if add not in new_sel:
                    if (new_sel & neighbors[add]) == 0:
                        # Do the swap
                        selected = new_sel | (1 << add)
                        selected_set.discard(rem)
                        selected_set.add(add)
                        pool.discard(add)
                        pool.add(rem)
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 1-for-2 swap: remove 1 from selected, add 2 from pool
        for rem in list(selected_set):
            new_sel = selected & ~(1 << rem)
            new_pool = list(pool | {rem})
            # Find two points in new_pool that are both compatible with new_sel and with each other
            candidates = [v for v in new_pool if v not in new_sel and (new_sel & neighbors[v]) == 0]
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    v1, v2 = candidates[i], candidates[j]
                    if (neighbors[v1] & (1 << v2)) == 0 and (neighbors[v2] & (1 << v1)) == 0:
                        # Both compatible with each other and with new_sel
                        selected = new_sel | (1 << v1) | (1 << v2)
                        selected_set.discard(rem)
                        selected_set.add(v1)
                        selected_set.add(v2)
                        pool.discard(v1)
                        pool.discard(v2)
                        pool.add(rem)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return selected

def construct():
    start_time = time.time()
    best_mask = 0
    best_size = 0
    
    # Multiple greedy restarts with different orderings
    for restart in range(100):
        if time.time() > DEADLINE - 40:
            break
        
        # Different ordering strategies
        strategy = restart % 4
        if strategy == 0:
            order = list(range(NUM_POINTS))
            random.shuffle(order)
        elif strategy == 1:
            # Row-major
            order = list(range(NUM_POINTS))
        elif strategy == 2:
            # Column-major
            order = [y * N + x for y in range(N) for x in range(N)]
        else:
            # Random
            order = list(range(NUM_POINTS))
            random.shuffle(order)
        
        result = _greedy_iset(order, neighbors)
        size = result.bit_count()
        
        if size > best_size:
            best_size = size
            best_mask = result
        
        # Local search improvement
        if size > 0 and time.time() < DEADLINE - 20:
            improved = _local_search(result, neighbors, time.time() + 3)
            imp_size = improved.bit_count()
            if imp_size > best_size:
                best_size = imp_size
                best_mask = improved
    
    # Convert to list of points
    result_points = []
    m = best_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result_points.append(tuple(points[idx]))
        m ^= lsb
    
    return result_points

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} points")
    pts = np.array(result)
    if len(pts) >= 3:
        d = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2)
        valid = True
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                for k in range(j+1, len(pts)):
                    if d[i,j] == d[i,k] or d[i,j] == d[j,k] or d[i,k] == d[j,k]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        print(f"Valid: {valid}")
