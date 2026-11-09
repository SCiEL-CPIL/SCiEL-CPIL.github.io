# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    Uses a structural change: product of independent sets from E8 and D3 lattices,
    plus greedy expansion.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    d = 11

    # ---------------------------------------------------------------
    # Step 1: Build E8 lattice points (240 points, all norm^2 = 8)
    # ---------------------------------------------------------------
    def build_e8_minimal():
        pts = []
        # Type 1: (±2, ±2, 0, ..., 0)
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)

        # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of -1's
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                pts.append(np.array(signs, dtype=np.int64))

        return pts

    e8_pts = build_e8_minimal()  # 240 points in R^8, all norm^2 = 8

    # ---------------------------------------------------------------
    # Step 2: Build D3 lattice minimal vectors (12 points, norm^2 = 8)
    # ---------------------------------------------------------------
    def build_d3_minimal():
        pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        return pts

    d3_pts = build_d3_minimal()  # 12 points in R^3, all norm^2 = 8

    # ---------------------------------------------------------------
    # Step 3: Find independent sets in E8 and D3 graphs
    # where edges connect points with dist^2 = 8 (i.e., dot = 4)
    # For the product to have min_dist^2 >= 16, we need:
    #   For E8 component: min dist^2 within selected set >= 16
    #   For D3 component: min dist^2 within selected set >= 16
    # Then product points (a,b) have norm^2 = ||a||^2 + ||b||^2 = 8+8 = 16
    # And dist^2 between (a,b) and (a',b') = ||a-a'||^2 + ||b-b'||^2 >= 16+16 = 32 >= 16 ✓
    # ---------------------------------------------------------------
    
    def find_independent_set(points, min_dist_sq=16):
        """Find a large independent set where all pairs have dist^2 >= min_dist_sq."""
        n = len(points)
        # Build conflict graph: edge if dist^2 < min_dist_sq
        # Use greedy: repeatedly pick point with fewest conflicts, remove its neighbors
        remaining = set(range(n))
        selected = []
        
        # Precompute conflicts for each point
        conflicts = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist_sq = int(np.sum(diff ** 2))
                if dist_sq < min_dist_sq:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        # Greedy: pick point with minimum degree among remaining
        while remaining and time.time() < deadline - 60:
            # Find point with fewest conflicts in remaining
            best_point = None
            best_degree = float('inf')
            for p in remaining:
                deg = len(conflicts[p] & remaining)
                if deg < best_degree:
                    best_degree = deg
                    best_point = p
            
            if best_point is None:
                break
            
            selected.append(best_point)
            # Remove best_point and its neighbors from remaining
            remaining.discard(best_point)
            for neighbor in conflicts[best_point]:
                remaining.discard(neighbor)
        
        return [points[i] for i in selected]

    # Find independent sets
    e8_indep = find_independent_set(e8_pts, min_dist_sq=16)
    d3_indep = find_independent_set(d3_pts, min_dist_sq=16)

    # ---------------------------------------------------------------
    # Step 4: Build product set
    # Points are (e8_point, d3_point) embedded in R^11
    # All have norm^2 = 8 + 8 = 16
    # Min dist^2 >= 16 + 16 = 32 >= 16 ✓
    # ---------------------------------------------------------------
    product_points = []
    for e8v in e8_indep:
        for d3v in d3_indep:
            v11 = np.zeros(11, dtype=np.int64)
            v11[:8] = e8v
            v11[8:] = d3v
            product_points.append(v11)

    best_points = product_points[:]
    best_k = len(product_points)

    # ---------------------------------------------------------------
    # Step 5: Also try the union approach as a fallback/comparison
    # Union of E8 (in first 8 coords) and D3 (in last 3 coords)
    # All norm^2 = 8, min dist^2 = 8
    # ---------------------------------------------------------------
    union_points = []
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        union_points.append(v11)
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        union_points.append(v11)

    if len(union_points) > best_k:
        best_points = union_points[:]
        best_k = len(union_points)

    # ---------------------------------------------------------------
    # Step 6: Greedy expansion on the best set
    # ---------------------------------------------------------------
    if time.time() < deadline - 120:
        max_norm_sq = 0
        for p in best_points:
            n_sq = int(np.sum(p ** 2))
            if n_sq > max_norm_sq:
                max_norm_sq = n_sq

        # Generate candidates with norm^2 <= max_norm_sq
        coord_range = list(range(-int(np.sqrt(max_norm_sq)) - 1, int(np.sqrt(max_norm_sq)) + 2))

        def generate_candidates(max_norm_sq, d):
            candidates = []

            def gen(idx, current, norm_sq):
                if norm_sq > max_norm_sq:
                    return
                if idx == d:
                    if norm_sq > 0:
                        candidates.append(current.copy())
                    return
                for val in coord_range:
                    new_norm = norm_sq + val * val
                    if new_norm > max_norm_sq:
                        continue
                    current[idx] = val
                    gen(idx + 1, current, new_norm)
                current[idx] = 0

            gen(0, np.zeros(d, dtype=np.int64), 0)
            return candidates

        try:
            candidates = generate_candidates(max_norm_sq, 11)
            base_set = set(tuple(p) for p in best_points)
            candidates = [p for p in candidates if tuple(p) not in base_set]
            candidates.sort(key=lambda p: int(np.sum(p ** 2)))

            current_set = best_points[:]
            for p in candidates:
                if time.time() > deadline - 60:
                    break
                diffs = np.array(current_set) - p[np.newaxis, :]
                dists_sq = np.sum(diffs ** 2, axis=1)
                min_dist = np.min(dists_sq)
                if min_dist >= max_norm_sq:
                    current_set.append(p)

            if len(current_set) > best_k:
                best_points = current_set
                best_k = len(current_set)
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Final verification and return
    # ---------------------------------------------------------------
    points_arr = np.array(best_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)

    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]

    if len(points_arr) < 2:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    k = len(points_arr)
    min_dist_sq = float('inf')
    max_norm_sq = 0

    for i in range(k):
        norm_sq = int(np.sum(points_arr[i] ** 2))
        if norm_sq > max_norm_sq:
            max_norm_sq = norm_sq
        for j in range(i + 1, k):
            diff = points_arr[i] - points_arr[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

    if min_dist_sq < max_norm_sq:
        # Fallback to union
        fallback = []
        for v8 in e8_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[:8] = v8
            fallback.append(v11)
        for v3 in d3_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[8:] = v3
            fallback.append(v11)
        points_arr = np.array(fallback, dtype=np.int64)

    return points_arr

# EVOLVE-BLOCK-END
