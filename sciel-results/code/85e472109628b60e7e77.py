# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    Uses E8 lattice points at norm^2=8 (240 roots) + D3 (12) = 252 base,
    then efficient greedy expansion with precomputed distance matrix.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin before 630s kill

    d = 11

    # ---------------------------------------------------------------
    # Step 1: Build E8 minimal vectors scaled to norm^2 = 8
    # Type 1: (±2, ±2, 0, ..., 0): C(8,2)*4 = 112
    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) even # of -1's: 128
    # Total: 240 points, all norm^2 = 8
    # Min dist^2 between distinct = 8
    # ---------------------------------------------------------------
    def build_e8_scaled():
        pts = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                pts.append(np.array(signs, dtype=np.int64))
        return pts

    # ---------------------------------------------------------------
    # Step 2: Build D3 minimal vectors scaled to norm^2 = 8
    # (±2, ±2, 0) and permutations: 12 points
    # ---------------------------------------------------------------
    def build_d3_scaled():
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

    e8_pts = build_e8_scaled()  # 240 in R^8
    d3_pts = build_d3_scaled()  # 12 in R^3

    # ---------------------------------------------------------------
    # Step 3: Build base 252-point config in R^11
    # E8 embedded in first 8 coords, D3 in last 3 coords
    # All norm^2 = 8, min_dist^2 = 8
    # ---------------------------------------------------------------
    base_points = []
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        base_points.append(v11)
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        base_points.append(v11)

    # Convert to numpy array for efficient operations
    base_arr = np.array(base_points, dtype=np.int64)
    base_set = set(map(tuple, base_points))
    n_base = len(base_points)  # 252

    # ---------------------------------------------------------------
    # Step 4: Generate ALL integer vectors in Z^11 with norm^2 <= 8
    # Using recursive generation with pruning
    # ---------------------------------------------------------------
    def generate_all_candidates(max_norm_sq=8, dim=11):
        """Generate all nonzero integer vectors with 0 < ||v||^2 <= max_norm_sq."""
        results = []
        # Max |coord| = floor(sqrt(max_norm_sq)) = 2
        # But we can prune: if remaining dims can't fit, skip
        
        def gen(idx, current, norm_sq):
            if norm_sq > max_norm_sq:
                return
            if idx == dim:
                if norm_sq > 0:
                    results.append(current.copy())
                return
            # Determine max value for this coordinate
            # Remaining coords after this: dim - idx - 1
            # Each can contribute at most max_norm_sq
            # So current[idx]^2 <= max_norm_sq - norm_sq
            max_val = int(np.sqrt(max_norm_sq - norm_sq))
            for val in range(-max_val, max_val + 1):
                new_norm = norm_sq + val * val
                if new_norm > max_norm_sq:
                    continue
                current[idx] = val
                gen(idx + 1, current, new_norm)
            current[idx] = 0

        gen(0, np.zeros(dim, dtype=np.int64), 0)
        return results

    candidates = generate_all_candidates(8, 11)
    # Filter out those already in base
    candidates = [p for p in candidates if tuple(p) not in base_set]
    
    # Sort by norm^2 ascending (smaller norms are easier to add)
    candidates.sort(key=lambda p: int(np.sum(p ** 2)))
    
    # ---------------------------------------------------------------
    # Step 5: Efficient greedy expansion
    # Key optimization: maintain a numpy array of current points and
    # use vectorized distance computation. Update incrementally.
    # ---------------------------------------------------------------
    current_arr = base_arr.copy()
    current_set = base_set.copy()
    k = n_base
    threshold = 8  # min_dist^2 must be >= 8 (which equals max_norm^2)

    for p in candidates:
        if time.time() > deadline - 30:
            break
        # Compute squared distances from p to all current points
        # ||p - v||^2 = ||p||^2 + ||v||^2 - 2*p.v
        p_norm_sq = int(np.sum(p ** 2))
        # Vectorized: dists_sq = p_norm_sq + norms_sq - 2 * (current_arr @ p)
        dot_products = current_arr @ p  # shape (k,)
        norms_sq = np.sum(current_arr ** 2, axis=1)  # shape (k,)
        dists_sq = p_norm_sq + norms_sq - 2 * dot_products
        
        min_dist = int(np.min(dists_sq))
        if min_dist >= threshold:
            # Add this point
            current_arr = np.vstack([current_arr, p])
            current_set.add(tuple(p))
            k += 1

    best_points = current_arr
    best_k = k

    # ---------------------------------------------------------------
    # Step 6: Try to add points with norm^2 > 8
    # If we add a point with norm^2 = N > 8, then threshold becomes N
    # and we need min_dist^2 >= N for ALL pairs
    # This is much harder. Try a few specific norms.
    # ---------------------------------------------------------------
    # For now, stick with the norm^2 <= 8 set which is valid.
    # The threshold is 8 and all points have norm^2 <= 8, so it's valid.

    # ---------------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------------
    points_arr = np.around(best_points).astype(np.int64)

    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Check no zero vector
    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]

    if len(points_arr) < 2:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Verify the constraint
    k_final = len(points_arr)
    min_dist_sq = float('inf')
    max_norm_sq = 0
    for i in range(k_final):
        n_sq = int(np.sum(points_arr[i] ** 2))
        if n_sq > max_norm_sq:
            max_norm_sq = n_sq
        for j in range(i + 1, k_final):
            diff = points_arr[i] - points_arr[j]
            d_sq = int(np.sum(diff ** 2))
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq

    if min_dist_sq < max_norm_sq:
        # Fallback to base 252
        points_arr = base_arr

    return points_arr

# EVOLVE-BLOCK-END
