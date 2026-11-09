# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    satisfying the kissing constraints, aiming to maximize k.

    Strategy:
      1. Build a large candidate pool from structured families:
         - D11 lattice minimal vectors (252 points)
         - All ±1 vectors (2048)
         - Sparse vectors with small integer coordinates
      2. Greedy construction with multiple restarts and random shuffles.
      3. Local search / hill-climbing to improve the packing.

    Returns:
        points: np.ndarray of shape (k, 11) with int64 entries.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s deadline, leave margin

    d = 11
    best_points = None
    best_k = 0

    def norm_sq(v):
        return int(np.dot(v, v))

    def dist_sq(a, b):
        diff = a - b
        return int(np.dot(diff, diff))

    def is_valid_set(points):
        """Check if the set of points satisfies all constraints."""
        n = len(points)
        if n < 2:
            return False
        for p in points:
            if np.all(p == 0):
                return False
        max_norm = 0
        for p in points:
            ns = norm_sq(p)
            if ns > max_norm:
                max_norm = ns
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                ds = dist_sq(points[i], points[j])
                if ds < min_dist:
                    min_dist = ds
        return min_dist >= max_norm

    def greedy_build(candidates, max_points, max_time=None):
        """
        Greedy construction: add candidates one by one if they maintain validity.
        Returns the list of added points.
        """
        points = []
        max_norm = 0
        min_dist = float('inf')

        for idx, cand in enumerate(candidates):
            if max_time is not None and time.time() > max_time:
                break
            if len(points) >= max_points:
                break

            # Check zero vector
            if np.all(cand == 0):
                continue

            ns = norm_sq(cand)
            new_max_norm = max(max_norm, ns)

            # Check distance to all existing points
            ok = True
            for p in points:
                ds = dist_sq(cand, p)
                if ds < new_max_norm:
                    ok = False
                    break

            if ok:
                points.append(cand)
                if ns > max_norm:
                    max_norm = ns
                # Update min_dist
                for p in points[:-1]:
                    ds = dist_sq(cand, p)
                    if ds < min_dist:
                        min_dist = ds

        return points

    # --- Build candidate pool ---
    candidates = []

    # Family 1: D11 minimal vectors (weight-2 even-sum vectors)
    # These are vectors in {0, ±1}^11 with even number of non-zero entries
    # Actually, let's use the standard D_n lattice minimal vectors:
    # weight 2 with even sum, or weight 4, etc.
    # For D11, minimal vectors have squared norm 2: pairs of ±1 with even count of nonzeros
    # Let's generate all vectors with exactly 2 non-zero entries (±1) -> C(11,2)*4 = 220
    # Plus vectors with 4 non-zero entries that sum to 0 mod something...
    # Actually, the 252 minimal vectors of D11 ⊕ 2D3... let me just use a broader set.

    # Generate all vectors with coordinates in {-1, 0, 1} and even Hamming weight
    # This gives the D11 lattice minimal vectors (norm 2) and some higher norm ones.
    # Let's be more targeted: use vectors with small coordinates.

    # Family 1: ±1 vectors (all 2^11 = 2048)
    for mask in range(1, 1 << d):
        v = np.zeros(d, dtype=np.int64)
        for i in range(d):
            if mask & (1 << i):
                v[i] = 1
        candidates.append(v)
        candidates.append(-v)

    # Family 2: Vectors with one coordinate = ±2, rest 0
    for i in range(d):
        for sign in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * sign
            candidates.append(v)

    # Family 3: Vectors with two coordinates = ±1, rest 0 (already covered by ±1 family? No, those have norm 2)
    # Already covered.

    # Family 4: Vectors with coordinates in {-2,-1,0,1,2} and small norm
    # Generate random sparse vectors
    rng = np.random.RandomState(42)
    for _ in range(20000):
        v = np.zeros(d, dtype=np.int64)
        # Choose 1-3 random positions
        n_nonzero = rng.randint(1, 4)
        positions = rng.choice(d, size=n_nonzero, replace=False)
        for pos in positions:
            v[pos] = rng.randint(-2, 3)  # -2, -1, 0, 1, 2
        if not np.all(v == 0):
            candidates.append(v)

    # Family 5: Structured D11-like vectors
    # Weight-2 vectors: ±1 in two positions
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for v in candidates:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)

    candidates = unique_candidates

    # --- Greedy construction with multiple restarts ---
    max_search_time = 550  # Leave 50s margin

    # Try different orderings
    best_points = None
    best_k = 0

    n_restarts = 0
    while time.time() < start_time + max_search_time and n_restarts < 50:
        n_restarts += 1

        # Shuffle candidates
        idx = np.arange(len(candidates))
        np.random.shuffle(idx)
        shuffled = [candidates[i] for i in idx]

        # Greedy build
        points = greedy_build(shuffled, len(candidates), max_time=start_time + max_search_time)

        if len(points) > best_k:
            best_k = len(points)
            best_points = points
            # Try local improvement: try to swap or add more
            # Simple local search: try adding candidates not in the set
            improved = True
            while improved and time.time() < start_time + max_search_time:
                improved = False
                point_set = set(tuple(p) for p in points)
                for cand in candidates:
                    if tuple(cand) in point_set:
                        continue
                    # Try adding cand
                    trial = points + [cand]
                    if is_valid_set(trial):
                        points = trial
                        point_set.add(tuple(cand))
                        improved = True
                        if len(points) > best_k:
                            best_k = len(points)
                            best_points = points
                        if time.time() > start_time + max_search_time:
                            break

    # Final validation
    if best_points is not None and len(best_points) >= 2:
        result = np.array(best_points, dtype=np.int64)
        result = np.around(result).astype(np.int64)
        return result
    else:
        # Fallback: the trivial 2-point solution
        return np.array([np.ones(d), -np.ones(d)], dtype=np.int64)


# EVOLVE-BLOCK-END
