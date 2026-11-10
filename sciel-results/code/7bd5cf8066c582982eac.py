# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    """
    n = 8
    deadline = time.monotonic() + 600.0
    safety_margin = 15.0

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)

    # Precompute: for every pair of points (i, j), the 4D lifted vector
    # (grid[i] - grid[j], grid_sq[i] - grid_sq[j])
    # Shape: (N, N, 4)
    diff3 = grid[:, None, :] - grid[None, :, :]  # (N, N, 3)
    diff4 = np.concatenate([diff3, (grid_sq[:, None] - grid_sq[None, :])[:, :, None]], axis=2)  # (N, N, 4)

    # Precompute determinant lookup for all 4-tuples of distinct grid points
    # det_lookup[a,b,c,d] = det of matrix with rows diff4[a,b], diff4[a,c], diff4[a,d], diff4[a,e]... 
    # Actually we need: for candidate p and 4 existing points q1,q2,q3,q4:
    #   det of 4x4 matrix with rows (q_i - p, |q_i|^2 - |p|^2) for i=1..4
    # This equals det of matrix with rows diff4[q_i, p] for i=1..4
    # So we need det of 4x4 matrix formed by 4 rows from diff4[., p]
    # Precompute: for each p (N choices), and each 4-tuple of other points (N-1 choose 4),
    #   the determinant. That's N * C(N-1,4) = 512 * C(511,4) ≈ 512 * 2.7e9 — too large.
    #
    # Alternative: precompute for each pair (i,j) the 4-vector v_ij = diff4[i,j].
    # Then for candidate p and 4 points q1..q4, det = det([v_{q1,p}, v_{q2,p}, v_{q3,p}, v_{q4,p}])
    # We can precompute all 4x4 determinants for each fixed p.
    # For a fixed p, we have 511 vectors in R^4. We need all C(511,4) determinants.
    # That's ~2.7e9 per p, times 512 p's = way too much.
    #
    # Better approach: precompute the determinant for each 4-tuple of grid points.
    # For points a,b,c,d (all distinct), define:
    #   D[a,b,c,d] = det([grid[a]-grid[d], grid[b]-grid[d], grid[c]-grid[d], 
    #                      |grid[a]|^2-|grid[d]|^2, |grid[b]|^2-|grid[d]|^2, |grid[c]|^2-|grid[d]|^2])
    # Wait, that's 3x3 not 4x4. Let me re-derive.
    #
    # The test: for point p and 4 earlier points q1,q2,q3,q4, form 4x4 matrix:
    #   row i = (q_i - p, |q_i|^2 - |p|^2)  for i=1..4
    # This is a 4x4 matrix. det < 0.5 means invalid.
    #
    # Note: this is equivalent to saying the 5 points p, q1, q2, q3, q4 are co-spherical
    # or co-planar. The determinant of the 4x4 matrix [q_i - p, |q_i|^2 - |p|^2]
    # is related to the "spherical determinant."
    #
    # Key insight: det([q_i - p, |q_i|^2 - |p|^2] for i=1..4) 
    #   = det([q_i, |q_i|^2, 1] for i=0..4) where q_0 = p
    # (up to a sign, by row operations)
    # Actually let's verify: the 5x5 matrix with rows (q_i, |q_i|^2, 1) for i=0..4
    # has determinant 0 iff the 5 points are co-spherical/co-planar.
    # The 4x4 matrix is the Schur complement / minor.
    #
    # Precompute: for every 5-tuple of distinct grid points, the 5x5 determinant.
    # C(512, 5) ≈ 3.5e11 — way too large.
    #
    # Alternative: precompute for every 4-tuple (a,b,c,d) the 4x4 determinant
    #   D4[a,b,c,d] = det of 4x4 matrix with rows (grid[a]-grid[d], |grid[a]|^2-|grid[d]|^2), etc.
    # C(512, 4) ≈ 2.7e9 — still too large to precompute all.
    #
    # So we need a different strategy. Let's use the fact that we're doing greedy:
    # At each step, we have m existing points and test one new point p.
    # We need to check C(m,4) determinants.
    # 
    # Speedup idea: precompute v_{i,p} = (grid[i]-grid[p], |grid[i]|^2-|grid[p]|^2) for all i,p.
    # This is diff4[i,p] which we already have.
    # Then for a given p and set of existing points S (size m), we need to check
    # all 4-subsets of S: det of 4x4 matrix from rows diff4[s_i, p] for s_i in subset.
    #
    # For m up to ~100, C(m,4) ~ 4e6 which is manageable if we vectorize.
    # For m up to ~200, C(m,4) ~ 6.5e8 which is too slow.
    #
    # Better: use the property that det is multilinear. We can compute all C(m,4) dets
    # by: form M = diff4[S, p] which is (m, 4). Then we need all 4x4 minors.
    # np.linalg.det on batched (C,4,4) arrays is fast for moderate C.
    #
    # The real bottleneck is when m is large. Let's optimize the batched determinant.

    def can_add_fast(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        """Check if p can be added to arr without creating 5 co-spherical points."""
        m = len(arr)
        if m < 4:
            return True
        # Build lifted vectors relative to p
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq

        # Check all C(m,4) combinations using batched det
        combos = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)
        if len(combos) == 0:
            return True
        sub = w[combos]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def can_add_batched(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        """Vectorized version for larger m with early termination."""
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq

        combos = list(itertools.combinations(range(m), 4))
        if len(combos) <= 100000:
            sub = w[np.array(combos)]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Chunked processing
            chunk_size = 100000
            for start in range(0, len(combos), chunk_size):
                end = min(start + chunk_size, len(combos))
                sub = w[np.array(combos[start:end])]
                dets = np.linalg.det(sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True

    def greedy_pass(order: np.ndarray, deadline: float, use_batched: bool = True) -> tuple:
        """Run a greedy pass over the given order of points."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        skipped = []

        for p in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
            if use_batched:
                ok = can_add_batched(arr, arr_sq, p, p_sq)
            else:
                ok = can_add_fast(arr, arr_sq, p, p_sq)
            if ok:
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([p_sq])
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append(p)

        return arr, arr_sq, skipped

    def local_improve(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float) -> tuple:
        """Try to insert previously skipped points."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        improved = True
        while improved and (time.monotonic() + safety_margin < deadline):
            improved = False
            for i in range(len(skipped)):
                if time.monotonic() + safety_margin >= deadline:
                    break
                p = skipped[i]
                p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
                if can_add_batched(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                    skipped.pop(i)
                    improved = True
                    break
        return arr, arr_sq, skipped

    def swap_improve(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float) -> tuple:
        """Try swap-based local improvement: remove one point, add two from skipped."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        improved = True
        max_swaps = 50
        swaps_done = 0
        while improved and swaps_done < max_swaps and (time.monotonic() + safety_margin < deadline):
            improved = False
            if not skipped:
                break
            # Try removing each point in arr and see if we can add 2 from skipped
            for i in range(len(arr)):
                if time.monotonic() + safety_margin >= deadline or swaps_done >= max_swaps:
                    break
                # Temporarily remove arr[i]
                arr_temp = np.delete(arr, i, axis=0)
                arr_sq_temp = np.delete(arr_sq, i)
                
                # Try to add 2 points from skipped
                added = []
                remaining = list(skipped)
                for p in remaining:
                    p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
                    if can_add_batched(arr_temp, arr_sq_temp, p, p_sq):
                        arr_temp = np.vstack([arr_temp, p])
                        arr_sq_temp = np.append(arr_sq_temp, p_sq)
                        added.append(p)
                        remaining.remove(p)
                        if len(added) == 2:
                            break
                
                if len(added) >= 2:
                    arr = arr_temp
                    arr_sq = arr_sq_temp
                    for p in added:
                        if p in skipped:
                            skipped.remove(p)
                    improved = True
                    swaps_done += 1
                    break
        return arr, arr_sq, skipped

    def local_improve_multi(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float) -> tuple:
        """More aggressive local improvement: multiple passes with shuffled skipped list."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        rng_local = np.random.default_rng(42)
        
        for _pass in range(30):
            if time.monotonic() + safety_margin >= deadline:
                break
            if not skipped:
                break
            order_skipped = list(skipped)
            rng_local.shuffle(order_skipped)
            improved_this_pass = False
            for p in order_skipped:
                if time.monotonic() + safety_margin >= deadline:
                    break
                p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
                if can_add_batched(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                    skipped.remove(p)
                    improved_this_pass = True
            if not improved_this_pass:
                break
        return arr, arr_sq, skipped

    # Generate diverse orderings
    def make_random_order(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return grid[rng.permutation(N)]

    def make_norm_sorted_order(reverse: bool = False) -> np.ndarray:
        """Sort by squared norm (distance from origin)."""
        idx = np.argsort(grid_sq)
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    def make_corner_biased_order(seed: int) -> np.ndarray:
        """Bias toward corners and edges."""
        rng = np.random.default_rng(seed)
        corner_score = np.zeros(N)
        for i, p in enumerate(grid):
            for c in p:
                if c == 0 or c == 7:
                    corner_score[i] += 1
        noise = rng.random(N) * 0.5
        scores = corner_score + noise
        idx = np.argsort(scores)
        return grid[idx]

    def make_axis_sorted_order(axis: int, reverse: bool = False) -> np.ndarray:
        """Sort by one axis coordinate."""
        idx = np.argsort(grid[:, axis])
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    def make_diagonal_order(seed: int) -> np.ndarray:
        """Sort by sum of coordinates (diagonal distance from corner)."""
        rng = np.random.default_rng(seed)
        diag = grid.sum(axis=1)
        noise = rng.random(N) * 2.0
        scores = diag + noise
        idx = np.argsort(scores)
        return grid[idx]

    best = None
    best_sq = None

    # Phase 1: Quick greedy passes with diverse orderings
    orderings = []
    for seed in range(15):
        orderings.append(('random', make_random_order(seed)))
    orderings.append(('norm_asc', make_norm_sorted_order(False)))
    orderings.append(('norm_desc', make_norm_sorted_order(True)))
    for seed in range(8):
        orderings.append(('corner', make_corner_biased_order(seed)))
    for axis in range(3):
        orderings.append((f'axis{axis}_asc', make_axis_sorted_order(axis, False)))
        orderings.append((f'axis{axis}_desc', make_axis_sorted_order(axis, True)))
    for seed in range(5):
        orderings.append(('diag', make_diagonal_order(seed)))

    for name, order in orderings:
        if time.monotonic() + safety_margin >= deadline:
            break
        arr, arr_sq, skipped = greedy_pass(order, deadline, use_batched=True)
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline)

        if best is None or len(arr) > len(best):
            best = arr
            best_sq = arr_sq

    # Phase 2: More intensive local improvement on the best solution
    if best is not None and time.monotonic() + safety_margin < deadline:
        best_set = set(map(tuple, best))
        all_points = [p for p in grid if tuple(p) not in best_set]
        best, best_sq, _ = local_improve_multi(best, best_sq, all_points, deadline)
    
    # Phase 2.5: Swap-based improvement
    if best is not None and time.monotonic() + safety_margin < deadline:
        best_set = set(map(tuple, best))
        remaining = [p for p in grid if tuple(p) not in best_set]
        best, best_sq, _ = swap_improve(best, best_sq, remaining, deadline)

    # Phase 3: Try additional random restarts if time remains
    extra_restarts = 40
    for i in range(extra_restarts):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = make_random_order(200 + i)
        arr, arr_sq, skipped = greedy_pass(order, deadline, use_batched=True)
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline)

        if best is None or len(arr) > len(best):
            best = arr
            best_sq = arr_sq

    # Phase 4: Final swap improvement on best
    if best is not None and time.monotonic() + safety_margin < deadline:
        best_set = set(map(tuple, best))
        remaining = [p for p in grid if tuple(p) not in best_set]
        best, best_sq, _ = swap_improve(best, best_sq, remaining, deadline)

    if best is None:
        best = np.empty((0, 3), dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
