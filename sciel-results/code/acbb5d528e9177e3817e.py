# EVOLVE-BLOCK-START
import numpy as np
import time


def _hv_increment(f, front, ref):
    """Hypervolume contribution of point f w.r.t. current front and ref (3D)."""
    if np.any(f >= ref):
        return 0.0
    # Points in front that are dominated by f (f strictly better in all obj)
    mask = np.all(front < f, axis=1)
    if not np.any(mask):
        return 0.0
    # For 3D: contribution = (ref0-f0)*(ref1-f1)*(ref2-f2) minus overlap with dominated front points
    # Use inclusion: iterate dominated front points, subtract their "shadow" that f covers.
    # Standard incremental HV for 3D:
    # contribution of f = box(f, ref) - sum over front points g dominated by f of box(g, f) ... 
    # but that's not quite right. Use the standard 3D incremental formula.
    # For each front point g that f dominates, f "reclaims" the region between g and f that was 
    # previously not counted because g was on the front. The correct incremental contribution:
    #   delta = (ref0-f0)*(ref1-f1)*(ref2-f2) - sum_{g dominated by f} [ (ref0-f0)*(ref1-f1)*(f2-g2) + ... ]
    # This gets complex. Use a simpler robust approach: compute HV of front+{f} minus HV of front.
    # But that's expensive. For 3D with small front, use the known incremental formula:
    #   delta = (ref0-f0)*(ref1-f1)*(ref2-f2)
    #   for each g in front with f dominates g:
    #       delta -= (ref0-f0)*(ref1-f1)*(f2-g2)   -- no, this double counts
    # Let's use the correct 3D incremental HV:
    #   The new point f adds the box [f, ref] minus the parts already covered by front.
    #   The parts already covered by front within [f,ref] are exactly the boxes [g, f] for g dominated by f? No.
    #   Actually the region [f, ref] is partially covered by front points g where g is dominated by f,
    #   the covered sub-box is [g, f] intersected... 
    # Simplest correct method: recompute HV of the new front (front + f, minus dominated) and subtract old.
    # For 3D with front size up to ~200, HV computation is O(n^2 log n) which is fine.
    return None  # placeholder, we'll recompute


def _hypervolume_3d(points, ref):
    """Exact hypervolume for 3D points w.r.t. ref. Only points strictly better than ref count."""
    pts = np.array(points, dtype=float)
    if len(pts) == 0:
        return 0.0
    # Filter points strictly better than ref in all objectives
    mask = np.all(pts < ref, axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    # Remove dominated points (keep only Pareto front)
    front = _pareto_front(pts)
    if len(front) == 0:
        return 0.0
    # 3D HV: sort by f0, sweep
    order = np.argsort(front[:, 0])
    front = front[order]
    n = len(front)
    # For each point i, it contributes a box in the (f1,f2) plane from (f1_i, f2_i) to (min of next better in f1, ...)
    # Standard 3D HV: 
    #   HV = sum over i of (f0_{i+1} - f0_i) * area_in_f1f2_plane
    # where f0_{n} = ref0, and area is computed from the "skyline" in f1,f2 of points i..n-1
    # More precisely:
    #   HV = sum_{i=0}^{n-1} (ref0 if i==n-1 else f0_{i+1} - f0_i) * A_i
    # where A_i is the area of the union of rectangles [f1_j, ref1] x [f2_j, ref2] for j>=i
    # but only the part not covered by points with smaller f0... 
    # Actually the standard formula: 
    #   HV = sum_{i=0}^{n-1} (f0_{i+1} - f0_i) * S_i
    # where f0_n = ref0, and S_i = area of the union of boxes in (f1,f2) for points i, i+1, ..., n-1
    #   where box for point j is [f1_j, ref1] x [f2_j, ref2]
    # But S_i is the "skyline" area. Compute S_i from the right.
    # S_{n-1} = (ref1 - f1_{n-1}) * (ref2 - f2_{n-1})
    # S_i = S_{i+1} + contribution of point i to the skyline
    # The contribution of point i to the skyline area: 
    #   It adds the rectangle [f1_i, ref1] x [f2_i, ref2] minus the part already covered by S_{i+1}.
    #   Since points are sorted by f0, and we're building the skyline in (f1,f2),
    #   point i is "better" (smaller f1 and/or f2) than some subset.
    #   The new area added = (ref1 - f1_i)*(ref2 - f2_i) - area of overlap with existing skyline.
    #   The existing skyline S_{i+1} consists of points j>i. Point i dominates in f1,f2 the points j>i
    #   where f1_j >= f1_i AND f2_j >= f2_i? No, point i is better if f1_i <= f1_j and f2_i <= f2_j.
    #   The region added by point i is the part of its box not covered by the union of boxes of j>i.
    #   Since all boxes share the same upper-right corner (ref1, ref2), the union is determined by the skyline.
    #   The area of the union of boxes for a set of points = sum over skyline points of (ref1 - f1_k)*(ref2 - f2_k) - overlaps.
    #   For boxes all anchored at (ref1, ref2), the union area = sum over k in skyline of (ref1 - f1_k)*(ref2 - f2_k) 
    #   where skyline is the set of points not dominated by another in the set, but with the property that
    #   for two skyline points, one has smaller f1 and larger f2.
    #   Actually for anchored rectangles, union area = integral. Let's use the standard O(n^2) method.
    # 
    # Let me use a clean O(n^2) 3D HV:
    #   Sort by f0 ascending.
    #   For i from n-1 down to 0:
    #       Compute the area in (f1,f2) of the union of rectangles for points i..n-1.
    #       This is done by collecting all points j>=i, finding their Pareto front in (f1,f2),
    #       and computing the 2D HV of that front w.r.t. (ref1, ref2).
    #   Then HV = sum (f0_{i+1} - f0_i) * area_i, with f0_n = ref0.
    #
    # This is O(n^3) worst case but n is small (~200), so n^3 = 8M, fine.
    
    # Precompute: for each i, the 2D HV of points i..n-1 in (f1,f2)
    areas = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sub = front[i:, 1:]  # f1, f2 for points i..n-1
        areas[i] = _hv_2d(sub, np.array([ref[1], ref[2]]))
    
    hv = 0.0
    for i in range(n):
        f0_next = ref[0] if i == n - 1 else front[i + 1, 0]
        hv += (f0_next - front[i, 0]) * areas[i]
    return hv


def _hv_2d(points, ref2):
    """2D hypervolume for points w.r.t. ref2. Only points strictly better count."""
    pts = np.array(points, dtype=float)
    if len(pts) == 0:
        return 0.0
    mask = np.all(pts < ref2, axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    # Pareto front in 2D
    front = _pareto_front_2d(pts)
    if len(front) == 0:
        return 0.0
    # Sort by f1 ascending
    order = np.argsort(front[:, 0])
    front = front[order]
    n = len(front)
    hv = 0.0
    for i in range(n):
        f1_next = ref2[0] if i == n - 1 else front[i + 1, 0]
        hv += (f1_next - front[i, 0]) * (ref2[1] - front[i, 1])
    return hv


def _pareto_front(points):
    """Return non-dominated points (3D)."""
    n = len(points)
    if n == 0:
        return points
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_dominated[i] = True
                break
    return points[~is_dominated]


def _pareto_front_2d(points):
    """Return non-dominated points (2D)."""
    n = len(points)
    if n == 0:
        return points
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and \
               (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1]):
                is_dominated[i] = True
                break
    return points[~is_dominated]


class Optimizer:
    """Multi-objective optimizer: LHS init + NSGA-II-style with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        def sample():
            return lo + span * np.random.random(dim)

        def sample_lhs(n):
            # Latin Hypercube Sampling
            X = np.zeros((n, dim))
            for d in range(dim):
                perm = np.random.permutation(n)
                X[:, d] = (perm + np.random.random(n)) / n
            return lo + span * X

        # Initial population: LHS for space-filling
        n_init = min(60, self.budget)
        X = sample_lhs(n_init)
        F = np.array([problem(x) for x in X])
        used = n_init

        # Keep all evaluated points (for final HV, all recorded vectors count)
        all_X = list(X)
        all_F = list(F)

        # Main loop: NSGA-II style
        pop_size = min(40, self.budget - used)
        if pop_size < 2:
            # Just use remaining budget for random sampling
            while used < self.budget:
                if time.time() > deadline:
                    break
                x = sample()
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1
            return np.array(all_X) if all_X else X

        # Build initial population for EA
        X_pop = X[:pop_size]
        F_pop = F[:pop_size]

        while used < self.budget:
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = _fast_non_dominated_sort(F_pop)

            # Select parents: pick from front 0 (or best available)
            parent_front = fronts[0] if len(fronts[0]) > 0 else np.arange(len(F_pop))
            p1_idx = parent_front[np.random.randint(len(parent_front))]

            # Crossover: SBX-like (blend)
            if np.random.random() < 0.9:
                parent_front2 = fronts[0] if len(fronts[0]) > 0 else np.arange(len(F_pop))
                p2_idx = parent_front2[np.random.randint(len(parent_front2))]
                beta = np.random.random(dim)
                alpha = 0.5
                x1 = X_pop[p1_idx]
                x2 = X_pop[p2_idx]
                child = np.where(beta < 0.5,
                                  (1 + 2*alpha)*x1 + (1 - 2*alpha)*x2,
                                  (1 + 2*alpha)*x2 + (1 - 2*alpha)*x1) / 2
            else:
                child = X_pop[p1_idx].copy()

            # Mutation: polynomial-like
            if np.random.random() < 0.8:
                eta = 20
                mut = np.random.random(dim)
                delta = (np.random.random(dim) * 2 - 1) / eta
                child = child + delta * span

            child = np.clip(child, lo, hi)
            f_child = problem(child)
            used += 1
            all_X.append(child)
            all_F.append(f_child)

            # Replace worst point in population
            # Use HV contribution heuristic: replace point with lowest HV contribution
            # Simple: replace the most dominated point, or random from dominated
            dominated_mask = np.array([
                np.any(np.all(F_pop <= F_pop[i], axis=1) & np.any(F_pop < F_pop[i], axis=1))
                for i in range(len(F_pop))
            ])
            if np.any(dominated_mask):
                candidates = np.where(dominated_mask)[0]
            else:
                candidates = np.arange(len(F_pop))
            replace_idx = candidates[np.random.randint(len(candidates))]
            X_pop[replace_idx] = child
            F_pop[replace_idx] = f_child

        return np.array(all_X) if all_X else X


def _fast_non_dominated_sort(F):
    """Fast non-dominated sort. Returns list of fronts (each is array of indices)."""
    n = len(F)
    if n == 0:
        return []
    S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
    nd = np.zeros(n, dtype=int)  # nd[i] = number of points dominating i
    fronts = []
    front0 = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                S[i].append(j)
            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                nd[i] += 1
        if nd[i] == 0:
            front0.append(i)
            # rank 0
    fronts.append(np.array(front0))
    k = 0
    while len(fronts[k]) > 0:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                nd[j] -= 1
                if nd[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(np.array(next_front))
            k += 1
        else:
            break
    return fronts
# EVOLVE-BLOCK-END
