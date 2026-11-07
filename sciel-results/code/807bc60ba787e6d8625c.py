"""SEARCH approach: multiple diverse random restarts with local refinement, keep best hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using a search-based approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])  # reference point from scoring

        # Track all evaluations
        self._eval_count = 0
        self._max_budget = self.budget

        def evaluate(x):
            """Evaluate a single point, tracking budget."""
            self._eval_count += 1
            if self._eval_count > self._max_budget:
                raise ValueError("Budget exceeded")
            return np.array(problem(x))

        def evaluate_budget(x, budget_ref):
            """Evaluate with explicit budget tracking via mutable list."""
            self._eval_count += 1
            if self._eval_count > self._max_budget:
                raise ValueError("Budget exceeded")
            budget_ref[0] += 1
            return np.array(problem(x))

        def time_ok():
            return time.time() < deadline

        def hypervolume_3d(points, ref):
            """Compute 3D hypervolume of non-dominated points with given reference."""
            # Filter points that are strictly better than reference in all objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0

            # Remove dominated points
            n = len(pts)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j or not non_dom[j]:
                        continue
                    if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                        non_dom[i] = False
                        break
            pts = pts[non_dom]
            n = len(pts)
            if n == 0:
                return 0.0

            # Compute 3D hypervolume using inclusion-exclusion / sweep
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]

            hv = 0.0
            for i in range(n):
                # For each point, compute the volume of the box it uniquely contributes
                # Box: [pts[i][0], pts[i-1][0]] x [pts[i][1], pts[i-1][1]] x [pts[i][2], pts[i-1][2]]
                # Actually, need to be more careful. Use the standard O(n^2) method for 3D.
                pass

            # Simpler approach: for each point, the contribution is
            # (ref[0] - pts[i][0]) if no other point dominates it in the relevant region
            # Use the standard incremental method
            # Sort by objective 0
            pts_sorted = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            for i in range(len(pts_sorted)):
                p = pts_sorted[i]
                # The width in obj0 that this point uniquely covers
                if i == 0:
                    width0 = ref[0] - p[0]
                else:
                    width0 = p[0] - pts_sorted[i-1][0]
                    if width0 < 0:
                        width0 = 0

                # For obj1 and obj2, find the max range not covered by other points
                # A point j (j < i, so pts_sorted[j][0] <= p[0]) "blocks" part of p's volume
                # if pts_sorted[j][1] < p[1] and pts_sorted[j][2] < p[2]
                # The blocked region in obj1 is [pts_sorted[j][1], min(ref[1], p[1])]
                # and in obj2 is [pts_sorted[j][2], min(ref[2], p[2])]
                # This gets complex. Let's use a simpler but correct method.

            # Alternative: use the O(n^2) 3D hypervolume algorithm
            # For each point, compute the box it "owns"
            # Sort by obj0 ascending
            idx = np.argsort(pts[:, 0])
            pts_s = pts[idx]
            hv = 0.0
            for i in range(len(pts_s)):
                p = pts_s[i]
                # Left boundary in obj0
                if i > 0:
                    left0 = pts_s[i-1][0]
                else:
                    left0 = p[0]
                right0 = ref[0]
                width0 = right0 - left0
                if width0 <= 0:
                    continue

                # Now compute 2D area in obj1-obj2 plane for the box [p[1], ref[1]] x [p[2], ref[2]]
                # minus regions covered by other points that have obj0 in [left0, p[0]]
                # i.e., points j with pts_s[j][0] >= left0 and pts_s[j][0] < p[0]
                # and pts_s[j][1] < p[1] and pts_s[j][2] < p[2]
                # Actually, any point with obj0 < p[0] (i.e., j < i) that also has
                # obj1 < p[1] and obj2 < p[2] will block part of the volume.

                # The 2D area of [p[1], ref[1]] x [p[2], ref[2]] is (ref[1]-p[1])*(ref[2]-p[2])
                # Subtract areas blocked by points j < i with pts_s[j][1] < p[1] and pts_s[j][2] < p[2]
                # But blocked regions may overlap, so we need union of rectangles.

                # For simplicity and correctness, use the standard algorithm:
                # The contribution of point i is:
                # (p[0] - prev_max[0]) * area_2d where area_2d accounts for blocking

                # Let's just use a direct computation: for each point, compute the box
                # [p[0], prev[0]] x [p[1], ref[1]] x [p[2], ref[2]] but subtract
                # overlaps with previously processed points.

                # Given complexity, let's use a different approach:
                # Brute force for small n (n <= ~50) using the fact that RE37
                # will have at most a few dozen non-dominated points.

            # Use a clean O(n^2) or O(n^3) hypervolume for 3D
            return self._hv3d(pts, ref)

        def _hv3d(points, ref):
            """3D hypervolume using the standard O(n^2 log n) or O(n^3) method."""
            n = len(points)
            if n == 0:
                return 0.0

            # Sort by first objective
            idx = np.argsort(points[:, 0])
            pts = points[idx]

            total = 0.0
            for i in range(n):
                p = pts[i]
                # Width in obj0: from p[0] to the obj0 of the previous point (or ref if none)
                # Actually, the contribution of point i in the "sweep" is:
                # The box [p[0], next_left] where next_left is min of obj0 of points that
                # don't dominate p in obj1 and obj2... this is getting complicated.

                # Let's use the inclusion-exclusion with sorted points.
                # The hypervolume = sum over all points of the volume of the box
                # [p[0], p_prev[0]] x [p[1], p_max1] x [p[2], p_max2]
                # where p_max1 = max of obj1 among points j with obj0 in [p[0], p_prev[0]]
                #   and obj1 < p[1], or ref[1] if none
                # This is still complex.

                # Use a verified simple method: for each point, compute the
                # "owned" box by checking all other points.
                # A point q is in the owned box of p if:
                # p[0] <= q[0] < next[0] (where next is the next point in sorted order with obj0 > p[0])
                # AND p[1] <= q[1] AND p[2] <= q[2]
                # The owned box for p is [p[0], next0] x [p[1], ref[1]] x [p[2], ref[2]]
                # MINUS the parts covered by other points.

                # Given the complexity, let's just use a Monte Carlo or exact
                # box decomposition. For n up to ~50, O(n^2) is fine.

                # Actually, let me use a known correct algorithm:
                # Sort by obj0. For each point i, the contribution is:
                # (p_i[0] - p_{i-1}[0]) * A_i
                # where A_i is the 2D hypervolume of the set of points {p_j : j <= i, p_j[1] < p_i[1], p_j[2] < p_i[2]}
                # projected onto obj1-obj2, with reference (ref[1], ref[2]).
                # Wait, that's not quite right either.

                # Let me use a different approach: enumerate all boxes.
                # For a set of non-dominated points in 3D, the hypervolume can be
                # computed by sorting and using a 2D hypervolume for each "slice".

            # I'll use a simpler but correct O(n^2) method for 3D:
            # For each point p, compute the volume of the box [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]]
            # that is NOT covered by any other point's box.
            # This is equivalent to: for each p, find the maximum region in the
            # positive orthant from p that is not dominated by any other point.

            # Given the small n, let's use a grid-based or exact method.
            # Actually, the simplest correct approach for n <= 50:
            # Use the fact that the hypervolume = integral over the region.
            # We can compute it by: for each point, the "unique" contribution is
            # the box from p to the "envelope" formed by other points.

            # Let me just implement a standard algorithm properly.

            # Standard 3D hypervolume (O(n^2)):
            # Sort by x1. For each point i (in order), compute:
            #   dx = x1[i] - x1[i-1] (or x1[i] - x1[0] if i=0... no)
            # Actually, I recall the algorithm:
            # Sort by first objective. Initialize hv=0.
            # For i from 0 to n-1:
            #   Let S = {j : j < i, x1[j] < x1[i], x2[j] < x2[i], x3[j] < x3[i]}
            #   The "shadow" of point i is blocked by points in S.
            #   The 2D area of the union of boxes [x2[j], ref2] x [x3[j], ref3] for j in S
            #   is subtracted from (ref2-x2[i])*(ref3-x3[i]).
            #   dv = (x1[i] - x1[i-1]) * remaining_area  ... no, this isn't right.

            # I'll use a well-known method: enumerate all rectangles in the 2D projection
            # for each slice. Given n <= 50, even O(n^3) is fine.

            # Actually, let me just use a simple but correct method:
            # The hypervolume of a set of points in 3D with reference r is:
            # sum over all "cells" in the arrangement of slabs.
            # For small n, we can use the following:
            # Sort points by x1. The hypervolume = sum_{i} (x1[i+1] - x1[i]) * A_i
            # where A_i is the 2D hypervolume of the projection of points with x1 <= x1[i]
            # onto the x2-x3 plane, with reference (ref[2], ref[3]).
            # Wait, I need to be careful about the indexing.

            # Let me use a different, verifiably correct approach:
            # For each point, compute the box it uniquely owns.
            # Point p owns the box [p[0], b0] x [p[1], b1] x [p[2], b2] where
            # b0 = min(ref[0], min_{j: not dominated by p} (some boundary))
            # This is still complex.

            # Given time constraints, let me use a Monte Carlo estimate for hypervolume
            # when n is small, or just use a simple box sum approximation.
            # Actually, for scoring purposes, I can just return the sum of
            # (ref[0]-p[0])*(ref[1]-p[1])*(ref[2]-p[2]) for non-dominated points,
            # which overestimates but is a reasonable proxy. No, that's too inaccurate.

            # Let me implement the O(n^2) 3D hypervolume properly.
            # Reference: "Computing the Hypervolume Indicator" by Zitzler, Thiele, Laumanns, Fonseca, Da Fonseca

            # For 3 objectives, the algorithm is:
            # 1. Sort by f1.
            # 2. For each point i, compute the 2D hypervolume of the set of points
            #    {j : j < i} that are not dominated by i in f2 and f3.
            #    Actually, the contribution of point i is:
            #    (f1[i] - f1[i-1]) * HV_2D({j < i : f2[j] > f2[i] and f3[j] > f3[i]}, ref_2d)
            #    where HV_2D is the 2D hypervolume of those points in the f2-f3 plane.
            #    Wait, I think it's the opposite: points j with f2[j] < f2[i] and f3[j] < f3[i]
            #    block part of i's volume.

            # I'll use a brute-force O(n^2) per-point check for the 2D area.

            idx = np.argsort(points[:, 0])
            pts = points[idx]
            n = len(pts)
            hv = 0.0

            for i in range(n):
                p = pts[i]
                # Width in obj0
                if i == 0:
                    w0 = p[0] - pts[0][0]  # 0, this point is the first
                    # Actually, the "slice" for point i is from pts[i][0] to the next point's obj0
                    # or to ref[0]. But in the standard algorithm, we consider the box
                    # [pts[i][0], ref[0]] and subtract overlaps.

                # Let me use the following correct method:
                # The hypervolume = sum over all maximal boxes.
                # For each point p, define its "owner box" as the set of points y
                # such that y >= p (component-wise) and y is not dominated by any other point.
                # The owner box of p is:
                # [p[0], b0(p)] x [p[1], b1(p)] x [p[2], b2(p)]
                # where b0(p) = min(ref[0], min_{q: q[0] > p[0]} q[0])
                #   b1(p) = min(ref[1], min_{q: q[1] > p[1] and q[0] in [p[0], b0(p)]} q[1])
                #   b2(p) = min(ref[2], min_{q: q[2] > p[2] and q[0] in [p[0], b0(p)] and q[1] in [p[1], b1(p)]} q[2])
                # This is still complex to compute exactly.

            # I'll use a simpler approach: since n is small (< 50), compute hypervolume
            # by summing volumes of all non-overlapping boxes formed by the arrangement.
            # For 3D, collect all unique x-coordinates, y-coordinates, z-coordinates.
            # Create a grid of boxes and check which are covered.

            # Collect unique coordinates
            x_coords = sorted(set([ref[0]] + list(pts[:, 0])))
            y_coords = sorted(set([ref[1]] + list(pts[:, 1])))
            z_coords = sorted(set([ref[2]] + list(pts[:, 2])))

            # For each box in the grid, check if it's covered by at least one point
            hv = 0.0
            for xi in range(len(x_coords) - 1):
                for yi in range(len(y_coords) - 1):
                    for zi in range(len(z_coords) - 1):
                        # Box: [x_coords[xi], x_coords[xi+1]] x [y_coords[yi], y_coords[yi+1]] x [z_coords[zi], z_coords[zi+1]]
                        # Check if the entire box is dominated by at least one point
                        # i.e., exists p such that p[0] <= x_coords[xi] and p[1] <= y_coords[yi] and p[2] <= z_coords[zi]
                        covered = False
                        for p in pts:
                            if p[0] <= x_coords[xi] and p[1] <= y_coords[yi] and p[2] <= z_coords[zi]:
                                covered = True
                                break
                        if covered:
                            vol = (x_coords[xi+1] - x_coords[xi]) * (y_coords[yi+1] - y_coords[yi]) * (z_coords[zi+1] - z_coords[zi])
                            hv += vol

            return hv

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            front = []

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    ranks[p] = 0
                    front.append(p)

            i = 0
            while len(front) > 0:
                i += 1
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            ranks[q] = i
                            next_front.append(q)
                front = next_front

            return ranks

        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)

            for rank in unique_ranks:
                members = np.where(ranks == rank)[0]
                if len(members) <= 2:
                    cd[members] = np.inf
                    continue

                for m in range(n_obj):
                    obj_vals = F[members, m]
                    sorted_idx = np.argsort(obj_vals)
                    cd[members[sorted_idx[0]]] = np.inf
                    cd[members[sorted_idx[-1]]] = np.inf
                    obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                    if obj_range > 1e-12:
                        for k in range(1, len(sorted_idx) - 1):
                            cd[members[sorted_idx[k]]] += (
                                obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]
                            ) / obj_range

            return cd

        def tournament_select(ranks, cd, pop_size, k=3):
            best_idx = np.random.randint(0, pop_size)
            for _ in range(k - 1):
                cand = np.random.randint(0, pop_size)
                if ranks[cand] < ranks[best_idx]:
                    best_idx = cand
                elif ranks[cand] == ranks[best_idx] and cd[cand] > cd[best_idx]:
                    best_idx = cand
            return best_idx

        def sbx_crossover(x1, x2, beta=15):
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                u = np.random.random()
                if u <= 0.5:
                    beta_c = (2 * u) ** (-1) if u > 0 else 1e6
                    beta_c = max(beta_c, 1e-6)
                    child1[i] = 0.5 * ((1 + beta_c) * x1[i] + (1 - beta_c) * x2[i])
                else:
                    beta_c = (2 * (1 - u)) ** (-1) if u < 1 else 1e6
                    beta_c = max(beta_c, 1e-6)
                    child1[i] = 0.5 * ((1 + beta_c) * x2[i] + (1 - beta_c) * x1[i])
                child2[i] = hi[i] + lo[i] - child1[i]
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def mutate(child, sigma_scale=0.1):
            for i in range(dim):
                if np.random.random() < 1.0 / dim:
                    sigma = sigma_scale * (hi[i] - lo[i])
                    child[i] += sigma * np.random.standard_normal()
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        def lhs_sample(n, d, lo, hi):
            samples = np.zeros((n, d))
            for dd in range(d):
                perm = np.random.permutation(n)
                samples[:, dd] = (perm + np.random.random(n)) / n
            return lo + (hi - lo) * samples

        def local_refinement(X, F, problem, budget_used, deadline, lo, hi, dim):
            """Do local search around non-dominated points using weight vectors with adaptive steps."""
            ranks = fast_non_dominated_sort(F)
            nondom_mask = (ranks == 0)
            nondom_idx = np.where(nondom_mask)[0]

            if len(nondom_idx) == 0:
                return X, F, budget_used

            # Generate weight vectors - more diverse set
            n_levels = 7
            weights = []
            for i in range(n_levels + 1):
                for j in range(n_levels + 1 - i):
                    k = n_levels - i - j
                    w = np.array([i, j, k], dtype=float)
                    if np.sum(w) > 0:
                        weights.append(w / np.sum(w))
            # Add some random weights for diversity
            for _ in range(10):
                w = np.random.random(n_obj)
                w /= w.sum()
                weights.append(w)

            for w in weights:
                if budget_used >= self.budget or time.time() > deadline:
                    break

                if len(nondom_idx) > 0:
                    scalarized = F[nondom_idx] @ w
                    best_local_idx = nondom_idx[np.argmin(scalarized)]
                else:
                    continue

                base = X[best_local_idx].copy()
                scalar_base = F[best_local_idx] @ w

                # Adaptive local search with decreasing step sizes
                for step in range(8):
                    if budget_used >= self.budget or time.time() > deadline:
                        break

                    scale = 0.10 * (0.60 ** step) * (hi - lo)
                    n_trials = 3
                    for trial in range(n_trials):
                        if budget_used >= self.budget or time.time() > deadline:
                            break

                        perturbed = base + scale * np.random.standard_normal(dim)
                        perturbed = np.clip(perturbed, lo, hi)
                        fp = evaluate(perturbed)
                        budget_used += 1

                        scalar_new = fp @ w

                        if scalar_new < scalar_base:
                            is_dom = False
                            for j in range(len(F)):
                                if np.all(F[j] <= fp) and np.any(F[j] < fp):
                                    is_dom = True
                                    break
                            if not is_dom:
                                F[best_local_idx] = fp
                                X[best_local_idx] = perturbed
                                base = perturbed.copy()
                                scalar_base = scalar_new

            return X, F, budget_used

        # SEARCH STRATEGY: Multiple diverse runs, keep best hypervolume
        # Strategy 1: LHS + NSGA-II + local refinement (like original but optimized)
        # Strategy 2: Random restarts with local search
        # Strategy 3: Weighted sum multi-start

        all_points = []  # List of (x, f) pairs

        def run_nsga2(pop_size, max_budget, deadline):
            """Run NSGA-II with given population size and budget."""
            X = lhs_sample(pop_size, dim, lo, hi)
            F = np.array([evaluate(x) for x in X])
            budget_used = pop_size

            if budget_used >= max_budget:
                return X, F, budget_used

            G = (max_budget - pop_size) // pop_size

            for gen in range(G):
                if time.time() > deadline or budget_used >= max_budget:
                    break

                ranks = fast_non_dominated_sort(F)
                cd = crowding_distance(F, ranks)

                offspring_X = []
                offspring_F = []

                for _ in range(pop_size // 2):
                    if time.time() > deadline or budget_used >= max_budget:
                        break

                    p1 = tournament_select(ranks, cd, pop_size)
                    p2 = tournament_select(ranks, cd, pop_size)

                    if np.random.random() < 0.85:
                        c1, c2 = sbx_crossover(X[p1], X[p2])
                    else:
                        c1, c2 = X[p1].copy(), X[p2].copy()

                    c1 = mutate(c1)
                    c2 = mutate(c2)

                    f1 = evaluate(c1)
                    budget_used += 1
                    f2 = evaluate(c2)
                    budget_used += 1

                    offspring_X.append(c1)
                    offspring_F.append(f1)
                    offspring_X.append(c2)
                    offspring_F.append(f2)

                    if budget_used >= max_budget:
                        break

                if budget_used >= max_budget or len(offspring_X) == 0:
                    break

                X_new = np.vstack([X, np.array(offspring_X)])
                F_new = np.vstack([F, np.array(offspring_F)])

                ranks_new = fast_non_dominated_sort(F_new)
                cd_new = crowding_distance(F_new, ranks_new)

                sort_idx = np.lexsort((-cd_new, ranks_new))
                X = X_new[sort_idx[:pop_size]]
                F = F_new[sort_idx[:pop_size]]

            return X, F, budget_used

        def run_random_restart(n_restarts, evals_per_restart, deadline):
            """Run multiple random restarts with local search."""
            best_hv = -1
            best_X = None
            best_F = None
            budget_used = 0

            for r in range(n_restarts):
                if budget_used >= self.budget or time.time() > deadline:
                    break

                # Random starting point
                x0 = lo + np.random.random(dim) * (hi - lo)
                f0 = evaluate(x0)
                budget_used += 1

                X = [x0]
                F = [f0]

                # Local search: random perturbations
                for step in range(evals_per_restart - 1):
                    if budget_used >= self.budget or time.time() > deadline:
                        break

                    # Pick a random existing point and perturb it
                    idx = np.random.randint(0, len(X))
                    scale = 0.05 * np.random.random() * (hi - lo)
                    perturbed = X[idx] + scale * np.random.standard_normal(dim)
                    perturbed = np.clip(perturbed, lo, hi)
                    fp = evaluate(perturbed)
                    budget_used += 1

                    # Add to population if not dominated
                    dominated = False
                    for j in range(len(F)):
                        if np.all(F[j] <= fp) and np.any(F[j] < fp):
                            dominated = True
                            break

                    if not dominated:
                        # Remove points dominated by fp
                        keep = []
                        for j in range(len(F)):
                            if not (np.all(fp <= F[j]) and np.any(fp < F[j])):
                                keep.append(j)
                        for j in keep:
                            X.append(X[j])
                            F.append(F[j])
                        # Actually, let me just append and filter later
                        X.append(perturbed)
                        F.append(fp)

                    # Keep population size manageable
                    if len(X) > 50:
                        ranks = fast_non_dominated_sort(np.array(F))
                        cd = crowding_distance(np.array(F), ranks)
                        sort_idx = np.lexsort((-cd, ranks))
                        X = [X[i] for i in sort_idx[:50]]
                        F = [F[i] for i in sort_idx[:50]]

                # Compute hypervolume
                if len(F) > 0:
                    F_arr = np.array(F)
                    hv = _hv3d(F_arr, ref)
                    if hv > best_hv:
                        best_hv = hv
                        best_X = np.array(X)
                        best_F = np.array(F)

            return best_X, best_F, budget_used

        # Execute strategies, tracking budget
        # Strategy 1: NSGA-II with larger population
        pop_size1 = 50
        X1, F1, budget_used = run_nsga2(pop_size1, min(self.budget, pop_size1 + 150), deadline)
        all_points.append((X1, F1))

        # Strategy 2: Local refinement on best points
        if budget_used < self.budget and time.time() < deadline:
            X1, F1, budget_used = local_refinement(X1, F1, problem, budget_used, deadline, lo, hi, dim)
            all_points[-1] = (X1, F1)

        # Strategy 3: More NSGA-II generations if budget remains
        if budget_used < self.budget and time.time() < deadline:
            remaining = self.budget - budget_used
            pop_size2 = min(30, remaining)
            if pop_size2 >= 10:
                X2, F2, budget_used2 = run_nsga2(pop_size2, remaining, deadline)
                budget_used += budget_used2
                if budget_used2 > 0:
                    all_points.append((X2, F2))

        # Strategy 4: Weighted sum scalarization multi-start with better step schedule
        if budget_used < self.budget and time.time() < deadline:
            n_starts = min(12, (self.budget - budget_used) // 12)
            for s in range(n_starts):
                if budget_used >= self.budget or time.time() > deadline:
                    break

                # Random weight vector
                w = np.random.random(n_obj)
                w /= w.sum()

                # Start from a random point
                x0 = lo + np.random.random(dim) * (hi - lo)
                f0 = evaluate(x0)
                budget_used += 1

                base = x0.copy()
                scalar_base = f0 @ w

                for step in range(11):
                    if budget_used >= self.budget or time.time() > deadline:
                        break

                    scale = 0.06 * (0.65 ** (step // 2)) * (hi - lo)
                    perturbed = base + scale * np.random.standard_normal(dim)
                    perturbed = np.clip(perturbed, lo, hi)
                    fp = evaluate(perturbed)
                    budget_used += 1

                    scalar_new = fp @ w
                    if scalar_new < scalar_base:
                        # Check if dominated
                        is_dom = False
                        for Xs, Fs in all_points:
                            if len(Fs) == 0:
                                continue
                            for j in range(len(Fs)):
                                if np.all(Fs[j] <= fp) and np.any(Fs[j] < fp):
                                    is_dom = True
                                    break
                            if is_dom:
                                break
                        if not is_dom:
                            base = perturbed.copy()
                            scalar_base = scalar_new
                            # Add to all_points[0] if it's non-dominated there
                            if len(F1) > 0:
                                dominated = False
                                for j in range(len(F1)):
                                    if np.all(F1[j] <= fp) and np.any(F1[j] < fp):
                                        dominated = True
                                        break
                                if not dominated:
                                    F1 = np.vstack([F1, fp])
                                    X1 = np.vstack([X1, perturbed])
                            else:
                                F1 = fp.reshape(1, -1)
                                X1 = perturbed.reshape(1, -1)

        # Strategy 5: Final high-precision polish on the best non-dominated points
        if budget_used < self.budget and time.time() < deadline:
            # Get current non-dominated points
            if len(F1) > 0:
                ranks_final = fast_non_dominated_sort(F1)
                nd_idx = np.where(ranks_final == 0)[0]
                
                if len(nd_idx) > 0:
                    # For each non-dominated point, do a very fine local search
                    n_polish = min(len(nd_idx), (self.budget - budget_used) // 5)
                    for idx in nd_idx[:n_polish]:
                        if budget_used >= self.budget or time.time() > deadline:
                            break
                        
                        base = X1[idx].copy()
                        f_base = F1[idx].copy()
                        
                        # Very small perturbations for high-precision refinement
                        for step in range(4):
                            if budget_used >= self.budget or time.time() > deadline:
                                break
                            
                            scale = 0.02 * (0.5 ** step) * (hi - lo)
                            for trial in range(2):
                                if budget_used >= self.budget or time.time() > deadline:
                                    break
                                
                                perturbed = base + scale * np.random.standard_normal(dim)
                                perturbed = np.clip(perturbed, lo, hi)
                                fp = evaluate(perturbed)
                                budget_used += 1
                                
                                # Check if this improves any objective without worsening others too much
                                improved = np.any(fp < f_base)
                                not_worsened = np.all(fp <= f_base + 0.01)
                                
                                if improved and not_worsened:
                                    # Check dominance against all current points
                                    is_dom = False
                                    for j in range(len(F1)):
                                        if np.all(F1[j] <= fp) and np.any(F1[j] < fp):
                                            is_dom = True
                                            break
                                    
                                    if not is_dom:
                                        F1[idx] = fp
                                        X1[idx] = perturbed
                                        base = perturbed.copy()
                                        f_base = fp.copy()

        # Collect all points and compute final hypervolume
        if len(all_points) > 0:
            # Merge all points
            all_X = []
            all_F = []
            for Xs, Fs in all_points:
                if len(Xs) > 0:
                    all_X.append(Xs)
                    all_F.append(Fs)

            if len(all_X) > 0:
                all_X = np.vstack(all_X)
                all_F = np.vstack(all_F)

                # Remove dominated points
                n = len(all_F)
                non_dom = np.ones(n, dtype=bool)
                for i in range(n):
                    if not non_dom[i]:
                        continue
                    for j in range(n):
                        if i == j or not non_dom[j]:
                            continue
                        if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                            non_dom[i] = False
                            break

                X_final = all_X[non_dom]
                F_final = all_F[non_dom]
            else:
                X_final = np.array([lo + 0.5 * (hi - lo)])
                F_final = np.array([evaluate(X_final[0])])
                budget_used += 1
        else:
            # Fallback: just evaluate some points
            X_final = lhs_sample(10, dim, lo, hi)
            F_final = np.array([evaluate(x) for x in X_final])
            budget_used = 10

        return X_final
# EVOLVE-BLOCK-END
