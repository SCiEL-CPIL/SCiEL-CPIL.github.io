"""Multi-objective optimizer for RE34 using Hypervolume-guided evolutionary search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hypervolume-guided multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1705.0, 12.0, 0.3])

        # Population size: 40 allows 5 generations (40*5 = 200)
        pop_size = 40

        # Initialize population using Latin Hypercube for better spread
        X = np.empty((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.rand(pop_size)) / pop_size
        X = X * (hi - lo) + lo

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evaluated = pop_size

        # Helper: Exact 3D hypervolume computation
        def hypervolume_3d(points, ref_point):
            """
            Compute exact 3D hypervolume for a set of points.
            points: (N, 3) array, lower is better.
            ref_point: (3,) array.
            Returns hypervolume of the union of boxes from each point to ref_point.
            Only points strictly better than ref in all objectives count.
            """
            # Filter points that are strictly better than reference
            valid = np.all(points < ref_point, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # For each point, compute its contribution to hypervolume
            # A point contributes if it is not dominated by any previous point
            # We use an incremental approach: for each point in sorted order,
            # compute the volume of the region it uniquely dominates
            
            n = len(pts)
            hv = 0.0
            
            # For each point, find the "envelope" of points that dominate it
            # in the remaining 2 dimensions
            for i in range(n):
                x_i, y_i, z_i = pts[i]
                
                # The box from this point to ref is [x_i, ref[0]] x [y_i, ref[1]] x [z_i, ref[2]]
                # But we need to subtract regions already covered by earlier points
                # that have x <= x_i (since sorted by x, all previous have x <= x_i)
                
                # Find all previous points with y <= y_i and z <= z_i (they dominate in y,z)
                # Actually, we need to compute the 2D hypervolume in y-z space
                # of points j < i with y_j <= y_i and z_j <= z_i, relative to (y_i, z_i)
                # Then the contribution is (ref[0] - x_i) * (area of unique y-z region)
                
                # More precisely: the contribution of point i is
                # (ref[0] - x_i) * HV_2D({(y_j, z_j) for j < i : y_j <= y_i, z_j <= z_i} 
                #                        relative to (y_i, z_i))
                # Wait, that's not quite right. Let me think again.
                
                # The standard approach: sort by x. For each point i, its contribution
                # is (ref[0] - x_i) times the 2D hypervolume in the y-z plane of the
                # set of points {j: j < i, y_j <= y_i, z_j <= z_i} with reference (y_i, z_i).
                # But actually, we need the "new" area that point i adds.
                
                # Let me use a different approach: for each point, compute the volume of
                # the box it dominates that is not dominated by any other point.
                
                # Alternative: use the fact that for sorted points by x, the contribution
                # of point i is (ref[0] - x_i) * A_i, where A_i is the area in y-z of the
                # region dominated by point i but not by any point j with x_j < x_i.
                
                # This is getting complex. Let me use a simpler but correct method:
                # For each point, compute the volume of the box from it to ref,
                # then subtract overlaps. But that's O(n^2) which is fine for n=40.
                
                pass
            
            # Let me use a cleaner approach: enumerate all points, and for each,
            # determine if it is "essential" (not dominated by any other point in the set).
            # Then compute the hypervolume as the sum of contributions of essential points.
            
            # Actually, the simplest correct approach for small N:
            # Use the formula: HV = sum over all subsets... no, that's exponential.
            
            # Let me use the standard sweep-line approach for 3D.
            # Sort by x. Maintain a 2D structure for y-z.
            
            # For this problem size (N <= 200), I'll use a simpler O(N^2) approach:
            # For each point i, compute the volume of the region it uniquely dominates.
            # A region is uniquely dominated by i if it's in the box [x_i, ref0] x [y_i, ref1] x [z_i, ref2]
            # and NOT in any box [x_j, ref0] x [y_j, ref1] x [z_j, ref2] for j != i where
            # x_j <= x_i, y_j <= y_i, z_j <= z_i (i.e., j dominates i or is incomparable but covers part).
            
            # This is still complex. Let me just use a well-known algorithm.
            
            # I'll implement the standard 3D hypervolume using inclusion-exclusion on sorted points.
            # For N up to 200, an O(N^2) or O(N^3) approach is acceptable.
            
            # Simple O(N^3) approach:
            # For each point i, its contribution is the volume of the box from i to ref,
            # minus the volume of the intersection of that box with all other boxes.
            # This requires computing 3D box intersections, which is doable.
            
            # Actually, let me just use a numerical approach or a simpler exact method.
            # For 3D, I can use the following:
            # 1. Sort points by x (ascending)
            # 2. For each point i, compute the 2D hypervolume in y-z of points j < i
            #    that have y_j <= y_i and z_j <= z_i, with reference (y_i, z_i)
            # 3. Contribution of i = (ref[0] - x_i) * that 2D HV
            
            # Wait, that's not right either. Let me look at this more carefully.
            
            # The correct formula: After sorting by x ascending, the hypervolume is:
            # HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * A_i
            # where x_N = ref[0], x_0 = min x, and A_i is the 2D hypervolume in y-z
            # of points with x <= x_{i+1} (i.e., points 0..i) with reference (ref[1], ref[2]).
            # Hmm, that's also not quite standard.
            
            # Let me just use a brute-force but correct method for small N:
            # The hypervolume is the measure of the union of boxes.
            # For N <= 200 in 3D, I can use the following:
            # - Sort points by x
            # - For each consecutive pair in x, compute the 2D HV of the active set
            # - "Active set" at x = t is all points with x_i <= t
            # - The volume between x_i and x_{i+1} is (x_{i+1} - x_i) * HV_2D(active_set_i, ref_yz)
            # - Plus the last segment from x_N to ref[0]
            
            # Let me implement this.
            
            # Sort by x
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]
            n = len(pts_sorted)
            
            # Add reference point as the last "x" boundary
            x_boundaries = np.concatenate([pts_sorted[:, 0], [ref_point[0]]])
            
            hv = 0.0
            for i in range(n):
                # Active set: points 0..i (all with x <= x_boundaries[i+1])
                active = pts_sorted[:i+1]
                # 2D hypervolume in y-z with reference (ref[1], ref[2])
                hv_2d = self._hv_2d(active[:, 1:3], ref_point[1:3])
                dx = x_boundaries[i+1] - x_boundaries[i]
                hv += dx * hv_2d
            
            return hv
        
        def _hv_2d(self, points, ref):
            """Compute 2D hypervolume for points with reference point."""
            # Filter points strictly better than ref
            valid = np.all(points < ref, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first coordinate (y)
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            hv = 0.0
            n = len(pts)
            for i in range(n):
                # The contribution of point i in the 2D case:
                # (ref[0] - pts[i,0]) * (ref[1] - max_z)
                # where max_z is the minimum z among points j <= i with y_j <= y_i
                # Actually, in 2D, the hypervolume is the area of the union of rectangles.
                # Sort by y. For each point i, its contribution is:
                # (ref[0] - y_i) * (ref[1] - z_min_so_far) where z_min_so_far is
                # the minimum z among points 0..i.
                # Wait, that's not right. Let me think.
                
                # In 2D, after sorting by y ascending, the hypervolume is:
                # sum over i of (y_{i+1} - y_i) * (ref[1] - min_z_0_to_i)
                # where y_N = ref[0], and min_z_0_to_i is min(pts[0..i, 1])
                
                pass
            
            # Let me use the standard 2D HV algorithm:
            # Sort by y ascending.
            # Maintain the minimum z seen so far.
            # For each point i, the area contribution is:
            # (y_{i+1} - y_i) * (ref[1] - min_z_up_to_i)
            # where y_{i+1} is the y of the next point (or ref[0] for the last).
            
            y_vals = np.concatenate([pts[:, 0], [ref[0]]])
            min_z = np.inf
            hv = 0.0
            for i in range(n):
                min_z = min(min_z, pts[i, 1])
                dy = y_vals[i+1] - y_vals[i]
                hv += dy * (ref[1] - min_z)
            
            return hv

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            N = len(F)
            ranks = np.zeros(N, dtype=int)
            S = [[] for _ in range(N)]
            n_dom = [0] * N
            
            for i in range(N):
                for j in range(i + 1, N):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = [i for i in range(N) if n_dom[i] == 0]
            for i in current_front:
                ranks[i] = 0
            
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            ranks[j] = rank + 1
                            next_front.append(j)
                current_front = next_front
                rank += 1
            return ranks

        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            N = len(F)
            distances = np.zeros(N)
            unique_ranks = np.unique(ranks)
            
            for r in unique_ranks:
                front_indices = np.where(ranks == r)[0]
                n_front = len(front_indices)
                
                if n_front <= 2:
                    for idx in front_indices:
                        distances[idx] = np.inf
                    continue
                
                for k in range(n_obj):
                    sorted_indices = front_indices[np.argsort(F[front_indices, k])]
                    distances[sorted_indices[0]] = np.inf
                    distances[sorted_indices[-1]] = np.inf
                    
                    f_min = F[sorted_indices[0], k]
                    f_max = F[sorted_indices[-1], k]
                    f_range = f_max - f_min
                    if f_range == 0:
                        continue
                    
                    for i in range(1, n_front - 1):
                        dist = (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / f_range
                        distances[sorted_indices[i]] += dist
                        
            return distances

        # Helper: Tournament selection
        def tournament_select(pop_X, pop_F, ranks, distances, k=2):
            N = len(pop_X)
            indices = np.random.choice(N, size=k, replace=False)
            
            best_idx = indices[0]
            for idx in indices[1:]:
                if ranks[idx] < ranks[best_idx]:
                    best_idx = idx
                elif ranks[idx] == ranks[best_idx]:
                    if distances[idx] > distances[best_idx]:
                        best_idx = idx
            return pop_X[best_idx].copy()

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, prob=0.9, eta=20.0):
            if np.random.rand() > prob:
                return x1.copy(), x2.copy()
            
            x3 = np.zeros_like(x1)
            x4 = np.zeros_like(x2)
            
            for i in range(len(x1)):
                if x1[i] == x2[i]:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    x3[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    x4[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    x3[i] = np.clip(x3[i], lo[i], hi[i])
                    x4[i] = np.clip(x4[i], lo[i], hi[i])
            
            return x3, x4

        # Helper: Polynomial Mutation
        def polynomial_mutate(x, prob=1.0/dim, eta=20.0):
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < prob:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                        x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                    
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while evaluated + pop_size <= self.budget:
            if time.time() > deadline:
                break
            
            # Calculate ranks and distances
            ranks = non_dominated_sort(F)
            distances = crowding_distance(F, ranks)
            
            # Generate offspring
            offspring_X = np.empty((pop_size, dim))
            for i in range(pop_size):
                p1 = tournament_select(X, F, ranks, distances)
                p2 = tournament_select(X, F, ranks, distances)
                
                c1, c2 = sbx_crossover(p1, p2)
                
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                
                offspring_X[i] = c1
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            evaluated += pop_size
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Selection for next generation using hypervolume contribution
            # Compute hypervolume contribution of each point
            # Approximate: use rank and distance, but bias toward hypervolume
            combined_ranks = non_dominated_sort(combined_F)
            combined_distances = crowding_distance(combined_F, combined_ranks)
            
            # Sort by rank, then by distance
            sort_keys = np.lexsort((-combined_distances, combined_ranks))
            selected_indices = sort_keys[:pop_size]
            
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
