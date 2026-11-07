import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 10

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n = self.n_obj
        pop_size = self.pop_size
        
        rng = np.random.default_rng(0)
        
        # Initialization
        # Use Sobol if possible, else random
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d, scramble=True)
            X = sampler.random(pop_size)
            X = lo + (hi - lo) * X
        except ImportError:
            X = lo + (hi - lo) * rng.random((pop_size, d))
            
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main Loop
        while used + pop_size <= self.budget:
            if time.time() >= deadline:
                break
            
            # Selection
            parents = self._nsga2_select_hv(X, F)
            
            # Offspring
            Q = np.zeros((pop_size, d))
            FQ = np.zeros((pop_size, n))
            
            for i in range(pop_size):
                p1 = self._tournament(X, F, rng)
                p2 = self._tournament(X, F, rng)
                
                x1 = X[p1]
                x2 = X[p2]
                
                # SBX Crossover
                child = self._sbx(x1, x2, lo, hi, rng)
                
                # Polynomial Mutation
                child = self._poly_mutate(child, lo, hi, rng)
                
                Q[i] = child
                FQ[i] = problem(child)
            
            used += pop_size
            
            # Combine and Select
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            
            X, F = self._nsga2_select_hv(R_X, R_F)
            
        return X

    def _nsga2_select_hv(self, X, F):
        # Non-dominated sort
        fronts = self._non_dominated_sort(F)
        
        selected_idx = []
        current_size = 0
        N = self.pop_size
        
        for front in fronts:
            if current_size + len(front) <= N:
                selected_idx.extend(front)
                current_size += len(front)
            else:
                # Need to pick from this front
                need = N - current_size
                # Get F values for this front
                F_front = F[front]
                # Select best 'need' based on HV
                best_local = self._select_by_hv(F_front, need)
                # Map back to global indices
                for i in best_local:
                    selected_idx.append(front[i])
                current_size += need
                break
                
        return X[selected_idx], F[selected_idx]

    def _non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        remaining = set(range(n))
        
        while remaining:
            front = []
            for i in list(remaining):
                dominated = False
                for j in remaining:
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
                    remaining.remove(i)
            if not front: break
            for i in front:
                ranks[i] = len(fronts)
            fronts.append(front)
            
        return fronts

    def _select_by_hv(self, F_front, k):
        # Greedy selection by marginal HV
        # F_front is array of shape (M, 3)
        # ref is [1.1, 1.3, 1.1]
        ref = np.array([1.1, 1.3, 1.1])
        
        # Filter points strictly better than ref
        valid_mask = np.all(F_front < ref, axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) <= k:
            # If not enough valid points, we need to handle this.
            # But usually front points are good.
            # If we don't have enough, we pick all valid and maybe some invalid?
            # Invalid points contribute 0 HV.
            # For selection, we just want the best k.
            # If we have fewer than k valid, we pick all valid and fill rest with arbitrary?
            # No, we must pick k points.
            # If a point is worse than ref, its contribution is 0.
            # So we pick the k points with highest marginal contribution.
            
            selected = []
            remaining_indices = list(range(len(F_front)))
            
            for _ in range(k):
                best_idx = -1
                best_hv = -1
                current_set = np.array([F_front[i] for i in selected])
                
                for idx in remaining_indices:
                    # Calculate HV of selected + F_front[idx]
                    if len(selected) == 0:
                        hv = self._hv_3d(np.array([F_front[idx]]), ref)
                    else:
                        hv = self._hv_3d(np.vstack([current_set, F_front[idx]]), ref)
                    
                    if hv > best_hv:
                        best_hv = hv
                        best_idx = idx
                
                if best_idx != -1:
                    selected.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    # Should not happen
                    break
            
            return selected

        # If we have enough valid points, do greedy on all
        selected = []
        remaining_indices = list(range(len(F_front)))
        
        for _ in range(k):
            best_idx = -1
            best_hv = -1
            current_set = np.array([F_front[i] for i in selected])
            
            for idx in remaining_indices:
                if len(selected) == 0:
                    hv = self._hv_3d(np.array([F_front[idx]]), ref)
                else:
                    hv = self._hv_3d(np.vstack([current_set, F_front[idx]]), ref)
                
                if hv > best_hv:
                    best_hv = hv
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        return selected

    def _hv_3d(self, points, ref):
        # points: (K, 3)
        # ref: (3,)
        # Returns hypervolume
        if len(points) == 0:
            return 0.0
        
        # Filter points dominated by ref (must be < ref in all dims)
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        pts = pts[np.argsort(pts[:, 0])]
        
        # 3D HV calculation
        # We can reduce to 2D by sweeping
        # Volume = sum over i of (x_i - x_{i-1}) * HV_2d(subset)
        
        # Simple O(N^3) or O(N^2) algorithm for small N
        # Since N is small (<=10), we can use a simple union of boxes or inclusion-exclusion?
        # Inclusion-exclusion is hard.
        # Let's use the standard sweep line for 3D.
        
        # Actually, for N<=10, we can just use a grid or a simple recursive calculation.
        # Or use the fact that HV is the volume of the union of hyper-rectangles [p_i, ref].
        
        # Let's implement a simple O(N^3) check or use a library? No library.
        # Let's use the "divide and conquer" or simple sorting.
        
        # Sort by x
        # pts is already sorted by x.
        
        # We can compute HV by iterating through x-slices.
        # But for small N, a simple method is to find the non-dominated set (which is pts)
        # and calculate the volume of the region covered.
        
        # Algorithm:
        # 1. Sort by x.
        # 2. For each point i, find the points that dominate it in y,z but are to the right in x?
        # No, standard 3D HV algorithm:
        
        # Let's use a simple recursive reduction or just brute force if N is very small?
        # N is max 10 in the front, but in _select_by_hv, we check subsets of size up to 10.
        # So max points in _hv_3d is 10.
        
        # Brute force union of boxes is not trivial.
        # Let's use the standard O(N^2 log N) or O(N^3) algorithm.
        
        # Reference: "Hypervolume" calculation for 3D.
        # We can project to 2D for each x-interval.
        
        # Sort by x.
        # x_coords = pts[:, 0]
        # For i in range(N):
        #    x_min = x_coords[i]
        #    x_max = x_coords[i+1] if i < N-1 else ref[0]
        #    dx = x_max - x_min
        #    Consider all points j such that x_coords[j] <= x_max (which is all points from 0 to i? No, points with x < x_max)
        #    Actually, the active set for the slice (x_i, x_{i+1}) is all points with x <= x_{i+1}.
        #    Wait, the region is [p, ref].
        #    The volume is integral over x of Area(y,z) dx.
        #    Area(y,z) is the area of the union of rectangles [y_j, z_j, ref_y, ref_z] for all j with x_j <= x.
        
        # Since N is small, we can do this:
        # 1. Sort by x.
        # 2. For each interval between consecutive x values (and ref[0]), calculate the 2D HV of the active set.
        
        x_vals = pts[:, 0]
        total_hv = 0.0
        
        # Add ref[0] as the upper bound
        x_bounds = np.append(x_vals, ref[0])
        
        # Iterate through intervals
        for i in range(len(x_bounds) - 1):
            x_left = x_bounds[i]
            x_right = x_bounds[i+1]
            
            if x_right <= x_left:
                continue
                
            # Active points: all points with x <= x_right
            # Since sorted, this is points[0 : i+1]
            # Note: pts is sorted by x.
            # The points included in the slice (x_left, x_right) are those with x <= x_right.
            # Which corresponds to indices 0 to i (since x_vals[i] is the left bound of the next interval? No.)
            # x_vals[0] < x_vals[1] ...
            # Interval 0: [x_vals[0], x_vals[1]). Active points: pts[0].
            # Interval 1: [x_vals[1], x_vals[2]). Active points: pts[0], pts[1].
            # ...
            # Interval i: [x_vals[i], x_vals[i+1]). Active points: pts[0...i].
            # Last Interval: [x_vals[-1], ref[0]). Active points: all pts.
            
            # So for interval i, active set is pts[0 : i+1]
            active_pts = pts[0 : i+1]
            
            # Calculate 2D HV of active_pts with ref_y, ref_z
            hv_2d = self._hv_2d(active_pts[:, 1:], np.array([ref[1], ref[2]]))
            
            total_hv += (x_right - x_left) * hv_2d
            
        return total_hv

    def _hv_2d(self, points, ref):
        # points: (K, 2)
        # ref: (2,)
        if len(points) == 0:
            return 0.0
        
        # Sort by y
        points = points[np.argsort(points[:, 0])]
        
        y_vals = points[:, 0]
        z_vals = points[:, 1]
        
        total_hv = 0.0
        y_bounds = np.append(y_vals, ref[0])
        
        for i in range(len(y_bounds) - 1):
            y_left = y_bounds[i]
            y_right = y_bounds[i+1]
            
            if y_right <= y_left:
                continue
            
            # Active points: points with y <= y_right
            # Indices 0 to i
            active_pts = points[0 : i+1]
            
            # 1D HV is max z distance
            # z_min = min(z of active points)
            # z_max = ref[1]
            # Width = z_max - z_min
            z_min = np.min(active_pts[:, 1])
            z_width = ref[1] - z_min
            
            if z_width > 0:
                total_hv += (y_right - y_left) * z_width
                
        return total_hv

    def _tournament(self, X, F, rng):
        # Standard tournament on ranks and crowding distance
        # We need ranks and cd.
        # To save time, we can just pick 2 random and compare their HV contribution?
        # Or just random?
        # Let's use a simple tournament based on non-domination rank.
        # We need to calculate ranks.
        # This is expensive if done every time.
        # In _nsga2_select_hv we calculate fronts.
        # We can store ranks and cd from the last selection?
        # Or just use random selection?
        # For small pop, random selection + strong local search (HV selection) might work.
        # But let's do a proper tournament.
        
        # We'll compute ranks on the fly for the current population X, F.
        # Since N=10, it's fast.
        
        n = len(X)
        ranks = np.zeros(n, dtype=int)
        # Simple rank calc
        for i in range(n):
            rank = 0
            for j in range(n):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    rank += 1
            ranks[i] = rank
            
        # Crowding distance
        cd = np.zeros(n)
        # ... (implement cd or just use rank)
        # Let's just use rank for simplicity.
        
        i1 = rng.integers(n)
        i2 = rng.integers(n)
        
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            return i1 if rng.random() < 0.5 else i2

    def _sbx(self, x1, x2, lo, hi, rng):
        eta = 15.0
        d = len(x1)
        child = np.zeros(d)
        for i in range(d):
            if rng.random() < 0.5:
                # Swap x1, x2 if needed? No, just use x1, x2
                if abs(x1[i] - x2[i]) < 1e-14:
                    child[i] = x1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    # We only need one child
                    child[i] = c1 if rng.random() < 0.5 else c2
            else:
                child[i] = x1[i] if rng.random() < 0.5 else x2[i]
        
        return np.clip(child, lo, hi)

    def _poly_mutate(self, x, lo, hi, rng):
        eta = 20.0
        d = len(x)
        for i in range(d):
            if rng.random() < 1.0 / d:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                x[i] += delta * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
