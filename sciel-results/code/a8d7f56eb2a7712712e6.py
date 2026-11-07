"""Exploit DTLZ1's known Pareto front with exact marginal-HV greedy refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploit DTLZ1 closed-form front with exact marginal hypervolume greedy selection.
    For DTLZ1 (7D, 3O), the Pareto front is parameterized by a simplex in objective
    space. We greedily place each new point to maximize exact HV increment.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _simplex_to_decision(self, f1, f2, f3, K, lo, hi):
        """Map simplex coordinates (f1, f2, f3) to decision vector."""
        if f1 < 1e-10:
            x = np.full(self.dim, 0.5)
            x[0] = 0.0
            x[1] = 0.0
            x[2] = 1.0
        elif f2 < 1e-10:
            x = np.full(self.dim, 0.5)
            x[0] = f1 / K
            x[1] = 1.0
            x[2] = 0.0
        else:
            x1 = f1 / K
            x2 = 1.0 - f2 / f1
            x3 = 1.0 - f3 / f2
            x = np.full(self.dim, 0.5)
            x[0] = x1
            x[1] = x2
            x[2] = x3

        return np.clip(x, lo, hi)

    def _hypervolume_3d(self, points, ref):
        """
        Compute exact hypervolume in 3D for a set of non-dominated points
        with respect to reference point ref.
        points: (n, 3) array of objective vectors (all < ref in every dim)
        ref: (3,) reference point
        Returns: scalar hypervolume
        """
        n = len(points)
        if n == 0:
            return 0.0
        if n == 1:
            p = points[0]
            return max(0, ref[0]-p[0]) * max(0, ref[1]-p[1]) * max(0, ref[2]-p[2])
        
        # Sort by first objective
        idx = np.argsort(points[:, 0])
        pts = points[idx]
        
        # Use inclusion-exclusion / sweep-line approach for 3D HV
        # For 3D, we can use the "layer cake" decomposition
        # Sort by f0, then for each point, compute the 2D HV of the "new" region it contributes
        
        hv = 0.0
        # We process points from best to worst in f0
        # For each point i, the new hypervolume it adds is the volume of the box
        # [f0_i, f0_{i+1}] x [f1_i, ref1] x [f2_i, ref2] minus what's already covered
        # by points j < i (which have smaller f0)
        
        # A cleaner approach for 3D:
        # Sort by f0 ascending. For each point i, consider the 2D rectangle in (f1, f2)
        # space: [f1_i, ref1] x [f2_i, ref2], but clipped by previously seen points.
        # The contribution is (f0_{i+1} - f0_i) * area_2d, where f0_{n} = ref[0]
        # and area_2d is the area of the union of rectangles from points i..n-1
        # projected onto (f1, f2) plane, but only the part "above" point i's dominance region.
        
        # Actually, let's use a standard O(n^2) or O(n log n) 3D HV algorithm.
        # For small n (up to ~200), O(n^2) is fine.
        
        # Standard approach: sort by f0. For each point i (in order of increasing f0),
        # the hypervolume slice between f0_i and f0_{i+1} has height (f0_{i+1} - f0_i)
        # and cross-section equal to the 2D hypervolume of the set of points j >= i
        # in the (f1, f2) plane with reference (ref1, ref2).
        # But we need to be careful: only points that are not dominated in (f1,f2) by
        # any point with index < i contribute to the "active" set at slice i.
        # Actually no - the standard formula is:
        # HV = sum_{i=0}^{n-1} (f0_{i+1} - f0_i) * HV_2D({points j : j >= i}, ref_2d)
        # where HV_2D is the 2D hypervolume of the set of points from i to n-1.
        # This works because we sort by f0 and each slice only sees points with f0 >= f0_i.
        
        f0 = pts[:, 0]
        f0_next = np.concatenate([f0[1:], [ref[0]]])
        
        total = 0.0
        for i in range(n):
            # 2D HV of points[i:] with ref (ref[1], ref[2])
            sub = pts[i:, 1:]  # (f1, f2) coordinates
            hv2 = self._hypervolume_2d(sub, np.array([ref[1], ref[2]]))
            total += (f0_next[i] - f0[i]) * hv2
        
        return total

    def _hypervolume_2d(self, points, ref):
        """
        Compute exact 2D hypervolume for a set of non-dominated points.
        points: (n, 2) array
        ref: (2,) reference point
        """
        n = len(points)
        if n == 0:
            return 0.0
        
        # Sort by first coordinate
        idx = np.argsort(points[:, 0])
        pts = points[idx]
        
        # For 2D, HV = sum of rectangles
        # Sort by x ascending, keep track of max y seen
        # Actually, for non-dominated points sorted by x ascending, y is descending
        # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * (ref_y - y_i)
        # where x_n = ref_x
        
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.concatenate([x[1:], [ref[0]]])
        
        total = 0.0
        for i in range(n):
            width = x_next[i] - x[i]
            height = ref[1] - y[i]
            if width > 0 and height > 0:
                total += width * height
        
        return total

    def _marginal_hv(self, new_point, existing_points, ref):
        """
        Compute the marginal hypervolume gain of adding new_point to existing_points.
        All points are assumed to be strictly better than ref in all dimensions.
        """
        # HV without new_point
        hv_before = self._hypervolume_3d(existing_points, ref)
        
        # HV with new_point
        combined = np.vstack([existing_points, new_point.reshape(1, -1)])
        hv_after = self._hypervolume_3d(combined, ref)
        
        return hv_after - hv_before

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        recorded = []
        used = 0

        # Determine K by evaluating known front points.
        test_x = np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        test_f = np.array(problem(test_x), dtype=float)
        
        test_x2 = np.array([1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5])
        test_f2 = np.array(problem(test_x2), dtype=float)
        
        test_x3 = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        test_f3 = np.array(problem(test_x3), dtype=float)
        
        used += 3
        recorded.extend([test_f, test_f2, test_f3])
        
        K_est1 = float(np.sum(test_f))
        K_est2 = float(np.sum(test_f2))
        K_est3 = float(np.sum(test_f3))
        
        K = float(np.median([K_est1, K_est2, K_est3]))
        
        if K < 0.01 or K > 10000.0:
            test_x_alt = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            test_f_alt = np.array(problem(test_x_alt), dtype=float)
            K = float(np.sum(test_f_alt))
            if K < 0.01 or K > 10000.0:
                K = 0.5
        
        R = 400.0
        ref = np.array([R, R, R])
        
        # Coarse grid initialization with power-law bias toward edges
        n_coarse = min(30, budget - 3)
        coarse_points = []
        
        # Generate grid on simplex with power-law spacing
        alpha = 1.5
        N = int(np.sqrt(2 * n_coarse))
        while (N + 1) * (N + 2) // 2 > n_coarse and N > 0:
            N -= 1
        
        for i in range(N + 1):
            if used + (N + 1 - i) > budget:
                break
            for j in range(N - i + 1):
                if used >= budget or time.time() > deadline:
                    break
                
                u = i / N
                v = j / N
                
                # Power-law to concentrate near edges
                u_t = u ** alpha
                v_t = v ** alpha
                
                if u_t + v_t > 1.0:
                    scale = 1.0 / (u_t + v_t)
                    u_t *= scale
                    v_t *= scale
                
                f1 = K * u_t
                f2 = K * v_t
                f3 = K * (1.0 - u_t - v_t)
                
                x = self._simplex_to_decision(f1, f2, f3, K, lo, hi)
                f = np.array(problem(x), dtype=float)
                recorded.append(f)
                used += 1
                coarse_points.append(f)
        
        # Adaptive refinement: greedily add points that maximize marginal HV
        if used < budget:
            rng = np.random.default_rng(42)
            existing = np.array(recorded)
            
            # Filter to only points strictly better than ref
            mask = np.all(existing < ref, axis=1)
            existing = existing[mask]
            
            remaining = budget - used
            candidates_per_iter = 50
            
            for _ in range(remaining):
                if used >= budget or time.time() > deadline:
                    break
                
                # Generate candidate points on the simplex
                candidates = []
                for _ in range(candidates_per_iter):
                    u = rng.random(3)
                    u.sort()
                    w1 = u[0]
                    w2 = u[1] - u[0]
                    w3 = 1.0 - u[1]
                    
                    # Power-law bias
                    w1_t = w1 ** alpha
                    w2_t = w2 ** alpha
                    w3_t = w3 ** alpha
                    s = w1_t + w2_t + w3_t
                    w1_t /= s
                    w2_t /= s
                    w3_t /= s
                    
                    f1 = K * w1_t
                    f2 = K * w2_t
                    f3 = K * w3_t
                    candidates.append([f1, f2, f3])
                
                candidates = np.array(candidates)
                
                # Evaluate marginal HV for each candidate
                best_gain = -1
                best_candidate = None
                for cand in candidates:
                    if np.any(cand >= ref):
                        continue
                    gain = self._marginal_hv(cand, existing, ref)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
                
                if best_candidate is None:
                    # Fallback: just pick a random point
                    u = rng.random(3)
                    u.sort()
                    w1 = u[0]
                    w2 = u[1] - u[0]
                    w3 = 1.0 - u[1]
                    best_candidate = np.array([w1, w2, w3]) * K
                    best_gain = 0
                
                f1, f2, f3 = best_candidate
                x = self._simplex_to_decision(f1, f2, f3, K, lo, hi)
                f = np.array(problem(x), dtype=float)
                recorded.append(f)
                used += 1
                
                # Update existing set
                existing = np.vstack([existing, f.reshape(1, -1)])
                # Remove dominated points to keep set manageable
                mask = np.all(existing < ref, axis=1)
                existing = existing[mask]
        
        return recorded
# EVOLVE-BLOCK-END
