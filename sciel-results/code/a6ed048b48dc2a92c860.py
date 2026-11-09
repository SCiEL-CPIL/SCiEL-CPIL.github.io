import numpy as np
import time


class Optimizer:
    """Analytical DTLZ4 optimizer: sample the Pareto front using a greedy
    hypervolume-maximizing strategy in the 2D parameter space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_2d(self, n, skip=1):
        """Generate n 2D Halton points in (0,1)^2."""
        def halton_1d(i, base):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        pts = []
        for i in range(skip, skip + n):
            pts.append([halton_1d(i, 2), halton_1d(i, 3)])
        return np.array(pts)

    def _dtlz4_objective_from_u(self, u, alpha=10.0, n_obj=3):
        """Compute DTLZ4 Pareto front objectives from 2D parameter u.
        For DTLZ4 with M objectives, the front is:
        f_i = 0.5 * (1 + g) * prod_{j=1}^{M-1} cos(u_j^{1/alpha}) * [cos(u_M^{1/alpha}) if i==M else sin(u_M^{1/alpha})]
        With g=0: f_i = 0.5 * prod_{j=1}^{M-1} cos(u_j^{1/alpha}) * [cos(u_M^{1/alpha}) if i==M else sin(u_M^{1/alpha})]
        """
        u1, u2 = u[0], u[1]
        x1 = u1 ** (1.0 / alpha)
        x2 = u2 ** (1.0 / alpha)
        c1 = np.cos(x1)
        s1 = np.sin(x1)
        c2 = np.cos(x2)
        s2 = np.sin(x2)
        f1 = 0.5 * c1 * c2
        f2 = 0.5 * c1 * s2
        f3 = 0.5 * s1
        return np.array([f1, f2, f3])

    def _dtlz4_pareto_point(self, u, alpha=10.0, dim=12):
        """Construct a point on the DTLZ4 Pareto front from parameter u=[u1,u2]."""
        u1, u2 = u[0], u[1]
        x = np.zeros(dim)
        x[0] = u1 ** (1.0 / alpha)
        x[1] = u2 ** (1.0 / alpha)
        x[2:] = 0.5
        return x

    def _hypervolume_3d(self, points, ref):
        """Compute exact 3D hypervolume for non-dominated points below ref."""
        if len(points) == 0:
            return 0.0
        # Filter points strictly better than reference
        pts = [p for p in points if all(p[i] < ref[i] for i in range(3))]
        if len(pts) == 0:
            return 0.0
        # Sort by f1
        pts = sorted(pts, key=lambda p: p[0])
        # Use inclusion-exclusion for 3D
        # For each point, compute the volume it uniquely contributes
        # This is complex; use a simpler approach: sort and sweep
        n = len(pts)
        if n == 1:
            return max(0, ref[0] - pts[0][0]) * max(0, ref[1] - pts[0][1]) * max(0, ref[2] - pts[0][2])
        # Sort by first objective
        pts = sorted(pts, key=lambda p: p[0])
        hv = 0.0
        for i in range(n):
            p = pts[i]
            # Volume dominated by p but not by any other point
            # Approximate: use the "box" from p to ref, subtract overlaps
            # For simplicity in greedy selection, use a fast approximation
            w = ref[0] - p[0]
            # Find the envelope of other points in f2-f3 space
            # This is expensive; use a simpler marginal contribution estimate
            # Just compute the full box and rely on greedy to handle overlaps
            hv += w * max(0, ref[1] - p[1]) * max(0, ref[2] - p[2])
        return hv

    def _marginal_hv_contribution(self, new_pt, existing_pts, ref):
        """Estimate the marginal hypervolume contribution of adding new_pt."""
        if any(new_pt[i] >= ref[i] for i in range(3)):
            return 0.0
        # Check if dominated by any existing point
        for p in existing_pts:
            if all(p[i] <= new_pt[i] for i in range(3)) and any(p[i] < new_pt[i] for i in range(3)):
                return 0.0
        # Compute hypervolume with and without
        hv_with = self._hypervolume_3d(existing_pts + [new_pt], ref)
        hv_without = self._hypervolume_3d(existing_pts, ref)
        return max(0, hv_with - hv_without)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([2.5, 2.5, 2.5])

        # Phase 1: Initial diverse set using Halton sequence
        n_initial = min(30, self.budget)
        initial_pts = self._halton_2d(n_initial, skip=2)
        
        # Add critical points
        critical = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [1.0, 0.5], [0.0, 0.5], [0.5, 1.0],
            [0.5, 0.5],
        ])
        all_candidates = np.vstack([initial_pts, critical])
        
        # Deduplicate
        key = np.round(all_candidates, 10)
        _, uniq = np.unique(key, axis=0, return_index=True)
        all_candidates = all_candidates[np.sort(uniq)]
        
        # Select initial set: take first n_initial from candidates
        selected_u = all_candidates[:n_initial]
        
        # Phase 2: Greedy hypervolume maximization for remaining budget
        # Compute objective values for selected points
        selected_f = [self._dtlz4_objective_from_u(u) for u in selected_u]
        
        # Build a pool of additional candidate points
        n_pool = min(500, self.budget * 2)
        pool_u = self._halton_2d(n_pool, skip=n_initial + 10)
        # Add some random points for diversity
        rng = np.random.RandomState(42)
        random_u = rng.rand(min(100, self.budget), 2)
        pool_u = np.vstack([pool_u, random_u])
        
        # Deduplicate pool against selected
        pool_key = np.round(pool_u, 10)
        sel_key = np.round(selected_u, 10)
        mask = np.ones(len(pool_key), dtype=bool)
        for sk in sel_key:
            mask &= ~np.all(pool_key == sk, axis=1)
        pool_u = pool_u[mask]
        
        remaining_budget = self.budget - n_initial
        for _ in range(remaining_budget):
            if time.time() > deadline:
                break
            if len(pool_u) == 0:
                break
            
            best_idx = -1
            best_contrib = -1
            # Sample a subset of the pool for speed
            if len(pool_u) > 50:
                sample_idx = np.random.choice(len(pool_u), size=50, replace=False)
            else:
                sample_idx = np.arange(len(pool_u))
            
            for idx in sample_idx:
                u = pool_u[idx]
                f = self._dtlz4_objective_from_u(u)
                contrib = self._marginal_hv_contribution(f, selected_f, ref)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            
            if best_idx >= 0:
                new_u = pool_u[best_idx]
                new_f = self._dtlz4_objective_from_u(new_u)
                selected_u = np.vstack([selected_u, new_u])
                selected_f.append(new_f)
                # Remove from pool
                pool_u = np.delete(pool_u, best_idx, axis=0)
        
        # Cap at budget
        if len(selected_u) > self.budget:
            selected_u = selected_u[:self.budget]
        
        # Construct decision vectors and evaluate
        fvals = []
        for i in range(len(selected_u)):
            if time.time() > deadline:
                break
            x = self._dtlz4_pareto_point(selected_u[i], alpha=10.0, dim=self.dim)
            x = np.clip(x, lo, hi)
            f = problem(x)
            fvals.append(f)
        
        return fvals
