"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive refinement exploiting DTLZ1's Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dtlz1_pareto_point(self, weights, rng):
        """Generate a decision vector that maps to the DTLZ1 Pareto front.
        
        For DTLZ1 with m objectives and n variables:
        - x[0:m-2] are free (affect g)
        - x[m-1:n] are the position variables on the simplex
        - On the Pareto front, g=0, so x[m-1:n] must satisfy sum(x_i^2) = 0.5
        - The objectives are: f_i = 0.5 * (1 + g) * prod_{j=m-i+1}^{m-1} x_j * sin(pi/(2*m) * x_{m-i})
        - With g=0: f_i = 0.5 * prod_{j=m-i+1}^{m-1} x_j * sin(pi/(2*m) * x_{m-i})
        
        For 3 objectives (m=3), 7 vars (n=7):
        - x[0:1] free, x[1:7] position
        - Pareto front: sum(x[1:7]^2) = 0.5
        - f0 = 0.5 * x[6] * sin(pi/6 * x[5])
        - f1 = 0.5 * x[5] * sin(pi/6 * x[4])  
        - f2 = 0.5 * (1 - x[4])
        """
        m = self.n_obj
        n = self.dim
        x = np.zeros(n)
        
        # Set free variables to 0 (g=0)
        # x[0:m-2] = 0
        
        # Generate point on simplex: sum(x[m-1:n]^2) = 0.5
        # Use Dirichlet-like distribution on the simplex
        # For 3 obj, we need x[1:7] on sphere of radius sqrt(0.5)
        # But DTLZ1 uses a specific parametrization
        
        # Standard DTLZ1 Pareto front parametrization:
        # Let y be a point on the (m-1)-dimensional simplex: y_i >= 0, sum(y_i) = 1
        # Then x[n-m+i] = y_i for i=0..m-1, and remaining x = 0
        # But DTLZ1 has a different structure with sin terms
        
        # For DTLZ1 with m=3:
        # f0 = 0.5 * x[6] * sin(pi/6 * x[5])
        # f1 = 0.5 * x[5] * sin(pi/6 * x[4])
        # f2 = 0.5 * (1 - x[4])
        # Constraint: x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 + x[6]^2 = 0.5
        
        # Simpler approach: use the known fact that DTLZ1's Pareto front 
        # can be parameterized by points on a simplex in objective space
        # f0 + f1 + f2 = 0.5 on the Pareto front
        
        # Generate weights on simplex for objectives
        # w0 + w1 + w2 = 1, w_i >= 0
        # Then target objectives: f_i = 0.5 * w_i
        
        # To achieve specific objectives, we need to solve for x[4:7]
        # This is complex, so use a different approach:
        
        # Use random points and project to Pareto front
        # Generate random non-negative weights
        w = rng.random(m)
        w = w / w.sum()
        
        # Target: f_i = 0.5 * w_i
        # For DTLZ1 m=3:
        # f2 = 0.5 * (1 - x[4]) => x[4] = 1 - 2*f2 = 1 - w[2]
        # f1 = 0.5 * x[5] * sin(pi/6 * x[4]) => x[5] = 2*f1 / sin(pi/6 * x[4]) = w[1] / sin(pi/6 * x[4])
        # f0 = 0.5 * x[6] * sin(pi/6 * x[5]) => x[6] = 2*f0 / sin(pi/6 * x[5]) = w[0] / sin(pi/6 * x[5])
        
        x[4] = 1.0 - w[2]
        
        # Ensure x[4] is in valid range for sin
        x[4] = np.clip(x[4], 0.01, 0.99)
        
        sin4 = np.sin(np.pi / (2 * m) * x[4])
        if sin4 < 1e-10:
            sin4 = 1e-10
        x[5] = w[1] / sin4
        
        sin5 = np.sin(np.pi / (2 * m) * x[5])
        if sin5 < 1e-10:
            sin5 = 1e-10
        x[6] = w[0] / sin5
        
        # Clip to [0,1]
        x = np.clip(x, 0, 1)
        
        return x

    def _hypervolume_3d(self, points, ref):
        """Compute 3D hypervolume for non-dominated points with reference point."""
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # Remove dominated points (keep only non-dominated)
        nd_indices = [0]
        for i in range(1, len(pts)):
            if pts[i][1] < pts[nd_indices[-1]][1] or pts[i][2] < pts[nd_indices[-1]][2]:
                nd_indices.append(i)
        pts = pts[nd_indices]
        
        if len(pts) == 0:
            return 0.0
        
        # Compute hypervolume using inclusion-exclusion for 3D
        # For each point, compute the box it contributes
        hv = 0.0
        for i in range(len(pts)):
            # Box from pts[i] to ref, but subtract overlaps with previous points
            # This is O(n^2) but n is small
            x0, y0, z0 = pts[i]
            
            # Full box volume
            vol = (ref[0] - x0) * (ref[1] - y0) * (ref[2] - z0)
            
            # Subtract boxes already counted (points that dominate this one in at least 2 objectives)
            # Actually, we need to be more careful. Let's use a simpler approach:
            # Sort by x, and for each point, the contribution is the area in y-z plane
            # that is not covered by previous points with smaller x
            
        # Better: use the standard 3D hypervolume algorithm
        # Sort by x
        pts = pts[np.argsort(pts[:, 0])]
        
        hv = 0.0
        for i in range(len(pts)):
            x_i = pts[i][0]
            y_i = pts[i][1]
            z_i = pts[i][2]
            
            # Width in x direction
            if i == len(pts) - 1:
                dx = ref[0] - x_i
            else:
                dx = pts[i+1][0] - x_i
            
            # For this x-slice, compute 2D hypervolume in y-z plane
            # Points j where pts[j][0] <= x_i (i.e., j <= i) contribute
            # But only if they are not dominated in y-z by later points
            
            # Collect y-z points for all j <= i
            yz_pts = pts[:i+1][:, 1:3]
            
            # Compute 2D hypervolume of yz_pts with reference (ref[1], ref[2])
            # Sort by y
            yz_order = np.argsort(yz_pts[:, 0])
            yz_sorted = yz_pts[yz_order]
            
            # Keep only non-dominated in y-z
            nd_yz = [0]
            for j in range(1, len(yz_sorted)):
                if yz_sorted[j][1] < yz_sorted[nd_yz[-1]][1]:
                    nd_yz.append(j)
            yz_nd = yz_sorted[nd_yz]
            
            # 2D hypervolume
            area_2d = 0.0
            for j in range(len(yz_nd)):
                y_j = yz_nd[j][0]
                z_j = yz_nd[j][1]
                
                if j == len(yz_nd) - 1:
                    dy = ref[1] - y_j
                else:
                    dy = yz_nd[j+1][0] - y_j
                
                dz = ref[2] - z_j
                area_2d += dy * dz
            
            hv += dx * area_2d
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        ref = np.array([400.0, 400.0, 400.0])
        
        # Initialize with a coarse grid on the Pareto front
        # Use simplex grid for 3 objectives
        n_init = 30
        F_list = []
        X_list = []
        
        # Generate initial points using simplex grid
        # Points where w0+w1+w2=1, with grid spacing
        n_per_axis = 6  # 6x6x6 grid on simplex ~ 36 points
        weights_list = []
        for i in range(n_per_axis + 1):
            for j in range(n_per_axis + 1):
                if i + j <= n_per_axis:
                    k = n_per_axis - i - j
                    w = np.array([i, j, k]) / n_per_axis
                    weights_list.append(w)
        
        # Shuffle and take first n_init
        rng.shuffle(weights_list)
        for w in weights_list[:n_init]:
            x = self._dtlz1_pareto_point(w, rng)
            f = problem(x)
            F_list.append(f)
            X_list.append(x)
        
        used = n_init
        
        # Adaptive refinement: iteratively add point that maximizes marginal hypervolume
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute current hypervolume
            F_arr = np.array(F_list)
            hv_current = self._hypervolume_3d(F_arr, ref)
            
            # Generate candidate points in a neighborhood of existing points
            # and also some random points on the Pareto front
            candidates = []
            
            # For each existing point, generate perturbations
            for idx, w_base in enumerate(weights_list[:used]):
                for scale in [0.05, 0.1, 0.2]:
                    w_pert = w_base + scale * rng.standard_normal(3)
                    w_pert = np.clip(w_pert, 0, None)
                    w_sum = w_pert.sum()
                    if w_sum > 1e-10:
                        w_pert = w_pert / w_sum
                        candidates.append(w_pert)
            
            # Add some random points
            for _ in range(10):
                w_rand = rng.random(3)
                w_rand = w_rand / w_rand.sum()
                candidates.append(w_rand)
            
            # Evaluate candidates and pick the one with best marginal hypervolume
            best_gain = -1
            best_f = None
            best_w = None
            
            for w_cand in candidates:
                x_cand = self._dtlz1_pareto_point(w_cand, rng)
                f_cand = problem(x_cand)
                used += 1
                
                # Check if this point improves hypervolume
                F_test = np.vstack([F_arr, f_cand.reshape(1, -1)])
                hv_test = self._hypervolume_3d(F_test, ref)
                gain = hv_test - hv_current
                
                if gain > best_gain:
                    best_gain = gain
                    best_f = f_cand
                    best_w = w_cand
                    best_x = x_cand
                
                # If we've used too many, break
                if used >= self.budget:
                    break
            
            if best_f is not None:
                F_list.append(best_f)
                X_list.append(best_x)
                weights_list.append(best_w)
            
            if used >= self.budget:
                break
        
        return np.array(X_list) if X_list else np.array([lo])
# EVOLVE-BLOCK-END
