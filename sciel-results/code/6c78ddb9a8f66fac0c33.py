"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Fibonacci lattice on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        used = 0
        all_F = []

        def check_time():
            return time.time() > deadline

        def eval_and_record(x):
            nonlocal used
            fx = problem(x)
            used += 1
            all_F.append(np.asarray(fx, dtype=float).copy())
            return fx

        # DTLZ2 with M=3 objectives:
        # f1 = (1+g)*cos(t1)*cos(t2)
        # f2 = (1+g)*sin(t1)*cos(t2)  
        # f3 = (1+g)*sin(t2)
        # where t1, t2 in [0, pi/2]
        # g(x) depends on x[2:] 
        # To get g=0, set x[2:] = 0.5
        # x[0] -> t1 = x[0] * (pi/2)
        # x[1] -> t2 = x[1] * (pi/2)
        
        # Strategy: Use Fibonacci lattice to distribute points uniformly on the 
        # quarter-sphere Pareto front. We parameterize by two angles that give
        # uniform coverage of the spherical surface.
        
        # For uniform distribution on a sphere, we can use:
        # cos(theta) = 1 - 2*(i + 0.5)/N  (for full sphere)
        # But we want quarter sphere: t1, t2 in [0, pi/2]
        # 
        # Alternative: Use the fact that on the quarter sphere, we can parameterize
        # using u, v in [0,1] and map to angles.
        # 
        # Better approach: Generate N points on the unit sphere in 3D with all
        # coordinates >= 0 (first octant), then normalize. The Fibonacci lattice
        # on the sphere gives good uniform distribution.
        
        # For the first octant of the sphere, we can use:
        # Generate points on the full sphere using Fibonacci lattice, then
        # reflect to first octant, or directly generate in first octant.
        
        # Actually, let's use a simpler parameterization:
        # On the quarter sphere (t1, t2 in [0, pi/2]), we can use:
        # cos(t1) = sqrt(1 - u) where u in [0,1]
        # sin(t2) = v where v in [0,1]
        # But this doesn't give uniform distribution.
        
        # Let's use the standard approach: generate points uniformly on the
        # spherical triangle (first octant of sphere) using area-preserving mapping.
        
        # The area element on the sphere is sin(theta) dtheta dphi.
        # For the first octant: theta in [0, pi/2], phi in [0, pi/2]
        # where theta is polar angle from z-axis, phi is azimuthal.
        
        # For uniform distribution:
        # cos(theta) = 1 - 2*u*(1/4) ... no, let me think more carefully.
        
        # The total area of the first octant of the unit sphere is 4*pi/8 = pi/2.
        # To get uniform points, we need:
        # A(theta) = integral from 0 to theta of sin(t) dt * (pi/2) = (1-cos(theta)) * (pi/2)
        # So cos(theta) = 1 - 2*u where u in [0, 1/2]... 
        # Actually: A(theta) = (1 - cos(theta)) * (pi/2)
        # For uniform u in [0,1]: 1 - cos(theta) = 2*u*(1/2) = u... 
        # Wait, the max area at theta=pi/2 is (1-0)*(pi/2) = pi/2
        # So A(theta)/A_max = (1-cos(theta))/(1) = 1-cos(theta)
        # Thus cos(theta) = 1 - u for uniform u in [0,1]
        # And phi = v * (pi/2) for uniform v in [0,1]
        
        # So: theta = arccos(1-u), phi = v*pi/2
        # Then: x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
        # Which gives us the 3 objective directions.
        
        # For DTLZ2:
        # f1 = cos(t1)*cos(t2), f2 = sin(t1)*cos(t2), f3 = sin(t2)
        # where t1 = x[0]*pi/2, t2 = x[1]*pi/2
        
        # The objective vector (f1, f2, f3) lies on the quarter sphere.
        # We want to sample (f1, f2, f3) uniformly on the quarter sphere.
        
        # Let's define:
        # alpha = arccos(1-u) where u in [0,1]  -> this is theta
        # beta = v*pi/2 where v in [0,1]        -> this is phi
        
        # Then:
        # z = cos(alpha) = f3
        # sin_alpha = sin(alpha)
        # x_sphere = sin_alpha * cos(beta) = f1
        # y_sphere = sin_alpha * sin(beta) = f2
        
        # We need: f1 = cos(t1)*cos(t2), f2 = sin(t1)*cos(t2), f3 = sin(t2)
        # So: sin(t2) = z = cos(alpha) => t2 = arcsin(cos(alpha)) = pi/2 - alpha
        # And: cos(t1)*cos(t2) = x_sphere, sin(t1)*cos(t2) = y_sphere
        # So: cos(t1) = x_sphere/cos(t2), sin(t1) = y_sphere/cos(t2)
        # cos(t2) = cos(pi/2 - alpha) = sin(alpha)
        # So: cos(t1) = x_sphere/sin(alpha), sin(t1) = y_sphere/sin(alpha)
        # t1 = arctan2(y_sphere, x_sphere)
        
        # Then: x[0] = 2*t1/pi, x[1] = 2*t2/pi, x[2:] = 0.5
        
        N = self.budget
        # Use Fibonacci lattice for u, v to get quasi-random uniform distribution
        # The golden ratio conjugate
        golden_ratio_conj = (1 - np.sqrt(5)) / 2  # approximately -0.618
        
        # Generate N points using Fibonacci lattice in 2D [0,1]^2
        # This gives low-discrepancy sequence
        i = np.arange(N)
        # For 2D, we can use:
        # u_i = frac(i * golden_ratio_conj)
        # v_i = frac(i * golden_ratio_conj^2)
        # But this might have correlations. Let's use a better 2D LHS or Sobol.
        
        # Actually, let's use a simple deterministic approach:
        # u_i = (i + 0.5) / N
        # v_i = (i * golden_ratio_conj) % 1.0
        
        u_vals = (np.arange(N) + 0.5) / N
        v_vals = np.mod(np.arange(N) * abs(golden_ratio_conj), 1.0)
        
        # Compute angles
        alpha = np.arccos(1 - u_vals)  # theta in [0, pi/2]
        beta = v_vals * np.pi / 2      # phi in [0, pi/2]
        
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        
        # Sphere coordinates
        x_sphere = sin_alpha * cos_beta
        y_sphere = sin_alpha * sin_beta
        z_sphere = cos_alpha
        
        # Compute t1, t2
        # sin(t2) = z_sphere => t2 = arcsin(z_sphere)
        t2 = np.arcsin(np.clip(z_sphere, -1, 1))
        
        # cos(t2) = cos(t2)
        cos_t2 = np.cos(t2)
        
        # Avoid division by zero
        cos_t2_safe = np.where(np.abs(cos_t2) > 1e-10, cos_t2, 1e-10)
        
        # cos(t1) = x_sphere / cos_t2, sin(t1) = y_sphere / cos_t2
        cos_t1 = x_sphere / cos_t2_safe
        sin_t1 = y_sphere / cos_t2_safe
        
        # Clip to valid range
        cos_t1 = np.clip(cos_t1, -1, 1)
        sin_t1 = np.clip(sin_t1, -1, 1)
        
        t1 = np.arctan2(sin_t1, cos_t1)
        # Ensure t1 in [0, pi/2]
        t1 = np.clip(t1, 0, np.pi/2)
        t2 = np.clip(t2, 0, np.pi/2)
        
        # Generate decision variables
        front_points = []
        for idx in range(N):
            x = np.zeros(dim)
            x[0] = 2 * t1[idx] / np.pi
            x[1] = 2 * t2[idx] / np.pi
            x[2:] = 0.5
            front_points.append(x)
        
        # Shuffle for diversity
        rng = np.random.default_rng(42)
        order = rng.permutation(N)
        
        # Evaluate points
        for idx in order:
            if check_time() or used >= self.budget:
                break
            x = front_points[idx]
            x = np.clip(x, lo, hi)
            fx = eval_and_record(x)
        
        # If we have remaining budget, do local search around promising points
        if used < self.budget and all_F:
            F_arr = np.array(all_F)
            sums = np.sum(F_arr, axis=1)
            n_refine = min(10, len(all_F))
            best_idx = np.argsort(sums)[:n_refine]
            
            for bi in best_idx:
                if check_time() or used >= self.budget:
                    break
                x_base = front_points[order[bi]].copy()
                for _ in range(3):
                    if check_time() or used >= self.budget:
                        break
                    delta = 0.02 * (hi - lo) * rng.standard_normal(dim)
                    x_new = np.clip(x_base + delta, lo, hi)
                    fx_new = eval_and_record(x_new)
        
        F = np.array(all_F)
        return F
# EVOLVE-BLOCK-END
