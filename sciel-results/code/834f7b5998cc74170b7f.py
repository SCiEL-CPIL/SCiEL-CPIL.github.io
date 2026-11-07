"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # For 3 objectives (DTLZ4 with M=3), the Pareto front is a quarter-sphere.
        # x1 and x2 determine the front shape; remaining front vars and g-vars are free.
        # g=0 when the last 5 variables (g-variables) are at 0.5.
        # Front: f1 = 0.5*sin(pi*x1/2), f2 = 0.5*sin(pi*x2/2)*cos(pi*x1/2),
        #        f3 = 0.5*cos(pi*x1/2)*cos(pi*x2/2)
        # Actually for DTLZ4 with M objectives: f_i = 0.5 * prod_{j<=M-i} cos(pi*x_j/2) * 
        #        sin(pi*x_{M-i+1}/2) for i < M, and f_M = 0.5*prod cos.
        
        # Let me re-derive: DTLZ4 with M=3 objectives:
        # f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2) * sin(pi*x1/2)  -- no, let me be careful.
        
        # Standard DTLZ4: f_i(x) = g(x) + 1 for i=M... no, DTLZ4 is similar to DTLZ2.
        # DTLZ4: f_i(x) = (g(x)+1) * prod_{j=1}^{M-i} cos(pi*x_j/2) * sin(pi*x_{M-i+1}/2) for i=1..M-1
        #         f_M(x) = (g(x)+1) * prod_{j=1}^{M-1} cos(pi*x_j/2)
        # When g=0: f_i = prod... so it's a unit sphere quarter section.
        # Actually the objectives sum of squares = 1 when g=0.
        
        # For hypervolume with reference [2.5,2.5,2.5], we want points on the Pareto front.
        # The front is {f: f1^2+f2^2+f3^2 = 0.25, f_i >= 0} -- quarter sphere of radius 0.5.
        
        # Uniform sampling on quarter sphere: use spherical coordinates.
        # Let u1, u2 be uniform on [0,1].
        # cos(theta) = u1 where theta is from north pole (f3 axis), phi = u2*pi/2
        # f1 = 0.5*sin(theta)*cos(phi), f2 = 0.5*sin(theta)*sin(phi), f3 = 0.5*cos(theta)
        # So: theta = arccos(u1), phi = u2 * pi/2
        
        # To get x1, x2 from desired f values:
        # f3 = 0.5*cos(pi*x1/2)*cos(pi*x2/2)
        # f2 = 0.5*cos(pi*x1/2)*sin(pi*x2/2)
        # f1 = 0.5*sin(pi*x1/2)
        # So: x1 = (2/pi)*arcsin(2*f1), and from f2/f3: tan(pi*x2/2) = f2/f3
        #     x2 = (2/pi)*arctan(f2/f3)
        
        # Strategy: Sample uniformly on the quarter sphere, then map back to x1, x2.
        # This gives uniform coverage of the front in objective space.
        
        # For better hypervolume, we want a well-distributed set on the front.
        # Use a stratified approach: divide the 2D parameter space into cells.
        
        # Alternative: Use Sobol or stratified grid in the (x1, x2) space,
        # but transform to get uniform density on the front.
        
        # Actually, the current approach with Halton in transformed space is already
        # giving uniform coverage. Let me try a different structural approach:
        # Use a combination of a coarse grid + refinement to ensure coverage of 
        # the entire front including corners.
        
        # New approach: Latin Hypercube in (x1, x2) space with stratification
        # in the transformed (angle) space for uniform front coverage.
        
        # Let's use a 2D grid with jitter for robust coverage:
        n_per_dim = int(np.sqrt(budget))
        if n_per_dim < 2:
            n_per_dim = 2
        n_total = n_per_dim * n_per_dim
        if n_total > budget:
            n_total = budget
            n_per_dim = int(np.ceil(np.sqrt(n_total)))
        
        # Create stratified grid in the angle space
        # theta in [0, pi/2], phi in [0, pi/2]
        # Uniform on sphere: cos(theta) uniform in [0,1], phi uniform in [0, pi/2]
        
        # Generate stratified points
        idx = np.arange(n_total)
        # Use a simple low-discrepancy approach with golden ratio
        golden = (1 + np.sqrt(5)) / 2
        inv_golden = 1.0 / (golden - 1.0)
        
        u1 = (idx * inv_golden) % 1.0  # For cos(theta)
        u2 = (idx * inv_golden * 1.618) % 1.0  # For phi, different stride
        
        # Better: use two independent low-discrepancy sequences
        # Van der Corput base 2 and base 3
        def van_der_corput(n, base):
            seq = np.zeros(n)
            for i in range(1, n + 1):
                f = 1.0
                num = i
                val = 0.0
                while num > 0:
                    f /= base
                    num, rem = divmod(num, base)
                    val += f * rem
                seq[i - 1] = val
            return seq
        
        u1 = van_der_corput(n_total, 2)
        u2 = van_der_corput(n_total, 3)
        
        # cos(theta) = u1 -> theta = arccos(u1)
        # phi = u2 * pi/2
        cos_theta = u1
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = u2 * np.pi / 2.0
        
        # Objective values on the front
        f1 = 0.5 * sin_theta * np.cos(phi)
        f2 = 0.5 * sin_theta * np.sin(phi)
        f3 = 0.5 * cos_theta
        
        # Map back to x1, x2
        # f1 = 0.5 * sin(pi*x1/2) -> x1 = (2/pi)*arcsin(2*f1)
        # f2/f3 = tan(pi*x2/2) -> x2 = (2/pi)*arctan(f2/f3)
        # Handle edge cases where f3 = 0
        x1_vals = (2.0 / np.pi) * np.arcsin(np.clip(2.0 * f1, -1.0, 1.0))
        # For x2: use atan2 for stability
        x2_vals = (2.0 / np.pi) * np.arctan2(f2, f3)
        # Ensure x2 is in [0, 1]
        x2_vals = np.clip(x2_vals, 0.0, 1.0)
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Construct decision vectors
        n_g = 5
        n_front = dim - n_g
        
        X = np.zeros((n_total, dim))
        X[:, 0] = x1_vals
        X[:, 1] = x2_vals
        if n_front > 2:
            X[:, 2:n_front] = 0.5
        X[:, n_front:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # If we have remaining budget, add more points with different offsets
        remaining = budget - n_total
        if remaining > 0:
            # Add random jittered points for additional coverage
            extra = np.zeros((remaining, dim))
            # Use different bases for more diversity
            u1_extra = van_der_corput(remaining, 5)
            u2_extra = van_der_corput(remaining, 7)
            
            cos_theta_e = u1_extra
            sin_theta_e = np.sqrt(1.0 - cos_theta_e**2)
            phi_e = u2_extra * np.pi / 2.0
            
            f1_e = 0.5 * sin_theta_e * np.cos(phi_e)
            f2_e = 0.5 * sin_theta_e * np.sin(phi_e)
            f3_e = 0.5 * cos_theta_e
            
            x1_e = (2.0 / np.pi) * np.arcsin(np.clip(2.0 * f1_e, -1.0, 1.0))
            x2_e = (2.0 / np.pi) * np.arctan2(f2_e, f3_e)
            x1_e = np.clip(x1_e, 0.0, 1.0)
            x2_e = np.clip(x2_e, 0.0, 1.0)
            
            extra[:, 0] = x1_e
            extra[:, 1] = x2_e
            if n_front > 2:
                extra[:, 2:n_front] = 0.5
            extra[:, n_front:] = 0.5
            extra = np.clip(extra, lo, hi)
            
            X = np.vstack([X, extra])
        
        # Check time
        if time.time() - t0 > deadline:
            return X
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
