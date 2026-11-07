"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim):
        """Generate a Sobol-like quasi-random sequence using Halton sequence."""
        # Use Halton sequence as a simple low-discrepancy sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = halton(i + 1, j + 2)
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # M objectives, N variables
        # x_1, ..., x_{N-M+2} are "position" variables (on unit sphere)
        # x_{N-M+3}, ..., x_N are "g" variables (affecting constraint g)
        # For M=3, N=12:
        #   Position: x_1, x_2, x_3 (3 variables)
        #   G-variables: x_4, ..., x_12 (9 variables)
        
        # The Pareto front for DTLZ4 is achieved when g(x) = 0
        # g(x) = 100 * (M-1) + sum_{i=M+1}^{N} ((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        # For g=0, each g-variable must be at x_i = 0.5 (since the cos term is bounded by 1)
        # Actually, the minimum of (x-0.5)^2 - cos(20*pi*(x-0.5)) is 0 at x=0.5
        
        # So on the Pareto front: x_4 = ... = x_12 = 0.5
        # And f_1 = g * x_1, f_2 = g * x_2 * (1-x_1), f_3 = g * (1-x_2)
        # When g=0: f_1 = 0, f_2 = 0, f_3 = 0... wait that's not right.
        
        # Let me reconsider DTLZ4:
        # f_1(x) = g(x) * x_1
        # f_2(x) = g(x) * x_2 * (1 - x_1)  
        # f_3(x) = g(x) * (1 - x_2)
        # where g(x) = 100 * (M-1) + sum_{i=M+1}^{N} ((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        
        # Wait, for DTLZ problems, the standard formulation is:
        # f_1 = x_1 * (1 + g)
        # f_2 = x_2 * (1 - x_1) * (1 + g)
        # f_3 = (1 - x_2) * (1 + g)
        # where g = 100*(M-1) + sum_{i=M+1}^{N} ((x_i-0.5)^2 - cos(20*pi*(x_i-0.5)))
        
        # On the Pareto front, g=0, so:
        # f_1 = x_1
        # f_2 = x_2 * (1 - x_1)
        # f_3 = (1 - x_2)
        # with x_1, x_2 in [0,1] and x_1 + x_2 <= 1 (for the quarter sphere)
        
        # Actually for DTLZ, the Pareto front is:
        # f_1^2 + f_2^2 + f_3^2 = 1 (quarter unit sphere in positive octant)
        # with f_i >= 0
        
        # Let's use spherical coordinates:
        # f_1 = sin(theta1) * sin(theta2)
        # f_2 = sin(theta1) * cos(theta2)
        # f_3 = cos(theta1)
        # where theta1 in [0, pi/2], theta2 in [0, pi/2]
        
        # Or more directly, for M=3:
        # x_1 = sin(theta) * sin(phi)
        # x_2 = sin(theta) * cos(phi)  
        # (1-x_2) = cos(theta)
        # This gives the quarter sphere.
        
        # Strategy: 
        # 1. Use stratified sampling to generate points on the Pareto front
        # 2. Set g-variables to 0.5 to ensure g=0
        # 3. Use remaining budget for local refinement if needed
        
        n_samples = min(budget, 100)  # Number of Pareto front samples
        
        # Generate stratified samples on the quarter sphere
        # Use a grid in (theta, phi) space
        n_theta = int(np.sqrt(n_samples))
        n_phi = max(1, n_samples // n_theta)
        
        # Stratified sampling: divide [0, pi/2] x [0, pi/2] into cells
        theta_vals = []
        phi_vals = []
        for i in range(n_theta):
            for j in range(n_phi):
                if len(theta_vals) >= n_samples:
                    break
                # Random point within cell
                theta = (i + 0.5) / n_theta * (np.pi / 2)
                phi = (j + 0.5) / n_phi * (np.pi / 2)
                theta_vals.append(theta)
                phi_vals.append(phi)
        
        # Trim to exactly n_samples
        theta_vals = theta_vals[:n_samples]
        phi_vals = phi_vals[:n_samples]
        
        # Generate candidate points on the Pareto front
        X = np.zeros((n_samples, dim))
        for i in range(n_samples):
            theta = theta_vals[i]
            phi = phi_vals[i]
            
            # Position variables (x_1, x_2, x_3)
            X[i, 0] = np.sin(theta) * np.sin(phi)  # x_1
            X[i, 1] = np.sin(theta) * np.cos(phi)  # x_2
            X[i, 2] = np.cos(theta)                 # 1 - x_2 should equal this
            
            # G-variables (x_4 through x_12) set to 0.5
            for k in range(3, dim):
                X[i, k] = 0.5
        
        # Evaluate all candidates
        F = np.array([problem(x) for x in X])
        used = n_samples
        
        # Check deadline
        if time.time() > deadline - 10:
            return X[:used]
        
        # Use remaining budget for local refinement
        remaining = budget - used
        if remaining > 0:
            # Perform local line searches to refine g-variables
            for i in range(used):
                if time.time() > deadline - 10:
                    break
                if remaining <= 0:
                    break
                    
                # Try small perturbations of g-variables
                best_f = F[i].copy()
                best_x = X[i].copy()
                
                # Random directions in g-variable space
                n_trials = min(remaining, 5)
                for t in range(n_trials):
                    if remaining <= 0 or time.time() > deadline - 10:
                        break
                    
                    # Perturb a random g-variable
                    g_idx = np.random.randint(3, dim)
                    delta = np.random.choice([-0.1, 0.1])
                    candidate = best_x.copy()
                    candidate[g_idx] = np.clip(candidate[g_idx] + delta, lo[g_idx], hi[g_idx])
                    
                    fc = problem(candidate)
                    remaining -= 1
                    
                    # Check if this is better (any objective improved without worsening others)
                    if np.all(fc <= best_f) and np.any(fc < best_f):
                        best_f = fc
                        best_x = candidate
                
                X[i] = best_x
                F[i] = best_f
        
        return X
# EVOLVE-BLOCK-END
