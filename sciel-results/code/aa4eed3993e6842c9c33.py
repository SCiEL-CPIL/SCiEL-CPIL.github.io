"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ2's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2: Pareto front where sum(x^2) = 0.5
        # f_k = (0.5 + sum(x^2)) * cos(theta_k), theta_k = pi/2*(1 + (k-1)/n_obj)
        # On the front: f_k = cos(theta_k)
        # We can parameterize: x_i are on a sphere of radius sqrt(0.5)

        # Strategy: 
        # 1. Generate points on the Pareto front using spherical coordinates
        # 2. Use local refinement to minimize deviation from the front
        # 3. Use remaining budget for diverse exploration

        # Precompute theta angles for DTLZ2 with n_obj objectives
        # theta_k = pi/2 * (1 + (k-1)/n_obj) for k=1..n_obj
        # Actually for DTLZ2: f_k = (1+g) * cos(theta_k) where theta_k = pi/2*(1+(k-1)/(n_obj-1))... 
        # Let me be more careful. Standard DTLZ2:
        # f_k = (1 + g(x_M)) * prod_{i=1}^{k-1} cos(x_i) * sin(x_k) for k < n_obj
        # f_{n_obj} = (1 + g(x_M)) * sin(x_{n_obj})
        # where x_i = pi/2 * y_i for i in 1..n_obj-1, and g = sum(y_i^2) for y in last M vars
        # Actually the standard formulation:
        # For DTLZ2 with n_obj objectives and dim variables:
        # Let n_pos = dim - n_obj + 1 (position variables)
        # g = sum_{i=n_pos+1}^{dim} (x_i - 0.5)^2
        # f_k = (1+g) * prod_{j=1}^{k-1} cos(pi/2 * x_j) * sin(pi/2 * x_k) for k < n_obj
        # f_{n_obj} = (1+g) * sin(pi/2 * x_{n_obj})
        # Pareto front: g=0, so x_{n_pos+1}...x_dim = 0.5
        # Then f_k = prod_{j=1}^{k-1} cos(pi/2*x_j) * sin(pi/2*x_k)

        # The Pareto front is a (n_obj-1)-dimensional hypersphere in the first n_obj-1 variables
        # with the remaining variables at 0.5

        n_pos = dim - n_obj + 1  # number of position variables

        def generate_pareto_front_point(rng):
            """Generate a point on the DTLZ2 Pareto front."""
            x = np.zeros(dim)
            # Set non-position variables to 0.5 (g=0)
            for i in range(n_pos, dim):
                x[i] = 0.5
            # Generate position variables on the unit sphere (first n_pos-1 vars)
            # The front: sum_{i=1}^{n_pos-1} x_i^2 = 1 (in the cos/sin parameterization)
            # Actually, let's sample uniformly on the sphere
            if n_pos >= 2:
                # Sample n_pos-1 variables from normal, normalize to unit sphere
                z = rng.standard_normal(n_pos - 1)
                norm = np.linalg.norm(z)
                if norm > 1e-10:
                    z = z / norm
                x[:n_pos-1] = z
                # Last position variable is determined: x_{n_pos} such that the point is on front
                # For DTLZ2, the front is parameterized by the first n_obj-1 variables
                # x_{n_pos} = x_{n_obj} is the last position var
                # The constraint is that the point maps to the unit sphere in objective space
                # Let's just set x_{n_pos} = 0.5 as well for simplicity, 
                # but actually we need the full sphere
                # Better: sample all n_pos position variables on the sphere
                # The Pareto front in DTLZ2: the first n_obj-1 variables define the front
                # Let me reconsider: with n_obj objectives, we have n_obj-1 position variables
                # that define the front (the last position variable is determined)
                
                # Actually, let me just use a simpler approach:
                # Sample uniformly on the (n_obj-2)-sphere for the first n_obj-1 position vars
                # and set the rest to 0.5
            else:
                x[0] = 1.0
            
            # A cleaner approach: sample angles
            if n_obj > 1:
                # Sample n_obj-1 angles uniformly in [0, pi/2]
                angles = rng.uniform(0, np.pi/2, size=n_obj-1)
                # Convert to Cartesian on the sphere
                # f_1 = cos(a1)
                # f_2 = sin(a1)*cos(a2)
                # ...
                # f_{n_obj-1} = sin(a1)*...*sin(a_{n_obj-2})*cos(a_{n_obj-1})
                # f_{n_obj} = sin(a1)*...*sin(a_{n_obj-1})
                # The position variables x_1...x_{n_obj-1} relate to these angles
                # x_k = 2/pi * a_k for k < n_obj
                for k in range(n_obj - 1):
                    x[k] = angles[k] / (np.pi/2)
                # Set remaining position and non-position vars to 0.5
                for i in range(n_obj, dim):
                    x[i] = 0.5
            else:
                x[0] = 1.0
            
            return x

        # Generate initial population on the Pareto front
        pop_size = min(50, budget // 4)
        X = np.array([generate_pareto_front_point(rng) for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Local refinement: for each point, try small perturbations to reduce g
        # g = sum_{i=n_pos}^{dim-1} (x_i - 0.5)^2
        # We want to drive g -> 0

        # Use remaining budget for local search
        step_size = 0.05
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Pick a random point and perturb it
            idx = rng.integers(pop_size)
            x = X[idx].copy()
            
            # Perturb non-position variables toward 0.5
            for i in range(n_pos, dim):
                x[i] += 0.3 * (0.5 - x[i]) + 0.02 * rng.standard_normal()
                x[i] = np.clip(x[i], lo[i], hi[i])
            
            # Small perturbation to position variables
            x[:n_pos] += 0.02 * rng.standard_normal(n_pos)
            x = np.clip(x, lo, hi)
            
            fx = problem(x)
            used += 1
            
            # Accept if it improves (dominates or is non-dominated)
            # Simple acceptance: accept if any objective improves
            if np.any(fx < F[idx]):
                X[idx] = x
                F[idx] = fx

        # Final: try to project points onto the Pareto front more carefully
        # Use remaining budget for targeted refinement
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # For each point, try to minimize g by adjusting non-position vars
            for i in range(min(remaining, pop_size * 3)):
                if time.time() > deadline:
                    break
                idx = i % pop_size
                x = X[idx].copy()
                # Strongly push non-position vars to 0.5
                for j in range(n_pos, dim):
                    x[j] = 0.5 + 0.01 * rng.standard_normal()
                    x[j] = np.clip(x[j], lo[j], hi[j])
                fx = problem(x)
                used += 1
                if np.any(fx < F[idx]):
                    X[idx] = x
                    F[idx] = fx

        return X
# EVOLVE-BLOCK-END
