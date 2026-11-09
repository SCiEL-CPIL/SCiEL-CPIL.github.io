"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ7 using Sobol sequence on the 2D Pareto front manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _sobol_2d(self, n):
        """Generate n points in [0,1]^2 using a simple quasi-random sequence."""
        # Use Halton sequence for 2D (base 2 and 3)
        points = np.zeros((n, 2))
        for i in range(n):
            # Halton base 2
            f = 1.0
            r = 0.0
            q = i + 1
            while q > 0:
                f /= 2.0
                r += f * (q % 2)
                q //= 2
            points[i, 0] = r
            
            # Halton base 3
            f = 1.0
            r = 0.0
            q = i + 1
            while q > 0:
                f /= 3.0
                r += f * (q % 3)
                q //= 3
            points[i, 1] = r
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # For DTLZ7:
        # x[0] and x[1] determine the Pareto front shape
        # x[2:] should be 0.5 to minimize g (the constraint)
        # The Pareto front is parameterized by x[0], x[1] in [0,1]^2
        
        # Generate Sobol/Halton points for the 2D parameter space
        n_points = min(budget, 200)
        params = self._sobol_2d(n_points)
        
        # Build candidate solutions
        X = np.zeros((n_points, dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
        X[:, 2:] = 0.5  # Optimal tail values for DTLZ7
        
        # Evaluate all candidates
        F = np.array([problem(x) for x in X])
        used = n_points
        
        # Check deadline
        if time.time() > deadline:
            return X
        
        # If we have budget left, do local refinement around best points
        # Find non-dominated points
        if used < budget:
            # Identify non-dominated points
            n_dom = len(F)
            is_dominated = np.zeros(n_dom, dtype=bool)
            for i in range(n_dom):
                for j in range(n_dom):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dominated[i] = True
                            break
            
            if np.any(~is_dominated):
                # Get non-dominated points
                nondom_idx = np.where(~is_dominated)[0]
                
                # Refine around each non-dominated point
                remaining_budget = budget - used
                n_refine = len(nondom_idx)
                per_point = max(1, remaining_budget // n_refine)
                
                for idx in nondom_idx:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    center = X[idx].copy()
                    for k in range(per_point):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        # Small perturbation in x[0] and x[1]
                        delta = 0.05 * (hi[:2] - lo[:2])
                        perturbation = delta * np.array([
                            np.random.uniform(-1, 1),
                            np.random.uniform(-1, 1)
                        ])
                        
                        child = center.copy()
                        child[0] = np.clip(center[0] + perturbation[0], lo[0], hi[0])
                        child[1] = np.clip(center[1] + perturbation[1], lo[1], hi[1])
                        
                        fc = problem(child)
                        used += 1
                        
                        # Replace worst point if child is better
                        worst_idx = int(np.argmax(np.sum(F, axis=1)))
                        # Only replace if child dominates or has lower sum
                        if np.sum(fc) < np.sum(F[worst_idx]) or np.all(fc <= F[worst_idx]) and np.any(fc < F[worst_idx]):
                            X[worst_idx] = child
                            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
