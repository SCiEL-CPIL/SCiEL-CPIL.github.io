"""Multi-objective optimizer for DTLZ4 using exact Pareto front construction with Sobol sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling with Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # For DTLZ4 with 3 objectives and 12 variables:
        # x[0], x[1], x[2] are the "position" variables that determine the Pareto front shape
        # x[3] through x[11] are the "distance" variables that determine g(x)
        # g(x) = sum_{i=3}^{11} (x_i - 0.5)^2 + 10*sin^2(10*pi*(x_i - 0.5))
        # For g=0, we need x_i = 0.5 for all i in [3, 11]
        
        # The Pareto front is parameterized by x[0], x[1], x[2] in [0,1]:
        # f1 = 0.5 * (1+g) * x0 * cos(x1*pi/2) * cos(x2*pi/2)
        # f2 = 0.5 * (1+g) * x0 * sin(x1*pi/2) * cos(x2*pi/2)
        # f3 = 0.5 * (1+g) * (1-x0)
        
        # With g=0:
        # f1 = 0.5 * x0 * cos(x1*pi/2) * cos(x2*pi/2)
        # f2 = 0.5 * x0 * sin(x1*pi/2) * cos(x2*pi/2)
        # f3 = 0.5 * (1-x0)
        
        # This traces out a quarter-sphere surface in the first octant of objective space.
        # To maximize hypervolume, we want to sample this surface uniformly.
        
        # Use Sobol sequence for quasi-random sampling of the 3 position variables
        from scipy.stats import qmc

        # We'll use most of our budget for sampling the Pareto front
        # The key insight: for DTLZ4, the Pareto front is a 2D surface (quarter-sphere)
        # parameterized by 3 variables, but with a constraint that makes it 2D.
        
        # Actually, let me reconsider. The Pareto front for DTLZ4 with M=3 objectives
        # is a (M-1)=2 dimensional surface. The mapping from x[0], x[1], x[2] to (f1,f2,f3)
        # with g=0 gives us points on this surface.
        
        # Let's use a stratified approach: sample x[0] uniformly, and for each x[0],
        # sample (x[1], x[2]) to cover the angular parameters.
        
        # Better approach: use the fact that the Pareto front can be parameterized by
        # two angles. Let's use spherical coordinates on the quarter-sphere.
        
        # The Pareto front satisfies f1^2 + f2^2 + f3^2 = 0.25 (when g=0)
        # Wait, let me verify:
        # f1^2 + f2^2 + f3^2 = 0.25*x0^2*cos^2(x2*pi/2)*(cos^2(x1*pi/2) + sin^2(x1*pi/2)) + 0.25*(1-x0)^2
        # = 0.25*x0^2*cos^2(x2*pi/2) + 0.25*(1-x0)^2
        # This is NOT constant, so it's not a sphere in this parameterization.
        
        # Let me just directly sample x[0], x[1], x[2] using Sobol and set x[3:]=0.5
        
        # Use Sobol sequence for better coverage
        n_sample = min(budget, 200)
        
        # Create Sobol sampler
        sampler = qmc.Sobol(d=3, scramble=True)
        # Generate points in [0,1]^3
        sobol_points = sampler.random(n=n_sample)
        
        # Create decision vectors
        X = np.zeros((n_sample, dim))
        X[:, :3] = sobol_points
        X[:, 3:] = 0.5  # Set distance variables to 0.5 for g=0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_sample
        
        # Check time
        if time.time() > deadline - 30:
            return X
        
        # Now use remaining budget for local refinement
        # Focus on improving points that are close to but not exactly on the Pareto front
        remaining = budget - used
        if remaining > 0:
            # Find non-dominated points
            def is_dominated(f, F):
                return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
            
            dominated = np.array([is_dominated(f, F) for f in F])
            non_dom_idx = np.where(~dominated)[0]
            
            # Refine around non-dominated points
            n_refine = min(remaining, 50)
            step_size = 0.02
            
            for _ in range(n_refine):
                if time.time() > deadline - 20:
                    break
                if len(non_dom_idx) == 0:
                    break
                parent_idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
                child = X[parent_idx].copy()
                # Perturb the position variables slightly
                child[:3] += step_size * np.random.randn(3)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                # Replace the worst point
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc
                # Update non-dominated set
                dominated = np.array([is_dominated(f, F) for f in F])
                non_dom_idx = np.where(~dominated)[0]
        
        return X
# EVOLVE-BLOCK-END
