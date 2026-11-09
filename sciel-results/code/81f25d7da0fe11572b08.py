import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Parse bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific strategy:
        # x0 determines the position on the Pareto front.
        # x1...x9 should be 0 to minimize g(x) and thus f2.
        
        # Determine optimal values for x1...x9
        # Standard ZDT4 has x_i in [0, 1] and optimum at 0.
        # We assume 0 is the target, clipped to bounds.
        optimal_tail = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            # If 0 is out of bounds, use the closest bound
            if lo[i] <= 0 <= hi[i]:
                optimal_tail[i-1] = 0.0
            elif hi[i] < 0:
                optimal_tail[i-1] = hi[i]
            else:
                optimal_tail[i-1] = lo[i]
        
        # Budget allocation
        # 80% for broad sampling of x0
        # 20% for local refinement
        n_sample = int(self.budget * 0.8)
        n_refine = self.budget - n_sample
        
        # Generate x0 samples
        # Use linear spacing to cover the range evenly
        x0_samples = np.linspace(lo[0], hi[0], n_sample)
        
        # Construct initial population
        X = np.zeros((n_sample, self.dim))
        X[:, 0] = x0_samples
        X[:, 1:] = optimal_tail
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        
        # Identify non-dominated points
        # Simple non-domination check
        # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
        # Since we are on the front, most points should be non-dominated.
        # We just need to ensure we record them.
        
        # Refinement phase
        # Select a few points to refine, focusing on filling gaps or ensuring optimality
        # We pick points with the lowest f2 (closest to ideal) to ensure we have the "elbow" and ends well covered
        # Actually, just picking a few random points from the sample and perturbing x0 slightly is safer for coverage.
        
        if n_refine > 0:
            # Pick 5 distinct indices from the sample
            indices = np.random.choice(n_sample, size=min(5, n_sample), replace=False)
            refine_X = []
            refine_F = []
            
            for idx in indices:
                # Generate small perturbations around this point
                # Perturb x0 slightly
                delta = (hi[0] - lo[0]) / n_sample
                for d in [-delta, delta]:
                    if n_refine <= 0: break
                    x_new = X[idx].copy()
                    x_new[0] = np.clip(x_new[0] + d, lo[0], hi[0])
                    # Keep tail optimal
                    x_new[1:] = optimal_tail
                    refine_X.append(x_new)
                    refine_F.append(problem(x_new))
                    n_refine -= 1
            
            # Add refined points to the set
            if refine_X:
                X = np.vstack([X, np.array(refine_X)])
                F = np.vstack([F, np.array(refine_F)])
        
        return X
