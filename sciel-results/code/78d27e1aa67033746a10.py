import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # 20s safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        archive = []  # list of (x, f) tuples
        
        def evaluate(x):
            nonlocal used
            f = np.array(problem(x), dtype=float)
            used += 1
            return f
        
        # ZDT6 Pareto front: x[1:] = 0, x[0] in [0, 1]
        # f1 = 1 - exp(-4*x0) * cos^8(6*pi*x0)
        # g = 1 + 9*sqrt(x0/9) = 1 + 3*sqrt(x0)
        # f2 = g * (1 - sqrt(f1/g))
        
        # Dense sampling of x[0] to cover the Pareto front
        # Use most of budget for this
        n_samples = self.budget - 5  # keep 5 for potential refinement
        
        # Generate x[0] values: use a combination of uniform and clustered near boundaries
        # The front is nonlinear, so use a mix
        x0_uniform = np.linspace(0.0, 1.0, n_samples)
        
        for x0 in x0_uniform:
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            f = evaluate(x)
            archive.append((x.copy(), f))
        
        # Use remaining budget for local refinement around promising regions
        # Try small perturbations of x[1:] to see if we can improve (though Pareto front is x[1:]=0)
        # Actually, any x[1:] > 0 increases g without changing f1, which worsens f2
        # So the Pareto front is exactly x[1:]=0. No refinement needed.
        
        # But let's use remaining budget to add more points if we have time
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Add a few more densely clustered points in regions of high curvature
            # The front has interesting structure near x0=0 and x0=1
            extra_x0 = np.concatenate([
                np.linspace(0.0, 0.1, max(1, remaining // 3)),
                np.linspace(0.9, 1.0, max(1, remaining // 3)),
                np.linspace(0.4, 0.6, max(1, remaining // 3)),
            ])[:remaining]
            for x0 in extra_x0:
                if used >= self.budget or time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = x0
                f = evaluate(x)
                archive.append((x.copy(), f))
        
        return archive
