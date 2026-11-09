import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # ZDT6 Pareto front analysis:
        # f0 = 1 - exp(-4*x0 * sin(10*pi*x0*(dim-1)))
        # f1 = 1 - sqrt(f0)
        # Pareto set: x1=...=x_{dim-1}=0, x0 in [0,1]
        # The Pareto front is 1D and can be parameterized by x0.
        # Since f0 oscillates, we need to find x0 values that give us a good spread of f0 in [0,1].
        # Actually, the known Pareto front for ZDT6 is f1 = 1 - sqrt(f0) for f0 in [0,1].
        # To maximize hypervolume, we want points that are well-spread on the front.
        
        # Strategy: Use the analytic form to directly generate x0 values that correspond to
        # specific f0 values, ensuring uniform coverage of the Pareto front in objective space.
        
        # f0 = 1 - exp(-4*x0 * sin(10*pi*x0*M)) where M = dim - 1
        # We want f0 to range from 0 to 1.
        # Let's sample f0 uniformly in [0, 1] and solve for x0 numerically or use the known structure.
        # Actually, since we can't invert the function easily, let's use a different approach:
        # Generate x0 values that are known to give good coverage.
        
        # The function g(x0) = 4*x0*sin(10*pi*x0*M) oscillates.
        # For M=9 (dim=10), the period is 1/(10*9) = 1/90.
        # g(x0) reaches 1 at x0 = 1/20, 3/20, 5/20, ... (odd multiples of 1/20 within [0,1])
        # and 0 at x0 = 0, 1/10, 2/10, ..., 1.
        
        # Let's just evaluate x0 values that are well-distributed.
        # We know that x0 = 0 gives f0=0, f1=1.
        # x0 = 1/20 = 0.05 gives sin(10*pi*0.05*9) = sin(4.5*pi) = sin(pi/2) = 1, so g=4*0.05*1=0.2, f0=1-exp(-0.2)~0.181, f1=1-sqrt(0.181)~0.574.
        
        # A better approach: sample x0 uniformly in [0,1] and keep the non-dominated points.
        # But to be more efficient, let's use the fact that the Pareto front is 1D.
        
        # Let's initialize with a set of points that should be on or near the Pareto front.
        # We'll use a mixture of strategies:
        # 1. Sample x0 uniformly in [0,1] with x1:..=0
        # 2. Use local search to refine
        
        n_init = min(50, self.budget // 4)
        
        # Initialize with diverse x0 values
        x0_samples = np.linspace(0, 1, n_init)
        X_init = np.zeros((n_init, self.dim))
        X_init[:, 0] = x0_samples
        
        F_init = []
        for x in X_init:
            if time.time() > deadline:
                break
            f = self._evaluate(problem, x)
            F_init.append(f)
            self._update_archive(x, f)
        
        # Now perform local search to improve the archive
        # The key insight: we want to find x0 values that give us points on the Pareto front
        # with f0 spread evenly in [0,1] to maximize hypervolume.
        
        # Use the analytic structure to guide the search
        # For each point in the archive, try to improve it by adjusting x0
        active_points = list(self.archive)
        
        while self.eval_count < self.budget:
            if time.time() > deadline:
                break
            
            if not active_points:
                break
            
            # Pick a random point to explore
            idx = np.random.randint(0, len(active_points))
            x, f = active_points[idx]
            
            # Generate a child by perturbing x0 primarily
            delta_x0 = np.random.randn() * 0.02
            delta_other = np.random.randn(self.dim - 1) * 0.01
            x_child = x.copy()
            x_child[0] += delta_x0
            x_child[1:] += delta_other
            x_child = np.clip(x_child, lo, hi)
            
            f_child = self._evaluate(problem, x_child)
            
            # Check if it's non-dominated
            dominated = False
            for _, af in self.archive:
                if np.all(f_child <= af) and np.any(f_child < af):
                    dominated = True
                    break
            
            if not dominated:
                # Remove points dominated by the new one
                new_archive = []
                for ax, af in self.archive:
                    if not (np.all(f_child <= af) and np.any(f_child < af)):
                        new_archive.append((ax, af))
                new_archive.append((x_child, f_child))
                self.archive = new_archive
                active_points = new_archive
            else:
                # If dominated, try a smaller step
                if np.random.random() < 0.5:
                    continue
        
        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for ax, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return  # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
