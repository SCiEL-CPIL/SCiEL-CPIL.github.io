import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 structure:
        # Active variables: x[0], x[1]
        # Inactive variables: x[2:] = 0
        # The Pareto front is a 2D surface.
        # Objective 1: f0 = x[0]
        # Objective 2: f1 = (1-x[0]) * h(x[0], x[1])
        # Objective 3: f2 = (1-x[1]) * h(x[0], x[1])
        # where h(x, y) = 1 + 9 * sum(x[2:]) / (n-1) (which is 1 when inactive vars are 0)
        # So on the Pareto front (inactive vars = 0):
        # f0 = x0
        # f1 = (1-x0) * (1 + 9 * (1-x1) / (n-1)) ? No, standard DTLZ7 h is different.
        # Standard DTLZ7:
        # g(x) = 1 + 9 * sum(x[2:]) / (n-1)
        # h(y, g) = g * (1 - y[0]/g - exp(-y[0]/(4g)))
        # f_i = y_i * h(y, g) for i < m-1
        # f_{m-1} = 1 - h(y, g)
        
        # Wait, the problem statement says "minimizing all 3 objectives of DTLZ7".
        # Standard DTLZ7 has m objectives. Here n_obj=3.
        # So m=3.
        # y = [x0, x1]
        # g(x) = 1 + 9 * sum(x[2:]) / (n-1)
        # h(y, g) = g * (1 - y[0]/g - exp(-y[0]/(4g)))
        # f0 = y0 * h(y, g)
        # f1 = y1 * h(y, g)
        # f2 = 1 - h(y, g)
        
        # On Pareto front, g=1 (inactive vars 0).
        # h(y, 1) = 1 - y0 - exp(-y0/4)
        # f0 = y0 * (1 - y0 - exp(-y0/4))
        # f1 = y1 * (1 - y0 - exp(-y0/4))
        # f2 = 1 - (1 - y0 - exp(-y0/4)) = y0 + exp(-y0/4)
        
        # Let's verify the reference point: [1.1, 1.1, 25.0]
        # f2 max is at y0=0: 0 + 1 = 1.
        # So f2 is always <= 1.
        # f0 max is at y0=1: 1 * (1 - 1 - exp(-0.25)) = 0.
        # f0 is 0 at y0=0 and y0=1?
        # Let's check derivative of f0(y0) = y0 * (1 - y0 - exp(-y0/4)).
        # At y0=0, f0=0.
        # At y0=1, f0=0.
        # In between, it's positive?
        # exp(-y0/4) < 1 for y0>0.
        # 1 - y0 - exp(-y0/4).
        # At y0=0.5: 1 - 0.5 - exp(-0.125) = 0.5 - 0.882 = -0.382.
        # So f0 is negative?
        # But the reference point is [1.1, 1.1, 25.0].
        # If objectives can be negative, the hypervolume calculation changes.
        # However, usually DTLZ objectives are non-negative.
        # Let's re-read DTLZ7 definition.
        # Zitzler et al. 2002:
        # f_i = y_i * h(y, g) for i = 1, ..., m-1
        # f_m = 1 - h(y, g)
        # h(y, g) = g * (1 - y_1/g - exp(-y_1 / (4g)))
        
        # If y1 is small, h is large.
        # If y1 is close to g, h is small.
        # If g=1, y1 in [0,1].
        # h(y, 1) = 1 - y1 - exp(-y1/4).
        # Let's check values:
        # y1=0: h = 1 - 0 - 1 = 0.
        # y1=1: h = 1 - 1 - exp(-0.25) = -0.22.
        # This implies h can be negative.
        # If h is negative, f_i = y_i * h is negative.
        # f_m = 1 - h. If h is negative, f_m > 1.
        
        # Reference point [1.1, 1.1, 25.0].
        # If f0, f1 can be negative, they are "better" than 0.
        # But typically, we want to minimize.
        # The score is hypervolume with reference point.
        # Points must be strictly better than reference point.
        # If f0 < 1.1, f1 < 1.1, f2 < 25.0.
        
        # Let's look at the "inspiration" which mentions "points near the origin".
        # This suggests the front is in the positive orthant.
        # Maybe my DTLZ7 formula recall is slightly off or the specific variant is different.
        # However, the current program (score 1.78) uses x0, x1 in [0,1] and inactive 0.
        # And it gets a good score.
        # The key insight from inspiration 1 is: "wastes evaluations on points that contribute little to the hypervolume (points near the origin...)".
        # This implies the "origin" of the Pareto front (in objective space) is not the best place for hypervolume.
        # Hypervolume is maximized by points that are "dominating" or "extreme".
        # In a minimization problem with a reference point R, the hypervolume contribution of a point p is the volume of the box [p, R].
        # To maximize the union of these boxes, we want points that are far from R (small objectives) but spread out to cover the space.
        
        # A better strategy for hypervolume maximization on a known front is to sample points that are well-distributed in the objective space, 
        # but specifically biased towards the "extremes" or "knees" that define the boundary of the dominated region.
        
        # However, since we don't know the exact front shape without evaluating, and we have a budget of 200, 
        # we can use a simple adaptive approach:
        # 1. Start with a coarse grid or Sobol sequence.
        # 2. Evaluate them.
        # 3. Use the remaining budget to refine the regions where the "gaps" in the objective space are largest, 
        #    or simply use a more efficient low-discrepancy sequence that is mapped to the front.
        
        # Actually, the most effective strategy for hypervolume on a smooth front is often just a well-distributed set.
        # The current program uses Sobol. 
        # What is "materially different"?
        # 1. Random search? No, worse.
        # 2. Evolutionary algorithm? Might be too slow or inefficient for 200 evals.
        # 3. **Adaptive Sampling based on Hypervolume Contribution**:
        #    - Evaluate a small set of points (e.g., 20).
        #    - Calculate the hypervolume of this set.
        #    - Identify the "largest gap" in the objective space (e.g., using a simple grid or k-means in objective space).
        #    - Sample new points near the boundaries of these gaps.
        #    - Repeat until budget is exhausted.
        
        # Given the time constraint and simplicity, a "coarse-to-fine" Sobol approach might be better:
        # Use a Sobol sequence, but map it to the front in a way that emphasizes the "corners".
        # For DTLZ7, the front is 2D.
        # If we assume the front is roughly convex towards the origin, sampling uniformly in x0, x1 might not be uniform in objective space.
        # But without knowing the exact mapping, we can't easily re-parameterize.
        
        # Let's try a different angle: **Local Search / Hill Climbing**.
        # 1. Generate a small random set of points on the front (e.g., 10 points).
        # 2. Evaluate them.
        # 3. For the remaining budget, perform local perturbations around the best points found so far, 
        #    accepting new points if they improve the local hypervolume or simply if they are non-dominated.
        #    This allows us to "fill in" the gaps between the initial random points.
        
        # Strategy:
        # 1. Initial phase: 20 evaluations using Sobol on the 2 active variables.
        # 2. Refinement phase: 180 evaluations.
        #    - Keep a list of evaluated points (decision variables and objective values).
        #    - For each new evaluation, pick a "parent" from the current set (e.g., random or best).
        #    - Perturb the parent's active variables (x0, x1) with a small noise.
        #    - Evaluate the child.
        #    - If the child is non-dominated by the current set, add it to the set (or replace a dominated one if we maintain a archive).
        #    - This effectively performs a stochastic search that concentrates on promising regions.
        
        # This is materially different from pure global sampling (Sobol) because it adapts to the landscape.
        
        n_init = 20
        n_refine = self.budget - n_init
        
        # 1. Initial Sobol samples
        # Use a simple deterministic set for reproducibility and speed
        # We'll use a grid for the initial phase to ensure coverage
        # 2D grid: 5x4 = 20 points
        x0_vals = np.linspace(0.01, 0.99, 5)
        x1_vals = np.linspace(0.01, 0.99, 4)
        
        X_init = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                X_init.append(x)
        
        # Evaluate initial set
        F_init = []
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            F_init.append(f)
        
        # Archive: list of (x, f)
        archive = list(zip(X_init, F_init))
        
        # 2. Refinement phase
        # We will use a simple random restart + local perturbation approach.
        # To avoid getting stuck, we mix random new points with local perturbations of existing good points.
        
        # Simple heuristic for "good": non-dominated points.
        # But maintaining a full non-dominated front check is O(N^2). N is small (<=200), so it's fine.
        
        def is_dominated(f_new, archive_f):
            """Check if f_new is dominated by any point in archive_f."""
            for f_old in archive_f:
                if all(f_old <= f_new) and any(f_old < f_new):
                    return True
            return False
        
        def remove_dominated(new_point, archive):
            """Remove points from archive that are dominated by new_point."""
            new_f = new_point[1]
            new_archive = []
            for x_old, f_old in archive:
                # If old is dominated by new, discard old
                if all(new_f <= f_old) and any(new_f < f_old):
                    continue
                new_archive.append((x_old, f_old))
            new_archive.append(new_point)
            return new_archive

        # We will alternate between exploring (random) and exploiting (local search).
        # 50% random, 50% local.
        
        for i in range(n_refine):
            if time.time() > deadline:
                break
                
            if np.random.rand() < 0.5:
                # Explore: Random point on the front
                x = np.zeros(self.dim)
                x[0] = np.random.rand()
                x[1] = np.random.rand()
            else:
                # Exploit: Perturb a random point from the archive
                # Pick a random parent from the archive
                if archive:
                    parent_x, parent_f = np.random.choice(len(archive))
                    parent_x, _ = archive[parent_x]
                    x = parent_x.copy()
                    # Perturb x0 and x1
                    # Use a Gaussian noise with decaying sigma? Or fixed small sigma?
                    # Fixed small sigma is safer.
                    sigma = 0.05
                    x[0] += np.random.normal(0, sigma)
                    x[1] += np.random.normal(0, sigma)
                    # Clip to [0, 1]
                    x[0] = np.clip(x[0], 0.0, 1.0)
                    x[1] = np.clip(x[1], 0.0, 1.0)
                else:
                    x = np.zeros(self.dim)
                    x[0] = np.random.rand()
                    x[1] = np.random.rand()
            
            # Evaluate
            f = problem(x)
            
            # Update archive
            # Check if dominated
            if not is_dominated(f, [f for _, f in archive]):
                archive = remove_dominated((x, f), archive)
                
        return archive
