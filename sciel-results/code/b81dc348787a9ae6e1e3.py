import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Parameters for Epsilon-MOEA
        # Epsilon box size for non-dominated sorting
        # We want roughly 5-10 boxes per objective for good resolution
        # Let's aim for a total of ~50-100 boxes in the hypervolume space
        # Since we don't know the exact range of objectives initially, 
        # we will use a relative epsilon based on the initial spread or a fixed small value.
        # A common heuristic: epsilon = (max - min) / N_boxes
        # For 3 objectives, if we want 10 boxes per dim, N=1000. 
        # Let's start with a slightly coarser grid to ensure we don't reject everything.
        # We will dynamically adjust or use a fixed small epsilon.
        
        # Let's use a fixed epsilon based on the known reference point and typical WFG4 ranges.
        # WFG4 objectives are usually in [0, 10] or similar. 
        # Reference point is [3.0, 5.0, 7.0].
        # Let's use epsilon = 0.1 for each objective. This creates 30x50x70 = 10500 boxes.
        # That might be too fine. Let's try 0.5. 6x10x14 = 840 boxes.
        # Let's try 0.2. 15x25x35 = 13125 boxes.
        # Actually, let's use a simpler approach: Keep a list of non-dominated points.
        # Use "epsilon box" to reject points that are too close to existing ones.
        
        eps = 0.2  # Box size
        
        # Archive: list of (x, f)
        archive = []
        
        # Helper: Check if new point is dominated by any in archive
        # If dominated, reject.
        # If it dominates existing, remove existing.
        # If non-dominated but in same epsilon box as existing, keep the one with better hypervolume contribution? 
        # Or just keep the first one? 
        # Standard Epsilon-MOEA: If in same box, compare by a secondary criterion (e.g., sum of objectives or distance to ideal).
        # Let's use sum of objectives as secondary.
        
        def is_dominated(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def in_same_box(f1, f2, eps):
            # Check if f1 and f2 fall into the same epsilon box
            # Box index = floor(f / eps)
            # If indices are same, they are in the same box.
            return np.all(np.floor(f1 / eps) == np.floor(f2 / eps))

        def add_to_archive(x_new, f_new):
            nonlocal archive
            
            # 1. Check if f_new is dominated by any point in archive
            for i in range(len(archive)):
                f_old = archive[i][1]
                if is_dominated(f_new, f_old):
                    return False # Reject

            # 2. Check if f_new dominates any point in archive
            # Also check epsilon box conflict
            new_archive = []
            for i in range(len(archive)):
                x_old, f_old = archive[i]
                if is_dominated(f_old, f_new):
                    continue # f_new dominates f_old, so f_old is removed
                elif in_same_box(f_new, f_old, eps):
                    # In same box, non-dominated. Keep the one with better secondary metric.
                    # Secondary: sum of objectives (lower is better)
                    if np.sum(f_new) < np.sum(f_old):
                        continue # f_new is better, remove f_old
                    else:
                        return False # f_old is better or equal, reject f_new
                else:
                    new_archive.append((x_old, f_old))
            
            new_archive.append((x_new, f_new))
            archive = new_archive
            return True

        # 1. Initialization
        # Use Latin Hypercube Sampling for initial spread
        pop_size_init = 30
        X_init = np.zeros((pop_size_init, self.dim))
        for i in range(self.dim):
            samples = np.linspace(0, 1, pop_size_init, endpoint=False)
            np.random.shuffle(samples)
            X_init[:, i] = samples
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            add_to_archive(x, f)
        
        used = pop_size_init
        
        # 2. Evolution
        # We will use a simple evolutionary loop that generates new candidates
        # by perturbing points from the archive.
        
        # To ensure we explore well, we can mix:
        # 1. Mutation of random archive points
        # 2. Crossover of random archive points
        
        eta_c = 20.0
        eta_m = 30.0
        
        def sbx_crossover(x1, x2, eta):
            x3 = np.zeros(self.dim)
            x4 = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if np.random.rand() <= 1.0 / (eta + 1):
                            beta = (2 * np.random.rand()) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (eta + 1))
                        x3[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        x4[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    else:
                        x3[i] = x1[i]
                        x4[i] = x2[i]
                else:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
            return x3, x4

        def polynomial_mutation(x, eta, lo, hi):
            x_new = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    if np.random.rand() <= 2.0 / (eta + 1):
                        delta1 = (2 * np.random.rand()) ** (1.0 / (eta + 1)) - 1
                        beta = 1 + delta1 * (x[i] - lo[i]) / (hi[i] - lo[i])
                    else:
                        delta2 = 1 - (2 * (1 - np.random.rand())) ** (1.0 / (eta + 1))
                        beta = 1 + delta2 * (hi[i] - x[i]) / (hi[i] - lo[i])
                    x_new[i] = x[i] + beta * (hi[i] - lo[i])
            x_new = np.clip(x_new, lo, hi)
            return x_new

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate 2 offspring per iteration to keep pace
            # Select parents from archive
            if len(archive) < 2:
                # If archive is small, just mutate random points
                x_parent1 = lo + (hi - lo) * np.random.rand(self.dim)
                x_parent2 = lo + (hi - lo) * np.random.rand(self.dim)
            else:
                idx1 = np.random.randint(0, len(archive))
                idx2 = np.random.randint(0, len(archive))
                x_parent1 = archive[idx1][0]
                x_parent2 = archive[idx2][0]
                
                # Crossover
                c1, c2 = sbx_crossover(x_parent1, x_parent2, eta_c)
                # Mutation
                c1 = polynomial_mutation(c1, eta_m, lo, hi)
                c2 = polynomial_mutation(c2, eta_m, lo, hi)
            
            # Evaluate c1
            f1 = problem(c1)
            used += 1
            add_to_archive(c1, f1)
            
            if used >= self.budget:
                break
                
            # Evaluate c2
            f2 = problem(c2)
            used += 1
            add_to_archive(c2, f2)
            
            # Optional: If archive grows too large, we might need to prune more aggressively,
            # but with epsilon boxes it should stay bounded.
            
        # Return the best non-dominated points found
        # The evaluator records ALL returned objective vectors.
        # We should return the final archive's objective values.
        if not archive:
            return np.array([])
            
        F_archive = np.array([a[1] for a in archive])
        return F_archive
