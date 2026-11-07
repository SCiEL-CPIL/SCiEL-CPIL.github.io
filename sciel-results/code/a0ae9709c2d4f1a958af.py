import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        archive = [] # List of (x, f)
        evals = 0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f):
            nonlocal archive
            # Check if x is dominated by any in archive
            for ax, af in archive:
                if dominates(af, f):
                    return False
            # Remove points dominated by x
            new_archive = []
            for ax, af in archive:
                if not dominates(f, af):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive = new_archive
            return True

        # Initial phase: Random sampling to populate archive
        # Let's use 20% of budget for random sampling to get good spread
        init_budget = max(10, budget // 5)
        
        for _ in range(init_budget):
            x = lo + (hi - lo) * np.random.rand(dim)
            f = np.array(problem(x))
            evals += 1
            add_to_archive(x, f)
            
        if evals >= budget:
            return [x for x, f in archive]

        # Local Search Phase
        # We will perform local search steps.
        # Strategy: Pick a point from archive, perturb it, evaluate, update archive.
        # To handle disconnected fronts, we occasionally pick a random point from the whole space 
        # or a point from the archive and perturb it heavily.
        
        while evals < budget:
            # Decide whether to do a local step or a restart step
            # 80% local, 20% restart
            if np.random.rand() < 0.2:
                # Restart: Pick a random point from the whole space
                x = lo + (hi - lo) * np.random.rand(dim)
            else:
                # Local: Pick a random point from archive and perturb
                if not archive:
                    x = lo + (hi - lo) * np.random.rand(dim)
                else:
                    idx = np.random.randint(len(archive))
                    x_base, _ = archive[idx]
                    # Perturb 1-3 dimensions
                    n_perturb = np.random.randint(1, 4)
                    indices = np.random.choice(dim, n_perturb, replace=False)
                    x = x_base.copy()
                    # Gaussian noise
                    sigma = 0.05
                    x[indices] += np.random.normal(0, sigma, n_perturb)
                    x = np.clip(x, lo, hi)
            
            f = np.array(problem(x))
            evals += 1
            add_to_archive(x, f)
            
        return [x for x, f in archive]
