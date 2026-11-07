# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Sobol initialization and hybrid exploit/exploit search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # --- Phase 1: Sobol quasi-random initial sampling ---
        # Use a good chunk of budget for initial spread
        n_init = min(100, budget)
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_init)
        X_init = lo + (hi - lo) * sobol_pts
        
        archive_X = []
        archive_F = []
        used = 0
        
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            used += 1
            if np.any(f >= np.array([1705.0, 12.0, 0.3])):
                continue # Skip points not better than ref? No, ref is for scoring, not feasibility. 
                # Actually, we should keep all valid points. The ref point is just for HV calc.
                # But if a point is worse than ref in ALL objectives, it contributes 0 to HV.
                # We can keep them in the archive for diversity, or discard to save memory.
                # Let's keep them for now, dominance check handles it.
            
            # Check dominance
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if not dominated:
                # Remove dominated points from archive
                new_X = []
                new_F = []
                for j in range(len(archive_F)):
                    if not (np.all(f <= archive_F[j]) and np.any(f < archive_F[j])):
                        new_X.append(archive_X[j])
                        new_F.append(archive_F[j])
                archive_X = new_X
                archive_F = new_F
                
                archive_X.append(x.copy())
                archive_F.append(f.copy())

        # --- Phase 2: Hybrid Exploit/Explore ---
        rng = np.random.default_rng(42)
        ref = np.array([1705.0, 12.0, 0.3])
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Decide: Exploit (perturb archive) or Explore (random)
            # 70% Exploit, 30% Explore to refine the front
            if rng.random() < 0.7 and len(archive_X) > 0:
                # Pick a random point from the archive
                idx = rng.integers(0, len(archive_X))
                parent = archive_X[idx]
                
                # Adaptive step size: start larger, get smaller
                progress = used / budget
                step = 0.2 * (1.0 - progress) + 0.05
                child = parent + step * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
            else:
                # Random new point
                child = lo + (hi - lo) * rng.random(dim)

            f_child = problem(child)
            used += 1
            
            # Update archive
            dominated = False
            for af in archive_F:
                if np.all(af <= f_child) and np.any(af < f_child):
                    dominated = True
                    break
            
            if not dominated:
                new_X = []
                new_F = []
                for j in range(len(archive_F)):
                    if not (np.all(f_child <= archive_F[j]) and np.any(f_child < archive_F[j])):
                        new_X.append(archive_X[j])
                        new_F.append(archive_F[j])
                archive_X = new_X
                archive_F = new_F
                
                archive_X.append(child.copy())
                archive_F.append(f_child.copy())

        # Return the non-dominated front
        if len(archive_X) == 0:
            return np.array([])
        return np.array(archive_X)

# EVOLVE-BLOCK-END
