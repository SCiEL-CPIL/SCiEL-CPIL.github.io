# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominates(a, b):
        """Check if vector a dominates vector b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 10.0  # safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Archive: list of (x, f) tuples for non-dominated points
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.asarray(f, dtype=float)
            # Check if new point is dominated by any archive point
            for i in range(len(archive_F)):
                if self._dominates(archive_F[i], f):
                    return
            # Remove archive points dominated by new point
            keep = []
            for i in range(len(archive_F)):
                if self._dominates(f, archive_F[i]):
                    continue
                keep.append(i)
            archive_F = [archive_F[i] for i in keep]
            archive_X = [archive_X[i] for i in keep]
            archive_F.append(f)
            archive_X.append(x)

        # Phase 1: Latin Hypercube Sampling for initial coverage
        # Use more points for better initial coverage
        n_init = max(30, min(100, budget // 3))
        
        # Generate LHS points with stratified sampling
        lxs = np.zeros((n_init, dim))
        for d in range(dim):
            intervals = np.linspace(lo[d], hi[d], n_init + 1)
            perm = rng.permutation(n_init)
            for i in range(n_init):
                row = perm[i]
                lxs[row, d] = intervals[i] + rng.random() * (intervals[i + 1] - intervals[i])
        
        # Evaluate initial population
        for i in range(n_init):
            if time.time() >= deadline:
                break
            x = lxs[i]
            f = problem(x)
            update_archive(x, f)

        # Phase 2: Differential Evolution with improved parameters
        remaining = budget - n_init
        
        # Larger population for better diversity
        pop_size = max(15, min(40, remaining // 3))
        
        # Initialize DE population from archive + random
        de_X = []
        de_F = []
        
        # Add best points from archive
        if archive_F:
            archive_idx = sorted(range(len(archive_F)), key=lambda i: np.sum(archive_F[i]))
            for i in archive_idx[:min(pop_size, len(archive_idx))]:
                de_X.append(archive_X[i].copy())
                de_F.append(archive_F[i].copy())
        
        # Fill rest with random points
        while len(de_X) < pop_size:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            update_archive(x, f)
            de_X.append(x.copy())
            de_F.append(f.copy())
        
        # Improved DE parameters for better exploration in 4D
        F_factor = 0.9  # Higher mutation scaling
        CR = 0.95  # Higher crossover rate
        
        # Main loop
        eval_count = n_init + len(de_X)
        
        while eval_count < budget:
            if time.time() >= deadline:
                break
            
            # Generate offspring
            idx = rng.choice(pop_size, size=3, replace=False)
            r1, r2, r3 = de_X[idx[0]], de_X[idx[1]], de_X[idx[2]]
            base = de_X[idx[2]]
            
            # Mutation
            mutant = base + F_factor * (r1 - r2)
            mutant = np.clip(mutant, lo, hi)
            
            # Crossover
            trial = np.zeros(dim)
            jrand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = base[j]
            
            # Evaluate
            f_trial = problem(trial)
            eval_count += 1
            update_archive(trial, f_trial)
            
            # Selection: replace the worst in the DE population
            sums = np.array([np.sum(f) for f in de_F])
            worst_idx = np.argmax(sums)
            
            if np.sum(f_trial) < sums[worst_idx]:
                de_X[worst_idx] = trial.copy()
                de_F[worst_idx] = f_trial.copy()
            
            # Periodically refresh DE population with archive points
            if eval_count % 40 == 0 and archive_F:
                archive_idx = sorted(range(len(archive_F)), key=lambda i: np.sum(archive_F[i]))
                n_refresh = min(8, len(archive_idx))
                for i in range(n_refresh):
                    de_X[i] = archive_X[archive_idx[i]].copy()
                    de_F[i] = archive_F[archive_idx[i]].copy()

        # Phase 3: Local refinement with multiple weight vectors and tighter tolerances
        if archive_F:
            archive_idx = sorted(range(len(archive_F)), key=lambda i: np.sum(archive_F[i]))
            n_refine = min(15, len(archive_idx), max(5, (budget - eval_count) // 3))
            
            # Structured weight grid for better coverage of the Pareto front
            weights_list = [
                np.ones(n_obj) / n_obj,
                np.array([0.6, 0.2, 0.2]),
                np.array([0.2, 0.6, 0.2]),
                np.array([0.2, 0.2, 0.6]),
                np.array([0.4, 0.35, 0.25]),
                np.array([0.25, 0.35, 0.4]),
                np.array([0.35, 0.4, 0.25]),
                np.array([0.45, 0.3, 0.25]),
                np.array([0.25, 0.45, 0.3]),
                np.array([0.3, 0.25, 0.45]),
            ]
            
            for i in range(n_refine):
                if eval_count >= budget:
                    break
                if time.time() >= deadline:
                    break
                
                x0 = archive_X[archive_idx[i]].copy()
                
                for w in weights_list:
                    if eval_count >= budget:
                        break
                    if time.time() >= deadline:
                        break
                    
                    def objective(x, w=w):
                        nonlocal eval_count
                        f = problem(x)
                        eval_count += 1
                        return np.dot(w, f)
                    
                    try:
                        result = minimize(objective, x0, method='Nelder-Mead',
                                        bounds=list(zip(lo, hi)),
                                        options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-5})
                        if result.fun < np.inf:
                            x_best = result.x
                            f_best = problem(x_best)
                            eval_count += 1
                            update_archive(x_best, f_best)
                    except:
                        pass

        # Return the archive
        if archive_X:
            return np.array(archive_X)
        return np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END
