"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential-evolution-style multi-objective optimizer with archive-guided search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 310  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # LHS-style initialization for better spread
        pop_size = min(40, max(20, budget // 5))
        
        # Generate LHS points
        init_points = []
        for i in range(pop_size):
            # LHS: divide each dimension into pop_size cells, pick one per cell
            x = np.zeros(dim)
            for d in range(dim):
                cell = (i + d) % pop_size
                frac = (cell + rng.random()) / pop_size
                x[d] = lo[d] + frac * span[d]
            init_points.append(x)
        
        X = np.array(init_points)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive management
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by any archive member
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove archive members dominated by new point
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X = new_X
            archive_F = new_F

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # DE parameters
        CR = 0.7
        F_mut_start = 0.5
        F_mut_end = 0.9

        while used < budget:
            if time.time() - start_time > DEADLINE:
                break

            # Adaptive mutation scale
            progress = used / budget
            F_mut = F_mut_start + (F_mut_end - F_mut_start) * progress

            # Select parents: mix of Archive (exploitation) and Population (exploration)
            use_archive = len(archive_X) >= 3 and rng.random() < 0.7
            if use_archive:
                idxs = rng.choice(len(archive_X), size=3, replace=False)
                x1 = archive_X[idxs[0]].copy()
                x2 = archive_X[idxs[1]].copy()
                x3 = archive_X[idxs[2]].copy()
            elif len(archive_X) >= 1:
                a_idx = rng.integers(len(archive_X))
                x1 = archive_X[a_idx].copy()
                p_idxs = rng.choice(pop_size, size=2, replace=False)
                x2 = X[p_idxs[0]].copy()
                x3 = X[p_idxs[1]].copy()
            else:
                p_idxs = rng.choice(pop_size, size=3, replace=False)
                x1 = X[p_idxs[0]].copy()
                x2 = X[p_idxs[1]].copy()
                x3 = X[p_idxs[2]].copy()

            # DE/rand/1 mutation
            mutant = x1 + F_mut * (x2 - x3)
            mutant = np.clip(mutant, lo, hi)

            # Select base vector from population
            base_idx = rng.integers(pop_size)

            # Binomial crossover
            j_rand = rng.integers(dim)
            trial = X[base_idx].copy()
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1

            # Update archive
            update_archive(trial, f_trial)

            # Population replacement
            r_idx = rng.integers(pop_size)
            X[r_idx] = trial
            F[r_idx] = f_trial

            # Periodic diversity injection
            if used % 25 == 0 and used < budget - 5:
                if time.time() - start_time > DEADLINE:
                    break
                for k in range(2):
                    if used >= budget:
                        break
                    rand_x = lo + span * rng.random(dim)
                    rand_f = np.array(problem(rand_x))
                    used += 1
                    update_archive(rand_x, rand_f)
                    r_idx = rng.integers(pop_size)
                    X[r_idx] = rand_x
                    F[r_idx] = rand_f

        # Final local polish: use remaining budget for Nelder-Mead on archive points
        if len(archive_X) > 0 and time.time() - start_time < DEADLINE - 30:
            # Sort archive by sum of objectives (approximate quality)
            archive_scores = [np.sum(af) for af in archive_F]
            archive_order = np.argsort(archive_scores)
            
            # Polish top 5 archive points with Nelder-Mead for each objective
            n_polish = min(5, len(archive_X))
            polish_budget_per_obj = max(10, (budget - used) // (n_polish * self.n_obj))
            
            for i in range(n_polish):
                if time.time() - start_time > DEADLINE - 10:
                    break
                if used >= budget:
                    break
                
                base_x = archive_X[archive_order[i]].copy()
                
                for obj_idx in range(self.n_obj):
                    if used >= budget:
                        break
                    if time.time() - start_time > DEADLINE - 10:
                        break
                    
                    # Define objective for this single objective
                    def obj_func(x, obj_i=obj_idx):
                        fx = problem(x)
                        return fx[obj_i]
                    
                    # Nelder-Mead optimization
                    try:
                        res = minimize(
                            obj_func,
                            base_x,
                            method='Nelder-Mead',
                            options={'maxiter': min(polish_budget_per_obj, 50), 'xatol': 1e-4, 'fatol': 1e-6}
                        )
                        # Count evaluations used by Nelder-Mead
                        n_evals = res.nfev
                        used += n_evals
                        
                        # Evaluate and add to archive
                        f_res = np.array(problem(res.x))
                        update_archive(res.x, f_res)
                        
                        # Also try perturbing in other objective directions
                        for other_obj in range(self.n_obj):
                            if other_obj == obj_idx:
                                continue
                            if used >= budget:
                                break
                            
                            def multi_obj_func(x, primary=obj_idx, secondary=other_obj):
                                fx = problem(x)
                                return fx[primary] + 0.5 * fx[secondary]
                            
                            res2 = minimize(
                                multi_obj_func,
                                base_x,
                                method='Nelder-Mead',
                                options={'maxiter': min(polish_budget_per_obj // 2, 25), 'xatol': 1e-4, 'fatol': 1e-6}
                            )
                            used += res2.nfev
                            f_res2 = np.array(problem(res2.x))
                            update_archive(res2.x, f_res2)
                    except Exception:
                        pass

        return X
# EVOLVE-BLOCK-END
