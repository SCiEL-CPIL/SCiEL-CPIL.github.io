"""Multi-objective optimizer for RE37: Hybrid DE with Tchebycheff decomposition and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Multi-Objective Optimizer: DE + Tchebycheff + Local Polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use fixed seed for reproducibility within the run
        rng = np.random.default_rng(42)

        # --- Phase 1: DE with Archive ---
        pop_size = min(40, max(20, budget // 5))
        archive_max = 300

        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive as list of (x, f)
        archive = list(zip(X, F))

        # Helper: check if f is dominated by any in archive_F (array (n, n_obj))
        def is_dominated(f, archive_F):
            if len(archive_F) == 0:
                return False
            leq = np.all(archive_F <= f, axis=1)
            lt = np.any(archive_F < f, axis=1)
            return np.any(leq & lt)

        # Helper: update archive
        def update_archive(archive, x, f):
            if not archive:
                return [(x, f)]
            
            arch_F = np.array([a[1] for a in archive])
            
            # Check if new point is dominated
            if is_dominated(f, arch_F):
                return archive

            # Remove dominated points from archive
            new_archive = []
            for ax, af in archive:
                # Keep if new point does NOT dominate old point
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))

            new_archive.append((x, f))

            # Prune if too large
            if len(new_archive) > archive_max:
                F_arr = np.array([a[1] for a in new_archive])
                cd = np.zeros(len(new_archive))
                for m in range(n_obj):
                    vals = F_arr[:, m]
                    order = np.argsort(vals)
                    cd[order[0]] = np.inf
                    cd[order[-1]] = np.inf
                    val_range = vals[order[-1]] - vals[order[0]]
                    if val_range > 1e-12:
                        for i in range(1, len(order) - 1):
                            cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
                
                order = np.argsort(cd)
                remove_count = len(new_archive) - archive_max
                # Remove from end to keep indices valid
                for idx in sorted(order[:remove_count], reverse=True):
                    new_archive.pop(idx)

            return new_archive

        # DE Loop
        F_cr = 0.8
        F_weight = 0.5
        
        while used < budget:
            if time.time() > deadline:
                break

            # Get archive arrays for selection
            if archive:
                arch_X = np.array([a[0] for a in archive])
                arch_F = np.array([a[1] for a in archive])
            else:
                arch_X = np.empty((0, dim))
                arch_F = np.empty((0, n_obj))

            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Strategy: 50% DE/rand/1, 50% DE/best/1 (from archive)
                if rng.random() < 0.5:
                    r1, r2 = rng.choice(pop_size, size=2, replace=False)
                    base = X[r1]
                    diff = X[r2] - X[r1]
                else:
                    if len(arch_X) > 0:
                        best_idx = rng.integers(len(arch_X))
                        base = arch_X[best_idx]
                        r1, r2 = rng.choice(pop_size, size=2, replace=False)
                        diff = X[r1] - X[r2]
                    else:
                        r1, r2 = rng.choice(pop_size, size=2, replace=False)
                        base = X[r1]
                        diff = X[r2] - X[r1]

                v = base + F_weight * diff
                v = np.clip(v, lo, hi)

                # Binary crossover
                jrand = rng.integers(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if rng.random() <= F_cr or j == jrand:
                        trial[j] = v[j]

                f_new = problem(trial)
                used += 1

                archive = update_archive(archive, trial, f_new)

                # Selection
                if is_dominated(f_new, F[i:i+1]):
                    pass
                elif is_dominated(F[i], f_new.reshape(1, -1)):
                    X[i] = trial
                    F[i] = f_new
                else:
                    if rng.random() < 0.5:
                        X[i] = trial
                        F[i] = f_new

        # --- Phase 2: Tchebycheff Decomposition Polish ---
        # If we have time and budget left, refine the front
        # This phase uses a subset of the remaining budget
        remaining_budget = budget - used
        if remaining_budget > 10 and time.time() < deadline - 30:
            
            # Generate weight vectors on simplex
            # For 3 objectives, use a grid
            n_per_axis = 3 # Small grid to conserve budget
            weights = []
            for i in range(n_per_axis + 1):
                for j in range(n_per_axis + 1 - i):
                    k = n_per_axis - i - j
                    w = np.array([i, j, k], dtype=float) / n_per_axis
                    weights.append(w)
            weights = np.array(weights)
            M = len(weights)
            
            # Initialize subproblem solutions from archive (random selection)
            if len(archive) > 0:
                arch_X = np.array([a[0] for a in archive])
                arch_F = np.array([a[1] for a in archive])
                # Sample M points from archive
                indices = rng.choice(len(arch_X), size=M, replace=False)
                X_sub = arch_X[indices].copy()
                F_sub = arch_F[indices].copy()
            else:
                X_sub = lo + (hi - lo) * rng.random((M, dim))
                F_sub = np.array([problem(x) for x in X_sub])
                used += M

            # Ideal and Nadir
            ideal = np.min(F_sub, axis=0)
            nadir = np.max(F_sub, axis=0)
            
            def tchebycheff(f, w, ideal_pt, nadir_pt):
                span = nadir_pt - ideal_pt
                span[span < 1e-12] = 1.0
                f_norm = (f - ideal_pt) / span
                return np.max(w * f_norm)

            # Iterate
            polish_steps = min(remaining_budget // 2, 50) # Limit steps to avoid timeout
            for _ in range(polish_steps):
                if used >= budget - 10: # Keep 10 for safety
                    break
                if time.time() > deadline:
                    break

                # Pick a random subproblem
                idx = rng.integers(M)
                w = weights[idx]
                
                # Pick a neighbor for mutation
                # Simple: pick another random index
                idx2 = rng.integers(M)
                
                # Mutation
                mutant = X_sub[idx] + 0.2 * (X_sub[idx2] - X_sub[idx])
                mutant = np.clip(mutant, lo, hi)
                
                # Evaluate
                f_mut = problem(mutant)
                used += 1
                
                # Update ideal/nadir
                ideal = np.minimum(ideal, f_mut)
                nadir = np.maximum(nadir, f_mut)
                
                # Check if mutant is better for this subproblem
                old_val = tchebycheff(F_sub[idx], w, ideal, nadir)
                new_val = tchebycheff(f_mut, w, ideal, nadir)
                
                if new_val < old_val:
                    X_sub[idx] = mutant
                    F_sub[idx] = f_mut
                    # Update global archive
                    archive = update_archive(archive, mutant, f_mut)
                elif rng.random() < 0.1:
                    # Diversity injection
                    X_sub[idx] = mutant
                    F_sub[idx] = f_mut
                    archive = update_archive(archive, mutant, f_mut)

        # --- Phase 3: Local Polish (Nelder-Mead) on best archive points ---
        # If we still have budget, refine a few top points
        if len(archive) > 0:
            # Pick up to 5 points to polish
            arch_F = np.array([a[1] for a in archive])
            # Sort by sum of objectives (simple heuristic for "best")
            sums = np.sum(arch_F, axis=1)
            order = np.argsort(sums)[:5]
            
            for idx in order:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x0 = np.array(archive[idx][0])
                f0 = np.array(archive[idx][1])
                
                # Simple local search: random steps decreasing in size
                for step_size in [0.1, 0.05, 0.01]:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try 10 random perturbations
                    for _ in range(10):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        delta = rng.standard_normal(dim) * step_size
                        x_new = np.clip(x0 + delta, lo, hi)
                        f_new = problem(x_new)
                        used += 1
                        
                        # Check if dominates f0
                        if np.all(f_new <= f0) and np.any(f_new < f0):
                            x0 = x_new
                            f0 = f_new
                            archive = update_archive(archive, x0, f0)
                        elif np.all(f0 <= f_new) and np.any(f0 < f_new):
                            # f0 dominates, keep x0
                            pass
                        else:
                            # Non-dominated, add to archive
                            archive = update_archive(archive, x0, f0)
                            archive = update_archive(archive, x_new, f_new)

        return X
# EVOLVE-BLOCK-END
