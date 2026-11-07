import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive of non-dominated solutions
        archive = []
        evals_used = 0
        max_evals = self.budget
        
        # 1. Initialization: Random sampling
        pop_size = 20
        X_init = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        evals_used += pop_size
        
        for i in range(pop_size):
            self._add_to_archive(archive, X_init[i], F_init[i])
            
        # 2. Main Loop: Differential Evolution with Dominance-based Selection
        # We use a smaller effective population for selection to keep it fast
        # We will generate candidates based on the current archive or a subset of it
        
        # To avoid O(N^2) dominance checks on a growing archive, 
        # we maintain a fixed-size population P and an archive A.
        # Actually, with 200 evals, O(N^2) on N=200 is 40k ops, which is fast.
        
        F = F_init.copy()
        X = X_init.copy()
        
        # Sort initial population by non-domination
        # We'll just use the archive for parent selection
        
        while evals_used < max_evals:
            if time.time() > deadline:
                break
                
            # Select parents from the current best (non-dominated) set
            # If archive is empty (shouldn't happen), use X
            if not archive:
                parents = X
            else:
                # Use the archive points as potential parents
                # To ensure diversity, we might mix in some random or current pop points
                # But for convergence, archive is better.
                # Let's use a mix: 50% from archive, 50% from current pop X
                # Actually, let's just use the current population X for DE, 
                # but update X with non-dominated points.
                
                # Let's maintain a population of size pop_size
                # If archive grows, we sample from it.
                pass
            
            # DE Mutation: v = x_r1 + F * (x_r2 - x_r3)
            # Select 3 distinct indices from the current population X
            idx1, idx2, idx3 = rng.choice(len(X), 3, replace=False)
            
            # Adaptive F: start with 0.8, reduce over time?
            # For small budget, constant 0.8 is usually fine.
            F_factor = 0.8
            
            mutant = X[idx1] + F_factor * (X[idx2] - X[idx3])
            mutant = np.clip(mutant, lo, hi)
            
            f_mutant = problem(mutant)
            evals_used += 1
            
            # Selection:
            # 1. Does mutant dominate any point in X?
            # 2. Is mutant non-dominated by X?
            
            # Efficient dominance check
            # Check if mutant is dominated by any in X
            dominated = False
            for i in range(len(X)):
                if np.all(F[i] <= f_mutant) and np.any(F[i] < f_mutant):
                    dominated = True
                    break
            
            if not dominated:
                # Remove points in X that are dominated by mutant
                keep_indices = []
                for i in range(len(X)):
                    if not (np.all(f_mutant <= F[i]) and np.any(f_mutant < F[i])):
                        keep_indices.append(i)
                
                # If we removed points, we have space. If not, we replace the worst.
                if len(keep_indices) < len(X):
                    # Replace the first removed index or append if we are growing?
                    # We keep X fixed size. Replace the worst point among those removed or the overall worst.
                    # Simpler: Replace the point that mutant dominates. If multiple, replace one.
                    # If none dominated, replace the point with the worst hypervolume contribution (approx: max sum).
                    
                    # Let's find an index to replace.
                    # If we removed some, pick one of the removed.
                    removed_indices = [i for i in range(len(X)) if i not in keep_indices]
                    if removed_indices:
                        replace_idx = removed_indices[0]
                    else:
                        # Mutant is non-dominated but doesn't dominate anyone.
                        # Replace the worst point in X (e.g., highest sum of objectives)
                        replace_idx = np.argmax(np.sum(F, axis=1))
                    
                    X[replace_idx] = mutant
                    F[replace_idx] = f_mutant
                    
                    # Update Archive
                    self._add_to_archive(archive, mutant, f_mutant)
            else:
                # Mutant is dominated. Do nothing? Or maybe keep it if it's close?
                # In small budgets, exploration is key. But DE provides that.
                # We can ignore dominated mutants to focus on the front.
                pass
                
        # 3. Final Polish: Local Search on the best points
        # Use remaining budget for local refinement
        if archive and evals_used < max_evals:
            # Sort archive by "quality" (e.g., sum of objectives or distance to ideal)
            # We'll pick the top 3 non-dominated points
            # Simple heuristic: minimize sum of objectives
            sorted_arch = sorted(archive, key=lambda p: np.sum(p[1]))
            best_points = sorted_arch[:3]
            
            for point, f_val in best_points:
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break
                
                # Local optimization using Nelder-Mead on a scalarized objective?
                # Multi-objective local search is tricky.
                # Instead, we can do a small random walk around the best points.
                # Or use a simple gradient-free local search.
                
                # Let's do a few random perturbations around the best point
                for _ in range(5):
                    if evals_used >= max_evals:
                        break
                    if time.time() > deadline:
                        break
                    
                    noise = 0.05 * (hi - lo)
                    new_point = point + rng.standard_normal(self.dim) * noise
                    new_point = np.clip(new_point, lo, hi)
                    
                    new_f = problem(new_point)
                    evals_used += 1
                    
                    self._add_to_archive(archive, new_point, new_f)
                    
        return X

    def _add_to_archive(self, archive, x, f):
        """
        Add a point to the archive if it is non-dominated.
        Remove any points dominated by the new point.
        """
        # Check if new point is dominated by existing
        for ax, af in archive:
            if np.all(af <= f) and np.any(af < f):
                return # Dominated, do not add
        
        # Remove points dominated by new point
        new_archive = []
        for ax, af in archive:
            if not (np.all(f <= af) and np.any(f < af)):
                new_archive.append((ax, af))
        
        new_archive.append((x, f))
        archive.clear()
        archive.extend(new_archive)
