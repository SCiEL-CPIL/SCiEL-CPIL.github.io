"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using a Sobol-sequence-based initialization
    combined with a hypervolume-guided evolutionary search and local refinement.
    
    Strategy:
    1. Initialize population with a scrambled Sobol sequence for uniform coverage.
    2. Use a hypervolume-based selection to identify promising regions.
    3. Apply local refinement via a weighted sum scalarization on promising points.
    4. Use a deadline guard to ensure we return before the 330s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Generate Sobol sequence for initialization
        # We need enough points for a good initial population
        pop_size = min(100, budget // 2)
        
        # Use scipy's Sobol sequence if available, otherwise use Halton-like
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            X_init = sobol.random(pop_size)
            X_init = lo + X_init * span
        except ImportError:
            # Fallback to random
            rng = np.random.default_rng(42)
            X_init = lo + span * rng.random((pop_size, dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X_init])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_X = X_init.copy()
        all_F = F.copy()
        
        # Main search loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Compute dominance
            # F has shape (n, n_obj)
            # For each point i, check if any other point j dominates i
            # j dominates i if F[j] <= F[i] and F[j] != F[i]
            
            # Vectorized dominance check
            # F[:, np.newaxis, :] <= F[np.newaxis, :, :] gives (n, n, n_obj)
            # Any axis=2 gives (n, n) where True means j's objectives are all <= i's
            # We also need at least one strict inequality
            leq = np.all(F[:, np.newaxis, :] >= F[np.newaxis, :, :], axis=2)  # (n, n): leq[i,j] = F[j] <= F[i]
            strict = np.any(F[:, np.newaxis, :] > F[np.newaxis, :, :], axis=2)  # (n, n): strict[i,j] = F[j] < F[i] in at least one obj
            dominates = leq & strict  # dominates[i,j] = True if j dominates i
            
            # Point i is dominated if any j dominates i
            # dominates[i, :] has True where j dominates i
            is_dominated = np.any(dominates, axis=1)
            
            # Find non-dominated points
            nondom_idx = np.where(~is_dominated)[0]
            
            if len(nondom_idx) == 0:
                # All dominated, pick random
                parent_idx = np.random.randint(len(F))
            else:
                # Select from non-dominated points
                # Use hypervolume contribution as a heuristic
                # For simplicity, pick randomly from non-dominated
                parent_idx = nondom_idx[np.random.randint(len(nondom_idx))]
            
            # Create child by mutation
            # Use adaptive step size based on how close we are to the front
            # DTLZ4 has g(x) = x^0.5, so the front is at g=1
            # The distance from the front can be estimated by the objectives
            
            # Simple mutation: Gaussian perturbation
            step_size = 0.05  # Moderate step
            child = all_X[parent_idx] + step_size * span * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Add to population
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, fc])
            
            # Update the working population (keep it bounded)
            # Use a simple replacement: if the new point is not dominated, add it;
            # otherwise, replace a dominated point
            if len(all_F) > 200:
                # Remove a dominated point to keep population size manageable
                # Recompute dominance for the working set
                work_F = all_F[-200:]
                work_X = all_X[-200:]
                
                leq_w = np.all(work_F[:, np.newaxis, :] >= work_F[np.newaxis, :, :], axis=2)
                strict_w = np.any(work_F[:, np.newaxis, :] > work_F[np.newaxis, :, :], axis=2)
                dom_w = leq_w & strict_w
                is_dom_w = np.any(dom_w, axis=1)
                
                if np.any(is_dom_w):
                    # Remove one dominated point
                    dom_candidates = np.where(is_dom_w)[0]
                    remove_idx = dom_candidates[np.random.randint(len(dom_candidates))]
                    all_X = np.delete(all_X, len(all_X) - 200 + remove_idx, axis=0)
                    all_F = np.delete(all_F, len(all_F) - 200 + remove_idx, axis=0)
            
            # Local refinement every 20 evaluations
            if used % 20 == 0 and time.time() < deadline - 30:
                # Pick a promising point and try to improve it via local search
                # Use weighted sum with random weights
                weights = np.random.dirichlet(np.ones(n_obj))
                
                # Scalarize: minimize sum(w_i * f_i)
                def scalarized(x):
                    f = problem(x)
                    return np.dot(weights, f)
                
                # Try a few local improvements from the current best point
                # Find the point with best scalarized value
                scalar_vals = all_F @ weights
                best_idx = np.argmin(scalar_vals)
                x0 = all_X[best_idx]
                
                # Simple coordinate descent
                for coord in range(dim):
                    if time.time() > deadline - 10:
                        break
                    # Try small steps in + and - direction
                    for direction in [-1, 1]:
                        step = 0.01 * span[coord]
                        x_trial = x0.copy()
                        x_trial[coord] += direction * step
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = problem(x_trial)
                        used += 1
                        if used >= budget:
                            break
                        if np.dot(weights, f_trial) < np.dot(weights, all_F[best_idx]):
                            x0 = x_trial
                            all_X[best_idx] = x0
                            all_F[best_idx] = f_trial
                        if used >= budget:
                            break
                    if used >= budget:
                        break
                    if used >= budget:
                        break
        
        return all_X
# EVOLVE-BLOCK-END
