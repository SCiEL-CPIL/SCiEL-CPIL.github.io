# EVOLVE-BLOCK-START
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
        deadline = start_time + 280  # Safety margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo
        
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        pop_size = 30
        if self.budget < pop_size:
            pop_size = self.budget
            
        # 1. Initialization using Latin Hypercube Sampling (LHS)
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            intervals = np.linspace(lo[d], hi[d], pop_size + 1)
            for i in range(pop_size):
                X[i, d] = rng.uniform(intervals[i], intervals[i+1])
                
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Helper: Calculate hypervolume for 3 objectives
        def calc_hv(F_set, ref):
            # Filter points strictly better than ref
            mask = np.all(F_set < ref, axis=1)
            F_set = F_set[mask]
            if len(F_set) == 0:
                return 0.0
            
            # Filter non-dominated
            nd_pts = []
            for i in range(len(F_set)):
                dominated = False
                for j in range(len(F_set)):
                    if i == j: continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_pts.append(F_set[i])
            
            if not nd_pts: return 0.0
            
            # Sort by x1 ascending
            nd_pts.sort(key=lambda p: p[0])
            
            hv = 0.0
            # Process from largest x1 to smallest
            active_2d = [] # List of (x2, x3)
            
            def calc_hv_2d_bounded(pts, bx2, bx3, r2, r3):
                adjusted = []
                for x2, x3 in pts:
                    ax2 = max(x2, bx2)
                    ax3 = max(x3, bx3)
                    if ax2 < r2 and ax3 < r3:
                        adjusted.append((ax2, ax3))
                
                if not adjusted: return 0.0
                
                adjusted.sort(key=lambda p: p[0])
                hv = 0.0
                max_x3 = 0.0
                prev_x2 = 0.0
                for x2, x3 in adjusted:
                    if x3 > max_x3:
                        hv += (x2 - prev_x2) * (r3 - max_x3)
                        max_x3 = x3
                        prev_x2 = x2
                hv += (r2 - prev_x2) * (r3 - max_x3)
                return hv

            for i in range(len(nd_pts) - 1, -1, -1):
                p = nd_pts[i]
                x1_next = nd_pts[i+1][0] if i < len(nd_pts) - 1 else ref[0]
                width = x1_next - p[0]
                
                covered_area = calc_hv_2d_bounded(active_2d, p[1], p[2], ref[1], ref[2])
                total_area = (ref[1] - p[1]) * (ref[2] - p[2])
                unique_area = total_area - covered_area
                
                hv += width * unique_area
                active_2d.append((p[1], p[2]))
                
            return hv

        # Main Loop: Hybrid MOEA/D + DE
        # We use a decomposition-based approach where we maintain a population
        # and use weight vectors to guide local search, but also use DE for global exploration.
        
        # Generate weight vectors (simplified Das-Dennis or just uniform on simplex)
        # For 3 objectives, we can generate points on the 2D simplex.
        n_weights = 20
        weights = []
        # Simple grid on simplex
        steps = 5
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                weights.append(np.array([i, j, k]) / steps)
        
        # Shuffle weights to mix them with the population
        rng.shuffle(weights)
        
        # We will use a mix of DE and local search guided by weights.
        # To keep it simple and robust, we will primarily use DE but use 
        # hypervolume contribution for selection, and occasionally do a local polish.
        
        # Adaptive step size
        step_size = 0.5 * np.max(range_vec)
        
        # We will run until evals < budget - 20 (reserve for polish)
        polish_budget = 20
        main_budget = self.budget - polish_budget
        
        while evals < main_budget:
            if time.time() > deadline: break
            
            # Identify non-dominated in current population
            non_dom_idx = []
            for i in range(pop_size):
                is_dom = False
                for j in range(pop_size):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_idx.append(i)
            
            if len(non_dom_idx) == 0:
                parent_idx = np.argmin(np.sum(F, axis=1))
            else:
                # Pick random non-dominated
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                
            parent = X[parent_idx]
            
            # DE Mutation: v = x_r1 + F * (x_r2 - x_r3)
            # Select 3 distinct indices from the current population X
            idx1, idx2, idx3 = rng.choice(len(X), 3, replace=False)
            
            # Adaptive F: start with 0.8, reduce over time?
            # For small budget, constant 0.8 is usually fine.
            F_factor = 0.8
            
            mutant = X[idx1] + F_factor * (X[idx2] - X[idx3])
            mutant = np.clip(mutant, lo, hi)
            
            fc = problem(mutant)
            evals += 1
            
            # Selection:
            # 1. Does mutant dominate any point in X?
            # 2. Is mutant non-dominated by X?
            
            # Efficient dominance check
            # Check if mutant is dominated by any in X
            dominated = False
            for i in range(len(X)):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    dominated = True
                    break
            
            if not dominated:
                # Remove points in X that are dominated by mutant
                keep_indices = []
                for i in range(len(X)):
                    if not (np.all(fc <= F[i]) and np.any(fc < F[i])):
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
                    F[replace_idx] = fc
                else:
                    # Mutant is dominated. Do nothing? Or maybe keep it if it's close?
                    # In small budgets, exploration is key. But DE provides that.
                    # We can ignore dominated mutants to focus on the front.
                    pass
            else:
                # Mutant is dominated. Do nothing? Or maybe keep it if it's close?
                # In small budgets, exploration is key. But DE provides that.
                # We can ignore dominated mutants to focus on the front.
                pass

        # Phase 2: Local Polish
        # Use remaining budget
        if evals < self.budget:
            # Get non-dominated points
            non_dom_idx = []
            for i in range(pop_size):
                is_dom = False
                for j in range(pop_size):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_idx.append(i)
            
            # Sort by sum of objectives (best first)
            non_dom_idx.sort(key=lambda i: np.sum(F[i]))
            
            # Polish top 5
            for idx in non_dom_idx[:5]:
                if evals >= self.budget: break
                if time.time() > deadline: break
                
                x0 = X[idx]
                
                def scalarized(x):
                    nonlocal evals
                    if evals >= self.budget:
                        return 1e10
                    if time.time() > deadline:
                        return 1e10
                    f = problem(x)
                    evals += 1
                    return np.sum(f)
                
                try:
                    res = minimize(
                        scalarized, x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 5, 'maxfun': 10}
                    )
                    # The minimize function calls scalarized, which increments evals.
                    # We need to check if the result is better.
                    # The result.x is the polished point.
                    # We need to evaluate it once more to get the vector?
                    # No, scalarized returned the sum, but we didn't store the vector.
                    # We should store the best vector found during minimization?
                    # Or just evaluate res.x again.
                    if evals < self.budget and time.time() < deadline:
                        new_f = problem(res.x)
                        evals += 1
                        # Check if new_f is better than F[idx]
                        # Simple check: if new_f dominates F[idx] or has better sum
                        if np.all(new_f <= F[idx]) or np.sum(new_f) < np.sum(F[idx]):
                            X[idx] = res.x
                            F[idx] = new_f
                except:
                    pass

        return X

# EVOLVE-BLOCK-END
