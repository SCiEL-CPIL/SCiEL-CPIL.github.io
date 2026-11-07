# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Phase 1: Deterministic space-filling initialization (Halton sequence).
      2. Phase 2: Iterative greedy selection in objective space to maximize coverage.
      3. Phase 3: Local Polish using Nelder-Mead on diverse front points.
      4. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a fixed seed for the internal RNG to ensure reproducibility of the search strategy
        # The problem function itself is deterministic given x, so this is safe.
        rng = np.random.RandomState(42)

        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = np.array(problem(x), dtype=float)
            eval_count += 1
            return fx

        def is_dominated(f, other_f):
            """Returns True if f is dominated by other_f."""
            return np.all(other_f <= f) and np.any(other_f < f)

        def is_non_dominated(f, population_f):
            """Returns True if f is not dominated by any point in population_f."""
            for pf in population_f:
                if is_dominated(f, pf):
                    return False
            return True

        def halton_sequence(n, dim, primes):
            """Generate n points in dim dimensions using Halton sequence."""
            points = np.zeros((n, dim))
            for d in range(dim):
                prime = primes[d]
                for i in range(n):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= prime
                        i_val, remainder = divmod(i_val, prime)
                        points[i, d] += f * remainder
            return points

        # --- Phase 1: Deterministic Space-Filling Initialization ---
        
        # Use Halton sequence for initial sampling
        n_init = 30
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        if n_init > budget:
            n_init = budget
            
        halton_points = halton_sequence(n_init, dim, primes)
        # Map to bounds
        population_x = lo + halton_points * (hi - lo)
        
        population_f = []
        
        for i in range(n_init):
            f = evaluate(population_x[i])
            if f is None:
                break
            population_f.append(f)
        
        population_f = np.array(population_f)
        
        # --- Phase 2: Iterative Greedy Selection in Objective Space ---
        # Continue adding points to maximize coverage of the objective space.
        # We will maintain a list of all evaluated points (x, f) and select the next
        # candidate from a pool of potential candidates.
        
        # Generate a pool of candidate points using a coarse grid or random sampling
        # For efficiency, we'll use random points and select the one that maximizes
        # the minimum distance to existing points in objective space.
        
        # However, generating random points on the fly and evaluating them is expensive.
        # Instead, we can use a "batch" approach: generate a batch of candidates,
        # evaluate the most promising one (based on distance to current front),
        # and repeat.
        
        # Let's use a simpler strategy:
        # 1. Generate a large set of random candidates (e.g., 100).
        # 2. For each candidate, estimate its "potential" based on distance to current front.
        #    But we don't know the objective values yet.
        # 3. So we must evaluate them.
        
        # Alternative: Use a surrogate-free approach where we evaluate points
        # that are far from existing points in DECISION space? No, the directive
        # says "objective space".
        
        # Let's stick to the directive: "maximizing the minimum Euclidean distance
        # to all previously evaluated points in the objective space".
        
        # This requires knowing the objective values of the new point, which requires evaluation.
        # So the strategy is:
        # 1. Generate a set of candidate points.
        # 2. Evaluate them.
        # 3. Select the one that is furthest from the current set in objective space.
        # 4. Add it to the set.
        # 5. Repeat.
        
        # To avoid wasting evaluations on points that are likely dominated or close,
        # we can use a heuristic: evaluate points that are far in decision space first,
        # or use a simple grid.
        
        # Given the small dimension (4), a grid is feasible.
        # Let's use a 5x5x5x5 grid (625 points) as candidates.
        # But we only have 200 budget. So we can't evaluate all.
        
        # Let's use a random sampling approach with a "batch size" of 10.
        # In each batch, generate 10 random points, evaluate them, and pick the best
        # one (max min distance in obj space) to keep.
        
        n_batches = (budget - eval_count) // 10
        if n_batches < 0:
            n_batches = 0
            
        for batch in range(n_batches):
            if eval_count >= budget or time.time() > deadline:
                break
                
            # Generate 10 random candidates
            batch_size = 10
            candidates_x = rng.random((batch_size, dim))
            candidates_x = lo + candidates_x * (hi - lo)
            
            best_candidate_idx = -1
            best_min_dist = -1
            best_candidate_f = None
            best_candidate_x = None
            
            for k in range(batch_size):
                if eval_count >= budget or time.time() > deadline:
                    break
                    
                x = candidates_x[k]
                f = evaluate(x)
                if f is None:
                    continue
                
                # Calculate min distance to existing points in objective space
                if len(population_f) > 0:
                    dists = np.linalg.norm(population_f - f, axis=1)
                    min_dist = np.min(dists)
                else:
                    min_dist = 0.0
                
                # We prefer points that are far from existing points
                # But we also don't want points that are dominated.
                # Let's just use distance for now.
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate_idx = k
                    best_candidate_f = f
                    best_candidate_x = x
            
            if best_candidate_idx != -1:
                population_x = np.vstack([population_x, best_candidate_x])
                population_f = np.vstack([population_f, best_candidate_f])

        # --- Phase 3: Local Polish ---
        
        # Extract current Pareto Front
        front_indices = []
        for i in range(len(population_f)):
            if is_non_dominated(population_f[i], population_f):
                front_indices.append(i)
        
        if not front_indices:
            # Fallback if something went wrong
            return lo

        # Select diverse points from the front for polishing
        # We want to polish points that are likely to be on the "knee" or extremes.
        # Simple selection: Pick the point with min obj0, min obj1, min obj2, and the one with min sum.
        candidates_to_polish = []
        
        if len(front_indices) > 0:
            # Min each objective
            for obj_idx in range(n_obj):
                min_val = float('inf')
                min_idx = -1
                for idx in front_indices:
                    if population_f[idx][obj_idx] < min_val:
                        min_val = population_f[idx][obj_idx]
                        min_idx = idx
                if min_idx != -1 and min_idx not in candidates_to_polish:
                    candidates_to_polish.append(min_idx)
            
            # Min sum (balanced)
            min_sum = float('inf')
            min_sum_idx = -1
            for idx in front_indices:
                s = np.sum(population_f[idx])
                if s < min_sum:
                    min_sum = s
                    min_sum_idx = idx
            if min_sum_idx != -1 and min_sum_idx not in candidates_to_polish:
                candidates_to_polish.append(min_sum_idx)

            # If we have very few front points, just take all
            if len(candidates_to_polish) < 3 and len(front_indices) > len(candidates_to_polish):
                for idx in front_indices:
                    if idx not in candidates_to_polish:
                        candidates_to_polish.append(idx)
                    if len(candidates_to_polish) >= 3:
                        break

        # Polish up to 3-4 points with Nelder-Mead
        # Budget remaining
        remaining_budget = budget - eval_count
        if remaining_budget > 0:
            # Allocate roughly equal budget to each polish run
            n_polish = min(4, len(candidates_to_polish))
            if n_polish > 0:
                evals_per_polish = max(5, remaining_budget // n_polish)
                
                for idx in candidates_to_polish[:n_polish]:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    
                    x0 = population_x[idx].copy()
                    
                    # Define a scalar objective that tries to minimize the distance to the ideal point 
                    # or a weighted sum. Since we don't know the ideal, minimizing the sum is a good heuristic 
                    # for the "balanced" region, but for extremes, we should minimize the specific objective?
                    # Nelder-Mead minimizes a single function.
                    # To improve the Pareto front, we want to move the point along the front or towards the ideal.
                    # A common trick is to minimize the Tchebycheff distance or just the sum.
                    # Let's minimize the sum of objectives. This tends to pull points towards the "knee".
                    # To cover extremes, we already picked min-obj points. Polishing them with sum-minimization
                    # might move them away from the extreme. 
                    # Better: For the point that is min-obj-k, minimize obj-k.
                    # For the balanced point, minimize sum.
                    
                    # Let's assign a specific weight vector based on the point's characteristics
                    # If this point was selected as min-obj-k, use weight [0,0,0,1] for obj k.
                    # If balanced, use [1,1,1].
                    
                    # Simpler: Just use a weighted sum where weights are inverse of the current objective values?
                    # Or just minimize the sum. It's a safe bet for hypervolume improvement in the center.
                    # Let's use a weighted sum where the weight for the objective this point is best at is higher.
                    
                    f_current = population_f[idx]
                    # Normalize weights to sum to 1
                    # If f is [0.1, 0.5, 0.5], weights should emphasize obj 0? 
                    # Actually, to improve HV, we want to push the point "inwards" (lower values).
                    # Minimizing sum is generally good.
                    
                    def objective(x):
                        nonlocal eval_count
                        if time.time() > deadline:
                            return 1e10
                        if eval_count >= budget:
                            return 1e10
                        fx = np.array(problem(x), dtype=float)
                        eval_count += 1
                        return np.sum(fx)
                    
                    try:
                        res = minimize(objective, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                        
                        if res.success or res.fun < 1e9:
                            x_best = np.clip(res.x, lo, hi)
                            # Re-evaluate to get exact f and ensure it's recorded
                            f_best = evaluate(x_best)
                            if f_best is not None:
                                # We don't strictly need to update the population array for scoring, 
                                # as the evaluator records every call. 
                                # But updating it helps if we do further steps (we don't).
                                pass
                    except Exception:
                        pass

        # Return any valid point (ignored by evaluator, but must return something)
        if population_x is not None and len(population_x) > 0:
            return population_x[0]
        return lo

# EVOLVE-BLOCK-END
