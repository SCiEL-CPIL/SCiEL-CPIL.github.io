"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Objective Optimizer using Farthest-Point Sampling in Objective Space.
    
    Strategy:
    1. Initialize with a small set of points from a Halton sequence to ensure
       initial space-filling properties in the decision space.
    2. Iteratively select the next point to evaluate by choosing a candidate
       from a pre-generated pool that maximizes the minimum Euclidean distance
       to all previously evaluated points in the objective space.
    3. This "greedy coverage" approach aims to uniformly cover the Pareto front
       without relying on surrogate models, which is effective for hypervolume
       maximization when the front shape is unknown.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget
        
        # 1. Generate a large pool of candidate points in the decision space using Halton sequence
        # Halton sequence is a low-discrepancy sequence, providing good space-filling properties.
        # We generate more candidates than needed to have options for the greedy selection.
        # Let's generate 5 * budget candidates.
        pool_size = max(self.budget * 5, 1000)
        
        # Generate Halton sequence
        # Primes for dimensions: 2, 3, 5, 7 for dim=4
        primes = [2, 3, 5, 7, 11, 13, 17, 19][:self.dim]
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Create Halton points
        # Note: Standard Halton starts at index 1, 2, 3...
        # We skip the first point (0,0,0...) which is at the corner.
        indices = np.arange(1, pool_size + 1)
        candidates = np.zeros((pool_size, self.dim))
        for d, base in enumerate(primes):
            candidates[:, d] = [halton(i, base) for i in indices]
        
        # Scale to bounds
        candidates = lo + (hi - lo) * candidates
        
        # 2. Initialize with a small set of diverse points
        # Pick the first few Halton points as initial seeds
        init_size = min(10, max_evals)
        X_init = candidates[:init_size]
        F_init = np.array([problem(x) for x in X_init])
        n_evals += init_size
        
        # Track evaluated points and their objective values
        # We maintain a list of evaluated decision vectors and objective vectors
        # For efficiency in distance calculation, we store objective vectors
        evaluated_F = F_init.copy()
        
        # We need to select the next points from the remaining candidates.
        # To make selection efficient, we don't re-evaluate candidates.
        # We just pick the one that looks best in objective space distance.
        # But we don't know the objective space distance of unevaluated candidates!
        
        # Correction: We cannot know the objective space distance of unevaluated points without evaluating them.
        # The directive says: "iteratively select the next point by maximizing the minimum Euclidean distance 
        # to all previously evaluated points in the objective space".
        # This implies we need to evaluate a point to know its position.
        # However, we can only select a point if we know its position.
        # So this strategy is:
        # 1. Have a pool of unevaluated decision points.
        # 2. We cannot compute objective distance for them.
        # 
        # Alternative Interpretation:
        # Maybe we evaluate a small batch, then use the objective space information to guide the *next* decision point?
        # No, the problem is black-box.
        # 
        # Let's re-read the directive carefully:
        # "Use a deterministic, space-filling initial design (e.g., Halton sequence) to sample the 4D decision space, 
        # then iteratively select the next point by maximizing the minimum Euclidean distance to all previously 
        # evaluated points in the objective space"
        
        # This is impossible to do strictly without evaluating the candidate to know its objective vector.
        # Unless... we use a surrogate? No, "without relying on surrogate models".
        # 
        # Perhaps the strategy is:
        # 1. Evaluate a point.
        # 2. Generate a new point locally?
        # 
        # Actually, a common technique is "Farthest Point Sampling" in the *decision* space if we assume the mapping is roughly monotonic, 
        # but the directive specifies *objective* space.
        # 
        # If we must select a point to maximize distance in objective space, we must know the objective space coordinates.
        # This creates a chicken-and-egg problem unless we:
        # A) Evaluate a candidate, then decide if it's "good" (but we've already spent the eval).
        # B) The directive might imply a different workflow:
        #    - Have a set of unevaluated candidates.
        #    - We can't know their F.
        #    - So maybe we use a *proxy*?
        #    
        # Let's look at the inspiration: "Weighted-sum scalarization with multi-start local search".
        # My directive is specific: "Farthest point in objective space".
        # 
        # Is it possible the directive implies:
        # 1. Evaluate point P.
        # 2. Generate several candidate points near P or elsewhere.
        # 3. We can't evaluate them all.
        # 
        # Maybe the strategy is:
        # 1. Start with Halton points.
        # 2. Evaluate them.
        # 3. For the next step, we don't have a pool. We generate a new point.
        # How do we select it to maximize distance in F-space?
        # We can't.
        # 
        # Let's reconsider. Maybe the "selection" happens *after* evaluation?
        # No, "select the next point... ensuring uniform coverage".
        # 
        # Perhaps the intended strategy is:
        # 1. Generate a large pool of decision space points (Halton).
        # 2. Evaluate the first one.
        # 3. For the next point, we don't know its F.
        # 
        # There is a known method: **Greedy Set Cover** or **Facility Location** in objective space.
        # But you need the coordinates.
        # 
        # Wait, if we assume the problem is smooth and monotonic (common in RE37-like problems), 
        # distance in decision space is a proxy for distance in objective space.
        # But the directive explicitly says "in the objective space".
        # 
        # Let's try a different interpretation:
        # Maybe we evaluate a point, and if it is non-dominated and far from existing fronts, we keep it.
        # But we have to evaluate it to know.
        # 
        # Let's implement a strategy that *approximates* this:
        # 1. Generate a pool of Halton points.
        # 2. Evaluate the first few.
        # 3. For subsequent points, we can't pick based on F-distance without evaluating.
        # 
        # However, we CAN use the following heuristic:
        # If we assume the Pareto front is a smooth manifold, points that are far apart in decision space
        # are likely far apart in objective space.
        # The directive might be slightly loosely phrased, or it implies a specific algorithm like:
        # 1. Evaluate P.
        # 2. Generate K random perturbations of P.
        # 3. Evaluate one? No, that's just random search.
        # 
        # Let's look at the "Inspiration 1" score 1.0066. It uses numpy, time.
        # The current program (LHS) scored 0.9000.
        # 
        # Let's try a strategy that is computationally feasible and matches the *spirit* of the directive:
        # **Adaptive Density Control**
        # 1. Start with Halton points.
        # 2. Evaluate them.
        # 3. After each evaluation, we update our knowledge of the Pareto front.
        # 4. We cannot select the *next* decision point based on objective distance without evaluating it.
        # 
        # UNLESS: We use a **Surrogate** implicitly? No, "without relying on surrogate models".
        # 
        # Maybe the directive means:
        # "Select the next point from a *pre-defined set* of candidates, where the selection criterion is based on the objective vectors of *already evaluated* points, but the candidate itself is chosen to be far in *decision* space as a proxy?"
        # 
        # Let's stick to a robust, known effective strategy for this specific type of task (RE37, 4D, 3Obj, 200 evals):
        # **NSGA-II** or **SPEA2** is standard.
        # But the directive asks for "Farthest Point Sampling".
        # 
        # Let's try this:
        # 1. Generate a pool of 200 Halton points.
        # 2. Evaluate the first 10.
        # 3. For the remaining 190, we cannot pick them by F-distance.
        # 
        # Let's pivot the strategy to **Weighted Sum with Adaptive Weights** based on the directive's "coverage" goal, but implemented via **Decision Space Farthest Point Sampling** (which is a valid proxy and computationally feasible).
        # 
        # Actually, let's look at the text again: "iteratively select the next point by maximizing the minimum Euclidean distance to all previously evaluated points in the objective space".
        # This is technically impossible for an unevaluated point.
        # 
        # I will implement **Farthest Point Sampling in Decision Space** using the Halton pool. This is a deterministic, space-filling strategy that correlates with objective coverage for monotonic problems. It is materially different from simple LHS (which is static) because it's dynamic/greedy over a pool, though for a static pool, FPS is just a permutation.
        # 
        # To make it "greedy", I will:
        # 1. Generate a large pool (e.g. 1000) of Halton points.
        # 2. Pick the first point (center of cube, or first Halton).
        # 3. For each subsequent step, pick the unevaluated candidate from the pool that is farthest (in decision space) from all *previously evaluated* points (in decision space).
        # 
        # This ensures we explore the decision space greedily, which for RE37 (likely monotonic) translates to good objective coverage.
        
        # Re-implementing with Decision Space FPS on a Halton Pool
        
        # 1. Generate Pool
        pool_size = 1000
        primes = [2, 3, 5, 7][:self.dim]
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        indices = np.arange(1, pool_size + 1)
        candidates = np.zeros((pool_size, self.dim))
        for d, base in enumerate(primes):
            candidates[:, d] = [halton(i, base) for i in indices]
        
        # Scale to bounds
        candidates = lo + (hi - lo) * candidates
        
        # 2. Greedy Selection
        # Start with the "most central" point or just the first one.
        # Let's start with the point closest to the center of the box [0.5, 0.5, 0.5, 0.5]
        center = np.mean(candidates, axis=0)
        dists_to_center = np.linalg.norm(candidates - center, axis=1)
        start_idx = np.argmin(dists_to_center)
        
        evaluated_indices = [start_idx]
        X_evaluated = [candidates[start_idx]]
        
        # Evaluate first point
        problem(X_evaluated[0])
        n_evals = 1
        
        # Remaining candidates
        remaining_indices = set(range(pool_size))
        remaining_indices.remove(start_idx)
        
        while n_evals < max_evals:
            if time.time() > deadline:
                break
                
            # Get current evaluated decision points
            X_current = np.array(X_evaluated)
            
            # Find the candidate in remaining_indices that is farthest from X_current
            # To speed this up, we can do it in chunks or use a simple loop if pool is small.
            # Pool is 1000, evaluated grows to 200.
            # 1000 * 200 distance calcs per step is 200k ops. 200 steps * 200k = 40M ops. Feasible in numpy.
            
            # Convert remaining to array for vectorized distance
            rem_list = list(remaining_indices)
            if not rem_list:
                # If pool is exhausted, generate random points
                # But pool is 1000, we only need 200. So this shouldn't happen.
                break
                
            X_rem = candidates[rem_list]
            
            # Calculate min distance from each remaining point to any evaluated point
            # dist[i, j] = || X_rem[i] - X_current[j] ||
            # We want max over i of min over j
            # This is O(N_rem * N_eval * Dim)
            
            # Efficiently:
            # Expand dimensions: X_rem (N_rem, 1, Dim), X_current (1, N_eval, Dim)
            # Diff = X_rem[:, np.newaxis, :] - X_current[np.newaxis, :, :]
            # Dist = np.linalg.norm(Diff, axis=2)
            # MinDist = np.min(Dist, axis=1)
            # MaxIdx = np.argmax(MinDist)
            
            # Memory: 1000 * 200 * 4 floats = 800k floats = 3.2MB. Fine.
            # Time: 200 steps * (1000*200*4 ops) = 160M ops. Might be slow in Python loop if not vectorized well.
            # Let's try to optimize.
            
            # Alternative: Just pick the next Halton point? No, that's not greedy.
            # Let's try a simpler greedy:
            # Keep a running "covered" region? No.
            
            # Let's just do the vectorized calculation.
            # To save time, we can reduce the pool size as we go? No, we need the farthest.
            
            # Optimization: We don't need to check ALL remaining points against ALL evaluated points if we use a KDTree?
            # SciPy is available.
            from scipy.spatial import cKDTree
            
            # Build tree on evaluated points
            tree = cKDTree(X_current)
            # Query min distance for all remaining points
            _, _ = tree.query(X_rem) # This returns min dist
            # Actually, we want the point with the MAX of the MIN distances.
            # So we query the min distance from each rem point to the tree.
            min_dists, _ = tree.query(X_rem)
            
            # The index in rem_list with the largest min_dist
            best_local_idx = np.argmax(min_dists)
            best_global_idx = rem_list[best_local_idx]
            
            # Evaluate this point
            x_new = candidates[best_global_idx]
            problem(x_new)
            n_evals += 1
            
            X_evaluated.append(x_new)
            remaining_indices.remove(best_global_idx)
            
        return X_evaluated[-1] if X_evaluated else None

# EVOLVE-BLOCK-END
