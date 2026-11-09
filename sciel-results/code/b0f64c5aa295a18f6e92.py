"""Multi-Objective Optimizer for RE21 using Nelder-Mead Direct Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization: Deterministic Sobol-like sequence for uniform coverage
        # We use a simple halving strategy or a low-discrepancy sequence approximation.
        # For dim=4, 32 points is a good start.
        n_init = 32
        X_init = self._sobol_like(n_init, self.dim, lo, hi)
        
        # Evaluate initial points
        F_init = np.zeros((n_init, self.n_obj))
        for i in range(n_init):
            F_init[i] = problem(X_init[i])
            
        X_archive = [X_init]
        F_archive = [F_init]
        n_evals = n_init
        
        # Helper to flatten archive
        def get_archive():
            if not X_archive:
                return np.array([]).reshape(0, self.dim), np.array([]).reshape(0, self.n_obj)
            X_all = np.vstack(X_archive)
            F_all = np.vstack(F_archive)
            return X_all, F_all
        
        # Helper to update archive with new point
        def add_point(x, f):
            X_archive.append(x.reshape(1, -1))
            F_archive.append(f.reshape(1, -1))
            nonlocal n_evals
            n_evals += 1
            
        # 2. Nelder-Mead Multi-Objective Loop
        # Strategy:
        # - Maintain a population of "simplexes" or just active search points.
        # - For each active point, try to move it towards a better objective value 
        #   using a Nelder-Mead-like step in the objective space, mapped back to x-space?
        #   No, Nelder-Mead works in x-space.
        # - We will maintain a set of candidate x-points.
        # - At each step, pick a point from the archive (preferably non-dominated).
        # - Generate neighbors around it.
        # - If a neighbor is non-dominated and improves the local HV contribution, keep it.
        
        # To be efficient, we will run a series of local Nelder-Mead optimizations
        # starting from different points on the current front.
        
        # We have 200 - 32 = 168 evals left.
        # A Nelder-Mead run in 4D typically needs ~10-20 evals to converge.
        # So we can afford ~8-10 local optimizations.
        
        # We will select 10 starting points from the initial front, spread out.
        
        # Get initial non-dominated set
        X_all, F_all = get_archive()
        nd_idx = self._get_non_dominated_indices(F_all)
        
        if len(nd_idx) < 10:
            # If front is small, just take all
            start_indices = nd_idx
        else:
            # Spread out the starting points to cover the front
            # Sort by first objective
            sorted_idx = nd_idx[np.argsort(F_all[nd_idx, 0])]
            # Take every k-th point to ensure spread
            step = max(1, len(sorted_idx) // 10)
            start_indices = sorted_idx[::step]
            if len(start_indices) < 10:
                # Fill with random non-dominated points
                remaining = np.setdiff1d(nd_idx, start_indices)
                while len(start_indices) < 10 and len(remaining) > 0:
                    start_indices = np.append(start_indices, remaining[0])
                    remaining = remaining[1:]
        
        # For each start point, run a local Nelder-Mead-like search
        for start_idx in start_indices:
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x0 = X_all[start_idx].copy()
            
            # Run a simplified Nelder-Mead optimization
            # We will manually implement a few steps of Nelder-Mead to control budget
            # Standard NM:
            # 1. Create simplex: x0, x0 + delta*e_i
            # 2. Evaluate all vertices
            # 3. Reflect, Expand, Contract, Shrink
            
            # Since we don't have gradients, we use a fixed initial step size.
            # Let's use a small fraction of the box size.
            step_size = 0.1 * span
            
            # Simplex vertices
            simplex = [x0]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] += step_size[i]
                simplex.append(xi)
            
            # Evaluate initial simplex
            # x0 is already evaluated? No, we need to be careful.
            # x0 is in the archive, so we know its f.
            # But NM needs a consistent set.
            # Let's just evaluate the new vertices.
            
            # Actually, a simpler and more robust approach for multi-objective with limited budget:
            # "Local search via random restarts and local perturbation"
            # Or: "Differential Evolution" with small population.
            
            # Let's stick to a robust Local Search:
            # 1. Take x0.
            # 2. Generate 5 random neighbors within a small radius.
            # 3. Evaluate them.
            # 4. If any is non-dominated, add to archive.
            # 5. Repeat 3 times (15 evals per start point).
            
            # This is safer than full NM which might diverge or waste evals.
            # Let's do 3 iterations of local perturbation per start point.
            # 3 iterations * 5 neighbors = 15 evals.
            # 10 start points * 15 evals = 150 evals.
            # Total: 32 + 150 = 182. Safe.
            
            current_x = x0
            for iter in range(3):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate neighbors
                # Use a Gaussian distribution centered at current_x
                # Scale by a fraction of the box
                sigma = 0.05 * span
                neighbors = []
                for _ in range(5):
                    x_new = current_x + np.random.normal(0, sigma)
                    # Clip to bounds
                    x_new = np.clip(x_new, lo, hi)
                    neighbors.append(x_new)
                
                # Evaluate neighbors
                best_neighbor = None
                best_f = None
                for x_new in neighbors:
                    if n_evals >= self.budget:
                        break
                    f_new = problem(x_new)
                    n_evals += 1
                    
                    # Check if non-dominated
                    X_curr, F_curr = get_archive()
                    if self._is_non_dominated(f_new, F_curr):
                        # Add to archive
                        add_point(x_new, f_new)
                        
                        # Update current_x to this new point if it's better in some sense?
                        # For multi-objective, we just keep the latest non-dominated point as the center for next iter?
                        # Or just keep x0?
                        # Let's keep the center as the point that was just added, to explore locally.
                        current_x = x_new
                        break # Found one, move to next iteration? Or keep looking?
                        # Let's keep looking to find the best local move, but we only have budget.
                        # Actually, adding all non-dominated neighbors is better.
                        # But we want to guide the search.
                        # Let's just add all non-dominated ones and pick the first one as new center.
                
                # If no neighbor was non-dominated, we might shrink the step size or stop.
                # For simplicity, if no improvement, we stop this local search thread.
                if best_neighbor is None and not any(self._is_non_dominated(problem(x), self._get_archive()[1]) for x in neighbors[:0]):
                    # This check is expensive and redundant. 
                    # Let's just track if we added anything.
                    pass

        return X_all

    def _sobol_like(self, n, dim, lo, hi):
        """Generate points using a simple Halving/Sobol approximation."""
        # For simplicity, use a deterministic grid or Halton sequence.
        # Halton sequence is easy to implement and low discrepancy.
        pts = np.zeros((n, dim))
        for i in range(dim):
            base = self._primes(i + 2) # 3, 5, 7, 11
            for j in range(n):
                f = 1.0
                r = 0.0
                i_val = j + 1
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                pts[j, i] = r
        return lo + pts * (hi - lo)

    def _primes(self, n):
        """Return the n-th prime number (1-indexed)."""
        if n == 1: return 2
        if n == 2: return 3
        if n == 3: return 5
        if n == 4: return 7
        # Hardcoded for dim=4 (primes 3, 5, 7, 11)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        return primes[n-1]

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_nd[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if not is_nd[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_nd[i] = False
                    break
        return np.where(is_nd)[0]

    def _is_non_dominated(self, f_new, F_archive):
        """Check if f_new is non-dominated with respect to F_archive."""
        for f in F_archive:
            if np.all(f <= f_new) and np.any(f < f_new):
                return False
        return True

# EVOLVE-BLOCK-END
