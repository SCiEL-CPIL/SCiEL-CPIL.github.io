"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondominated_sort(self, F):
        """
        Perform non-dominated sorting on F (n x m array).
        Returns a list of lists of indices, where each inner list is a front.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of points dominating i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        if not front:
            return [[]]
            
        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            front = next_front
            
        return fronts

    def _crowding_distance(self, F, front_indices):
        """
        Compute crowding distance for points in front_indices.
        Returns array of distances for the points in front_indices.
        """
        if len(front_indices) <= 2:
            return np.full(len(front_indices), np.inf)
            
        m = len(front_indices)
        distances = np.zeros(m)
        F_front = F[front_indices]
        n_obj = F.shape[1]
        
        for k in range(n_obj):
            sorted_idx = np.argsort(F_front[:, k])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            f_min = F_front[sorted_idx[0], k]
            f_max = F_front[sorted_idx[-1], k]
            if f_max > f_min:
                for j in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[j]] += (F_front[sorted_idx[j+1], k] - F_front[sorted_idx[j-1], k]) / (f_max - f_min)
                    
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Phase 1: Initial sampling with Sobol-like space-filling sequence
        # Use a deterministic quasi-random sequence for initial exploration
        n_init = 50  # Number of initial points
        # Generate points using a simple Halton sequence or random with stratification
        # For simplicity and robustness, use random but with stratification per dimension
        X_init = np.random.uniform(lo, hi, size=(n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        evaluations = n_init
        
        # Keep all evaluated points
        archive_X = X_init.copy()
        archive_F = F_init.copy()
        
        # Phase 2: Iterative refinement using local search on non-dominated points
        # We will perform local searches (Nelder-Mead style) on promising points
        # until budget is exhausted or deadline reached
        
        # Define a local search function
        def local_search(x0, f0, max_evals=5, step_size=0.1):
            """
            Perform a simple local search around x0.
            Returns best point found.
            """
            nonlocal evaluations
            best_x = x0.copy()
            best_f = f0.copy()
            
            # Simple coordinate-wise perturbation
            for _ in range(max_evals):
                if evaluations >= self.budget or time.time() > deadline:
                    break
                
                # Try small random perturbations
                for trial in range(3):
                    if evaluations >= self.budget or time.time() > deadline:
                        break
                    
                    # Random direction in [-step_size, step_size]
                    direction = np.random.uniform(-step_size, step_size, dim)
                    x_new = best_x + direction
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    evaluations += 1
                    
                    # Check if x_new is non-dominated by best_f
                    # If f_new is better in at least one objective and not worse in any
                    if np.all(f_new <= best_f) and np.any(f_new < best_f):
                        best_x = x_new
                        best_f = f_new
                        break  # Found improvement, restart perturbations
                    elif np.all(f_new >= best_f) and np.any(f_new > best_f):
                        pass  # Worse, ignore
                    else:
                        # Non-dominated relationship, keep it in mind but don't update best
                        # For simplicity, we only track one best per local search
                        pass
                        
            return best_x, best_f
        
        # Main loop: select non-dominated points and refine them
        while evaluations < self.budget and time.time() < deadline:
            # Get non-dominated fronts from archive
            fronts = self._nondominated_sort(archive_F)
            
            if not fronts or not fronts[0]:
                break
                
            # Select points from the first front to refine
            # Choose up to 5 points with high crowding distance
            front1 = fronts[0]
            if len(front1) > 5:
                # Use crowding distance to select diverse points
                cd = self._crowding_distance(archive_F, front1)
                # Select top 5 by crowding distance
                top5_idx = np.argsort(cd)[::-1][:5]
                selected_idx = [front1[i] for i in top5_idx]
            else:
                selected_idx = front1[:]
                
            # For each selected point, perform local search
            for idx in selected_idx:
                if evaluations >= self.budget or time.time() > deadline:
                    break
                
                x0 = archive_X[idx]
                f0 = archive_F[idx]
                
                # Perform local search
                x_best, f_best = local_search(x0, f0, max_evals=8, step_size=0.05)
                
                # Add to archive if non-dominated
                # Check if f_best is non-dominated by archive
                is_dominated = False
                for i in range(len(archive_F)):
                    if np.all(archive_F[i] <= f_best) and np.any(archive_F[i] < f_best):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    archive_X = np.vstack([archive_X, [x_best]])
                    archive_F = np.vstack([archive_F, [f_best]])
                    
                    # Remove points dominated by f_best
                    new_mask = np.ones(len(archive_F), dtype=bool)
                    for i in range(len(archive_F)):
                        if i == len(archive_F) - 1:
                            continue
                        if np.all(f_best <= archive_F[i]) and np.any(f_best < archive_F[i]):
                            new_mask[i] = False
                    if not np.all(new_mask):
                        archive_X = archive_X[new_mask]
                        archive_F = archive_F[new_mask]
                        
            # Also add some random points to maintain exploration
            if evaluations < self.budget and time.time() < deadline:
                n_rand = min(5, self.budget - evaluations)
                if n_rand > 0:
                    X_rand = np.random.uniform(lo, hi, size=(n_rand, dim))
                    F_rand = np.array([problem(x) for x in X_rand])
                    evaluations += n_rand
                    
                    # Add non-dominated points to archive
                    for i in range(n_rand):
                        f_new = F_rand[i]
                        is_dominated = False
                        for j in range(len(archive_F)):
                            if np.all(archive_F[j] <= f_new) and np.any(archive_F[j] < f_new):
                                is_dominated = True
                                break
                        if not is_dominated:
                            archive_X = np.vstack([archive_X, [X_rand[i]]])
                            archive_F = np.vstack([archive_F, [f_new]])
                            # Remove dominated points
                            new_mask = np.ones(len(archive_F), dtype=bool)
                            for j in range(len(archive_F)):
                                if j == len(archive_F) - 1:
                                    continue
                                if np.all(f_new <= archive_F[j]) and np.any(f_new < archive_F[j]):
                                    new_mask[j] = False
                            if not np.all(new_mask):
                                archive_X = archive_X[new_mask]
                                archive_F = archive_F[new_mask]
                                
        return archive_X
# EVOLVE-BLOCK-END
