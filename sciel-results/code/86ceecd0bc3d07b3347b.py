import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        def time_check():
            return time.time() < deadline

        # 1. Initialization: Latin Hypercube Sampling
        pop_size = 50
        if self.budget <= pop_size:
            pop_size = self.budget
            
        X = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            perm = np.random.permutation(pop_size)
            X[perm, i] = np.linspace(0, 1, pop_size)
        X = lo + X * range_
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # 2. Weight Generation for MOEA/D
        # Use a uniform grid on the simplex for 3 objectives
        # n_points per dimension on the grid
        n_grid = 10 
        weights = []
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                k = n_grid - i - j
                weights.append([i / n_grid, j / n_grid, k / n_grid])
        
        # If we need more weights (unlikely for 3 obj with n_grid=10 -> 66 weights), 
        # but let's ensure we have a good set.
        # Actually, for 3 objectives, a grid of size 10 gives 66 points.
        # We have 50 individuals. This is fine.
        weights = np.array(weights)
        n_weights = len(weights)
        
        # Assign each individual to the best weight (min weighted sum)
        S = F @ weights.T
        best_w_idx = np.argmin(S, axis=1)
        
        # 3. Main Loop
        # Use a mix of exploitation and exploration
        while evals < self.budget and time_check():
            # Select a weight to focus on
            # Weighted random selection based on number of individuals assigned? 
            # Or just uniform random. Uniform is simpler and effective.
            w_idx = np.random.randint(n_weights)
            w = weights[w_idx]
            
            # Select parents
            # Calculate weighted sum scores for all individuals for this specific weight
            ws_scores = F @ w
            
            # Selection: Tournament or Best-2
            # Let's use a mix: 50% best 2, 50% random 2 (for diversity)
            if np.random.rand() < 0.5:
                # Exploitation
                parent_indices = np.argsort(ws_scores)[:2]
            else:
                # Exploration
                parent_indices = np.random.choice(pop_size, 2, replace=False)
            
            p1 = X[parent_indices[0]]
            p2 = X[parent_indices[1]]
            
            # SBX Crossover
            eta_c = 20.0
            child = np.zeros_like(p1)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(p1[i] - p2[i]) < 1e-14:
                        child[i] = p1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                        child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                else:
                    if abs(p1[i] - p2[i]) < 1e-14:
                        child[i] = p1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                        child[i] = 0.5 * ((1 + beta) * p2[i] + (1 - beta) * p1[i])
                        
            # Polynomial Mutation
            eta_m = 20.0
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    u = np.random.rand()
                    # Avoid division by zero if hi==lo
                    if hi[i] == lo[i]:
                        continue
                    delta1 = (1 + 2 * (child[i] - lo[i]) / (hi[i] - lo[i])) ** (eta_m + 1)
                    delta2 = (1 + 2 * (hi[i] - child[i]) / (hi[i] - lo[i])) ** (eta_m + 1)
                    if u <= 0.5:
                        deltaq = 1 / delta1
                    else:
                        deltaq = 1 - 1 / delta2
                    child[i] = child[i] + (hi[i] - lo[i]) * (2 * u - 1) * deltaq
                    
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Update Population
            # Find the individual in the population that is "worst" for the selected weight w
            # Standard MOEA/D: Replace the individual in the subproblem if the new one is better.
            # Here, we identify the subproblem (weight) that the child is best suited for.
            child_scores = fc @ weights.T
            best_child_w = np.argmin(child_scores)
            
            # Find incumbents for this weight
            incumbents = np.where(best_w_idx == best_child_w)[0]
            
            if len(incumbents) > 0:
                # Pick the worst incumbent for this weight
                incumbent_scores = S[incumbents, best_child_w]
                worst_incumbent_local = np.argmax(incumbent_scores)
                worst_incumbent_idx = incumbents[worst_incumbent_local]
                
                # Check if child is better
                if fc @ weights[best_child_w] < S[worst_incumbent_idx, best_child_w]:
                    X[worst_incumbent_idx] = child
                    F[worst_incumbent_idx] = fc
                    S[worst_incumbent_idx, :] = fc @ weights.T
                    best_w_idx[worst_incumbent_idx] = best_child_w
            else:
                # If no incumbent (shouldn't happen if weights cover space well), replace global worst
                # Or just pick a random one
                idx_replace = np.random.randint(pop_size)
                X[idx_replace] = child
                F[idx_replace] = fc
                S[idx_replace, :] = fc @ weights.T
                best_w_idx[idx_replace] = best_child_w

        # 4. Polish Phase
        # Use remaining budget to polish the best non-dominated points
        # Find non-dominated set
        def non_dominated_indices(F_set):
            n = len(F_set)
            nd_indices = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        nd_indices = non_dominated_indices(F)
        
        if evals < self.budget and len(nd_indices) > 0 and time_check():
            # Select a few points to polish
            # We have budget - evals evaluations left
            polish_budget = self.budget - evals
            # Sort non-dominated by their best weighted sum score to pick "best" ones
            # Or just pick the first few
            points_to_polish = nd_indices[:min(5, len(nd_indices))]
            
            for idx in points_to_polish:
                if evals >= self.budget or not time_check():
                    break
                
                x_curr = X[idx].copy()
                f_curr = F[idx].copy()
                
                # Define a scalar objective based on a weight vector
                # Use the weight vector that this point was best for
                w_polish = weights[best_w_idx[idx]]
                
                # Simple Local Search: Coordinate Descent with adaptive step
                step_size = 0.1 * range_
                improved = True
                max_iter = 10
                iter_count = 0
                
                while improved and evals < self.budget and time_check() and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    for i in range(self.dim):
                        if hi[i] == lo[i]:
                            continue
                        for delta in [1, -1]:
                            x_trial = x_curr.copy()
                            x_trial[i] += delta * step_size[i]
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            # Check if this is a new point (avoid re-evaluating same point)
                            # Simple check: if it's exactly the same as current, skip
                            if np.allclose(x_trial, x_curr):
                                continue
                                
                            f_trial = problem(x_trial)
                            evals += 1
                            
                            if evals >= self.budget or not time_check():
                                break
                                
                            # Check if better in terms of weighted sum
                            if f_trial @ w_polish < f_curr @ w_polish:
                                x_curr = x_trial
                                f_curr = f_trial
                                improved = True
                                # Shrink step size
                                step_size *= 0.5
                                
                            if evals >= self.budget or not time_check():
                                break
                        if evals >= self.budget or not time_check():
                            break
                    if evals >= self.budget or not time_check():
                        break
                
                # Update population with polished point
                X[idx] = x_curr
                F[idx] = f_curr
                S[idx, :] = f_curr @ weights.T
                best_w_idx[idx] = np.argmin(S[idx, :])

        # 5. Return Non-Dominated Set
        # Re-calculate non-dominated set from final F
        nd_indices_final = non_dominated_indices(F)
        
        # If no non-dominated points found (should not happen), return all
        if not nd_indices_final:
            nd_indices_final = list(range(len(F)))
            
        result_X = X[nd_indices_final]
        return result_X
