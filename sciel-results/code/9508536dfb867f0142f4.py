# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DTLZ1-specific multi-objective optimizer.
    
    Strategy:
    1. Exploit knowledge of DTLZ1 structure: convergence variables (x_1..x_{d-1}) should be 0.5,
       and the diversity variable x_d parameterizes the Pareto front.
    2. Use stratified sampling on the simplex for x_d to ensure good spread.
    3. Perform local search to refine convergence variables towards 0.5.
    4. Use deadline guard to avoid timeout.
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
        
        # DTLZ1 structure: x_1..x_{d-1} are convergence vars, x_d is diversity var
        # Pareto front: x_1..x_{d-1} = 0.5, x_d varies in [0,1]
        # Objectives: f1 = 0.5*g*(1+z), f2 = 0.5*g*(1-z), f3 = 0.5*g
        # where g = sum_{i=1}^{d-1}(x_i - 0.5)^2, z = x_d
        
        n_convergence = self.dim - 1
        n_diversity = 1
        
        # Phase 1: Stratified sampling on the Pareto front structure
        # Use Latin hypercube-like sampling for x_d (diversity variable)
        # and set convergence variables to 0.5 initially
        
        n_phase1 = min(40, self.budget // 5)
        if n_phase1 < 5:
            n_phase1 = 5
        
        # Stratified sampling for x_d in [0, 1]
        x_d_values = np.linspace(0.0, 1.0, n_phase1 + 2)[1:-1]  # Avoid exact 0 and 1
        # Add some jitter for robustness
        x_d_values += (hi[self.dim-1] - lo[self.dim-1]) * 0.02 * (np.random.rand(n_phase1) - 0.5)
        x_d_values = np.clip(x_d_values, lo[self.dim-1], hi[self.dim-1])
        
        X = np.zeros((n_phase1, self.dim))
        # Set convergence variables to midpoint of bounds (assumed to be 0.5 for [0,1])
        mid_convergence = (lo[:n_convergence] + hi[:n_convergence]) / 2.0
        X[:, :n_convergence] = mid_convergence
        X[:, n_convergence] = x_d_values
        
        F = np.array([problem(x) for x in X])
        used = n_phase1
        
        # Phase 2: Local search to refine convergence variables
        # For each point, try to improve by moving convergence variables towards 0.5
        
        n_phase2 = min(self.budget - used, 100)
        
        # Sort by sum of objectives to prioritize worst points
        order = np.argsort(np.sum(F, axis=1))
        
        for idx in range(n_phase2):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            point_idx = order[idx % len(order)]
            x = X[point_idx].copy()
            
            # Try small perturbations on convergence variables
            best_x = x
            best_f = F[point_idx].copy()
            best_sum = np.sum(best_f)
            
            n_trials = min(8, self.budget - used)
            for t in range(n_trials):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                # Perturb convergence variables
                perturbation = 0.05 * (hi[:n_convergence] - lo[:n_convergence]) * np.random.randn(n_convergence)
                x_trial = x.copy()
                x_trial[:n_convergence] += perturbation
                x_trial = np.clip(x_trial, lo, hi)
                
                f_trial = np.array(problem(x_trial))
                used += 1
                
                # Check if it's non-dominated or better in sum
                if np.sum(f_trial) < best_sum:
                    best_x = x_trial
                    best_f = f_trial
                    best_sum = np.sum(best_f)
            
            # Update if improved
            if np.sum(best_f) < np.sum(F[point_idx]):
                X[point_idx] = best_x
                F[point_idx] = best_f
                order = np.argsort(np.sum(F, axis=1))
        
        # Phase 3: Use remaining budget for random exploration with exploitation
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Mix of random and guided search
            n_random = min(remaining // 3, 20)
            n_guided = remaining - n_random
            
            for i in range(n_random):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                x = lo + (hi - lo) * np.random.rand(self.dim)
                f = np.array(problem(x))
                used += 1
                
                # Replace worst point if this is non-dominated
                if self._is_non_dominated(f, F):
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    X[worst_idx] = x
                    F[worst_idx] = f
            
            for i in range(n_guided):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                # Select a non-dominated parent
                non_dom = self._find_non_dominated_indices(F)
                if len(non_dom) > 0:
                    parent_idx = non_dom[np.random.randint(len(non_dom))]
                else:
                    parent_idx = np.random.randint(len(X))
                
                x = X[parent_idx].copy()
                # Small perturbation
                x += 0.03 * (hi - lo) * np.random.randn(self.dim)
                x = np.clip(x, lo, hi)
                
                f = np.array(problem(x))
                used += 1
                
                if self._is_non_dominated(f, F):
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    X[worst_idx] = x
                    F[worst_idx] = f
        
        return X
    
    def _is_non_dominated(self, f, F):
        """Check if f is non-dominated by any point in F."""
        for fi in F:
            if np.all(fi <= f) and np.any(fi < f):
                return False
        return True
    
    def _find_non_dominated_indices(self, F):
        """Find indices of non-dominated points in F."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]
# EVOLVE-BLOCK-END
